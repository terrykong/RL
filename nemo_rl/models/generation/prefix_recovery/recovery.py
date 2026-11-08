# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Checkpoint-time cuts of live model calls, and resuming calls from them.

The host worker builds one ``PrefixRecovery`` (or ``NoPrefixRecovery`` when
prefix cuts are off) in ``setup_token_capture`` and calls only its public
methods. Nothing here touches a generation engine except through
``PrefixRecoveryParts.progress_reader``.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal

from nemo_rl.models.generation.prefix_recovery.parts import PrefixRecoveryParts
from nemo_rl.models.generation.prefix_recovery.restore import (
    _ends_inside_character,
    _generation_cut_blockers,
    restore_reply_text,
)
from nemo_rl.models.generation.prefix_recovery.state import (
    _CompletedCaptureState,
    _RequestCaptureState,
    _TokenCaptureSnapshotGate,
)

LOGGER = logging.getLogger(__name__)

# Completion evidence only bridges the short race between a terminal response
# and Gym freezing its admitted-call inventory. Time-based retention avoids
# evicting fresh evidence merely because a large deployment completed more
# than 100k calls, while still bounding idle-job memory growth.
_COMPLETED_CAPTURE_RETENTION_S = 60.0 * 60.0

# Why a model call was left uncut, one counter each. Fixed names only: no error
# message or call identity ever becomes a metric name.
GENERATION_CUT_FAILURE_REASONS = (
    # The worker has no live or finished record of the call, e.g. Gym admitted
    # it before its request reached this worker.
    "no_capture_or_completed_state",
    "completed_not_staged",
    "completed_missing_output_limit",
    "completed_missing_finish_reason",
    "uncuttable_request",
    "observation_error",
    "missing_output_limit",
    "zero_generated_tokens",
    "incomplete_utf8_boundary",
    # Staging, identity or any other error while cutting this one call.
    "cut_error",
    "control_deadline",
)


def initial_generation_cut_counts() -> dict[str, int]:
    """Zeroed cut counters, so every category is reported from the first step."""
    return {
        "completed_cuts_kept": 0,
        "cut_lifecycle_races": 0,
        # Reads of vLLM's decoding progress: how many, for how many calls, and
        # the total time spent waiting for and holding vLLM's output lock.
        "progress_reads": 0,
        "progress_read_calls": 0,
        "progress_read_wait_us": 0,
        "progress_read_hold_us": 0,
        **{f"failed_cuts_{reason}": 0 for reason in GENERATION_CUT_FAILURE_REASONS},
    }


# Reasons already logged by this worker process; calls sharing a setting log once.
_LOGGED_UNCUTTABLE_REASONS: set[str] = set()


def _record_terminal_reason(
    state: _RequestCaptureState, finish_reason: Any, stop_reason: Any
) -> None:
    """Record how vLLM finished a call; the caller holds ``state.lock``."""
    if finish_reason is None:
        return
    if finish_reason not in ("stop", "length"):
        state.observation_error = (
            f"unsupported terminal finish_reason {finish_reason!r}"
        )
    elif (
        state.terminal_finish_reason is not None
        and state.terminal_finish_reason != finish_reason
    ):
        state.observation_error = (
            "conflicting terminal finish reasons: "
            f"{state.terminal_finish_reason!r} and {finish_reason!r}"
        )
    elif not isinstance(stop_reason, (str, int, type(None))):
        state.observation_error = (
            "terminal stop_reason must be a string, integer, or None"
        )
    else:
        state.terminal_finish_reason = finish_reason
        state.terminal_stop_reason = stop_reason


def _require_inventory_identity(
    prefix: Any, worker_capture_key: str, *, call_kind: str
) -> None:
    """Refuse a cut whose inventory entry names another episode than the call."""
    # Deferred: nemo_gym is an optional extra absent in non-Gym runs.
    from nemo_gym.episode_types import EpisodeId

    expected = EpisodeId(rollout_id=prefix.rollout_id, attempt=prefix.attempt)
    if worker_capture_key != expected.capture_key:
        raise RuntimeError(
            "generation-prefix inventory identity does not match the "
            f"{call_kind} call: model_call_id={prefix.model_call_id!r}, "
            f"inventory_rollout_id={prefix.rollout_id!r}, "
            f"worker_capture_key={worker_capture_key!r}"
        )


class PrefixRecovery:
    """Cuts live calls at a checkpoint and resumes restored calls from their prefix."""

    def __init__(self, parts: PrefixRecoveryParts) -> None:
        self._parts = parts
        # Guards the completed-call evidence, receipts and counters below.
        self._lock = threading.Lock()
        self._completed: dict[str, _CompletedCaptureState] = {}
        self._receipts: dict[tuple[str, str], Any] = {}
        # Calls this process resumed from a restored generation prefix, and
        # their prefix tokens; the driver logs the sums each step.
        self._restored_calls = 0
        self._restored_tokens = 0
        self._counts = initial_generation_cut_counts()
        self._gate = _TokenCaptureSnapshotGate()
        # A dedicated executor prevents completions waiting on the closed gate
        # from consuming all threads needed to create the checkpoint cut.
        self._control_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="nrl-generation-cut-control"
        )
        # Fence closes get their own thread so a stale cut still staging rows
        # cannot delay the next checkpoint's fence past the driver's timeout.
        self._fence_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="nrl-token-capture-fence"
        )
        # An aborted call's chunk rows are cleared here, off the serving loop;
        # the clear waits at the snapshot gate like a terminal write.
        self._cleanup_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="nrl-generation-cut-cleanup"
        )

    @property
    def control_token(self) -> str | None:
        return self._parts.control_token

    # ---- Serving a call -------------------------------------------------

    def resolve_restore(
        self, admission: Any, prefix_token_ids: list[int]
    ) -> tuple[Any, Any | None]:
        """Resolve an admission's restored prefix, or drop it so the call regenerates.

        Gym has already consumed the restored cut when it sends this call, and
        generating the whole response from the turn boundary is always valid,
        so a prefix that cannot be fetched or does not match its admission
        costs time, not the rollout. Returns the admission to serve and the
        verified prefix snapshot, or the admission without its cut and None.
        """
        continuation = admission.generation_cut
        if continuation is None:
            return admission, None
        try:
            snapshot = self._rebuild_prefix(admission, prefix_token_ids)
            # Checked here, before the prefix is spliced into the prompt; Gym's
            # begin_call repeats it and would otherwise fail the call.
            if snapshot.digest != continuation.prefix_digest:
                raise RuntimeError("rebuilt prefix digest does not match the admission")
            generated = sum(mask == 1.0 for mask in snapshot.token_mask_delta)
            if generated != continuation.prefix_token_count:
                raise RuntimeError(
                    f"rebuilt prefix has {generated} generated tokens, "
                    f"admission expects {continuation.prefix_token_count}"
                )
            return admission, snapshot
        except Exception:  # noqa: BLE001 - any unusable prefix regenerates
            LOGGER.warning(
                "cannot restore the generation prefix of model call %s; "
                "regenerating it from its turn boundary",
                admission.model_call_id,
                exc_info=True,
            )
        # The regenerated call will not extend these rows, so they are retired
        # here instead of by its terminal write.
        self.retire_rows_later(
            admission.model_call_id, tuple(continuation.staging_keys)
        )
        return admission.model_copy(update={"generation_cut": None}), None

    def on_begin(self, admission: Any) -> None:
        """Count a call Gym's ``begin_call`` accepted with a restored cut."""
        if admission.generation_cut is None:
            return
        with self._lock:
            self._restored_calls += 1
            self._restored_tokens += admission.generation_cut.prefix_token_count

    def on_sampling_params(self, request: Any, sampling_params: Any) -> None:
        """Record vLLM's resolved output budget, and whether the call can be cut."""
        state = self._parts.registry.get(request)
        effective_output_limit = int(sampling_params.max_tokens)
        if effective_output_limit <= 0:
            raise ValueError("effective output limit must be positive")
        if state is not None:
            with state.lock:
                if state.effective_output_limit is None:
                    state.effective_output_limit = effective_output_limit
        # Such a call regenerates from its previous boundary after a restore,
        # which costs time but not correctness; see _generation_cut_blockers.
        blockers = _generation_cut_blockers(sampling_params, self._parts.tokenizer)
        if not blockers or state is None:
            return
        reason = f"{', '.join(blockers)} do not carry across a generation cut"
        with state.lock:
            state.uncuttable_reason = reason
        if reason not in _LOGGED_UNCUTTABLE_REASONS:
            _LOGGED_UNCUTTABLE_REASONS.add(reason)
            LOGGER.warning(
                "model calls using %s are not cut; they regenerate after a restore",
                ", ".join(blockers),
            )

    def observe(self, request: Any, request_output: Any) -> None:
        """Record the tokens a finished output adds past the last cut."""
        state = self._parts.registry.get(request)
        if state is None:
            return
        outputs = getattr(request_output, "outputs", None)
        if not outputs:
            return
        with state.lock:
            observed = state.observed_generation_token_count
        try:
            output = outputs[0]
            # Finished, so vLLM appends to these lists no more.
            generation_token_ids = list(getattr(output, "token_ids", None) or ())[
                observed:
            ]
            generation_logprobs = self._parts.progress_reader.logprobs(
                generation_token_ids, getattr(output, "logprobs", None), start=observed
            )
            finish_reason = getattr(output, "finish_reason", None)
            stop_reason = getattr(output, "stop_reason", None)
        except (RuntimeError, TypeError, ValueError) as error:
            with state.lock:
                state.observation_error = f"{type(error).__name__}: {error}"
            return
        with state.lock:
            state.observe(generation_token_ids, generation_logprobs, start=observed)
            _record_terminal_reason(state, finish_reason, stop_reason)

    def restore_reply_text(
        self, request: Any, request_output: Any, *, tokenizer: Any
    ) -> None:
        """Prepend a resumed call's prefix text to its final output."""
        state = self._parts.registry.get(request)
        if state is None or not state.resumed_generation_token_ids:
            return
        with state.lock:
            prefix_token_ids = list(state.resumed_generation_token_ids)
            engine_prompt_token_ids = list(state.prompt_token_ids)
        restore_reply_text(
            request,
            request_output,
            prefix_token_ids=prefix_token_ids,
            engine_prompt_token_ids=engine_prompt_token_ids,
            tokenizer=tokenizer,
        )

    # ---- Ending a call --------------------------------------------------

    def on_terminal(
        self,
        model_call_id: str,
        coords: Any,
        generation_token_count: int,
        *,
        effective_output_limit: int | None = None,
        terminal_finish_reason: Literal["stop", "length"] | None = None,
        terminal_stop_reason: str | int | None = None,
    ) -> None:
        """Keep a finished call's evidence for a cut racing its response."""
        now = time.monotonic()
        with self._lock:
            # Dict insertion order is completion order. Remove only entries old
            # enough that the response/inventory race has certainly elapsed;
            # never discard fresh evidence because the deployment crossed an
            # arbitrary call-count threshold.
            while self._completed:
                oldest_id = next(iter(self._completed))
                oldest = self._completed[oldest_id]
                if (
                    now - oldest.completed_at_monotonic
                    <= _COMPLETED_CAPTURE_RETENTION_S
                ):
                    break
                self._completed.pop(oldest_id)
            self._completed.pop(model_call_id, None)
            self._completed[model_call_id] = _CompletedCaptureState(
                coords=coords,
                generation_token_count=generation_token_count,
                effective_output_limit=effective_output_limit,
                terminal_finish_reason=terminal_finish_reason,
                terminal_stop_reason=terminal_stop_reason,
                completed_at_monotonic=now,
            )

    @contextlib.contextmanager
    def final_write(self) -> Iterator[None]:
        """Hold a permit so a terminal write never lands inside a snapshot."""
        self._gate.enter()
        try:
            yield
        finally:
            self._gate.exit()

    def retire_rows(self, model_call_id: str, staging_keys: tuple[str, ...]) -> None:
        """Delete an ended call's chunk rows; the caller holds a ``final_write`` permit.

        Once the call has ended no live reader needs them, and a committed
        snapshot that names them saved its own copy of TransferQueue, so only a
        snapshot still in progress must be waited out.
        """
        sink = self._parts.sink
        if not staging_keys or sink is None:
            return
        try:
            sink.clear(list(staging_keys))
        except Exception:  # noqa: BLE001 - the call has already ended
            LOGGER.exception(
                "failed to clear obsolete generation chunks for model call %s",
                model_call_id,
            )

    def retire_rows_later(
        self, model_call_id: str, staging_keys: tuple[str, ...]
    ) -> None:
        """Delete an ended call's chunk rows off the caller's thread, outside a snapshot."""
        if staging_keys:
            self._cleanup_executor.submit(
                self._retire_rows_after_fence, model_call_id, staging_keys
            )

    def _retire_rows_after_fence(
        self, model_call_id: str, staging_keys: tuple[str, ...]
    ) -> None:
        with self.final_write():
            self.retire_rows(model_call_id, staging_keys)

    # ---- Snapshot fence (Ray-facing through the worker) -----------------

    async def hold_final_writes(self) -> None:
        """Drain terminal writes and block new ones until ``release_final_writes``."""
        # Take the epoch on the loop, in call order with the matching release:
        # if the driver times out and releases first, the late close is a no-op.
        epoch = self._gate.begin_epoch()
        await asyncio.get_running_loop().run_in_executor(
            self._fence_executor, self._gate.close_and_wait, epoch
        )

    def release_final_writes(self) -> None:
        """Let terminal writes through again once the TQ snapshot is durable."""
        self._gate.reopen()

    def shutdown(self) -> None:
        """Release a waiting terminal write and retire the executors."""
        # A terminal write may be waiting at the TQ snapshot fence when the
        # actor is asked to stop. Release it before retiring the executors;
        # waiting for a wedged TQ write here would make shutdown hang.
        self._gate.reopen()
        for executor in (
            self._control_executor,
            self._fence_executor,
            self._cleanup_executor,
        ):
            executor.shutdown(wait=False, cancel_futures=True)

    # ---- Metrics --------------------------------------------------------

    def counts(self) -> dict[str, int]:
        """Generation-prefix restore and cut counts since this process started.

        Returns:
            ``restored_calls`` and their total ``restored_prefix_tokens``;
            ``completed_cuts_kept``, finished calls a cut kept whole;
            ``cut_lifecycle_races``, calls that finished while being cut;
            ``progress_reads``, ``progress_read_calls`` and the microseconds
            they waited for and held vLLM's output lock; and ``failed_cuts``
            with one ``failed_cuts_<reason>`` per
            ``GENERATION_CUT_FAILURE_REASONS`` entry.
        """
        with self._lock:
            return _count_report(
                self._restored_calls, self._restored_tokens, dict(self._counts)
            )

    def _count(self, name: str) -> None:
        with self._lock:
            self._counts[name] += 1

    # ---- Checkpoint cut -------------------------------------------------

    async def run_cut(self, inventory: Any) -> Any:
        """Run one cut on its own thread, so blocked terminal writes cannot stall it."""
        # Measured from arrival so time queued behind a stale cut counts.
        timeout_s = self._parts.control_timeout_s
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        return await asyncio.get_running_loop().run_in_executor(
            self._control_executor,
            lambda: self.checkpoint_cut(inventory, deadline=deadline),
        )

    def checkpoint_cut(self, inventory: Any, *, deadline: float | None = None) -> Any:
        """Stage a stable prefix for every call named by Gym's frozen inventory.

        ``deadline`` is a ``time.monotonic()`` bound. Gym stops waiting for the
        receipt at its own control deadline, so calls not reached by then are
        acknowledged as failures (they regenerate after restore) instead of
        holding the single cut-control thread for the next checkpoint.
        """
        from nemo_gym._checkpoint.generation_cut import (
            GenerationCutReceipt,
        )

        if self._parts.capture is None or self._parts.sink is None:
            raise RuntimeError("generation-prefix cuts require token capture setup")
        registry = self._parts.registry
        receipt_key = (inventory.checkpoint_id, inventory.inventory_digest)
        with self._lock:
            cached_receipt = self._receipts.get(receipt_key)
        if cached_receipt is not None:
            return cached_receipt
        active_states = [
            state
            for prefix in inventory.active_prefixes
            if (state := registry.get_by_model_call_id(prefix.model_call_id))
            is not None
        ]
        self._record_progress(active_states, deadline=deadline)
        acknowledgements = []
        for index, prefix in enumerate(inventory.active_prefixes):
            if deadline is not None and time.monotonic() >= deadline:
                remaining = inventory.active_prefixes[index:]
                LOGGER.warning(
                    "generation cut passed its control deadline: checkpoint_id=%s "
                    "uncut=%d of %d model_call_ids=%s",
                    inventory.checkpoint_id,
                    len(remaining),
                    len(inventory.active_prefixes),
                    [uncut.model_call_id for uncut in remaining],
                )
                acknowledgements.extend(
                    self._failed_ack(uncut, reason="control_deadline")
                    for uncut in remaining
                )
                break
            state = registry.get_by_model_call_id(prefix.model_call_id)
            try:
                acknowledgement = (
                    self._cut_active(prefix, state, inventory.checkpoint_id)
                    if state is not None
                    else None
                )
                if acknowledgement is None:
                    acknowledgement = self._completed_ack(
                        prefix, inventory.checkpoint_id
                    )
            # Gym acknowledges cuts per call, so one call's failure (a TQ write,
            # an identity mismatch) regenerates that call alone after a restore
            # instead of failing the receipt for every call on this worker.
            except Exception:  # noqa: BLE001 - see above
                LOGGER.exception(
                    "cannot cut model call %s; it regenerates after a restore",
                    prefix.model_call_id,
                )
                if state is not None:
                    with state.lock:
                        state.rollback_frozen_cut_if_owned(inventory.checkpoint_id)
                acknowledgement = self._failed_ack(prefix, reason="cut_error")
            acknowledgements.append(acknowledgement)
        receipt = GenerationCutReceipt(
            checkpoint_id=inventory.checkpoint_id,
            cut_id=f"worker-{inventory.inventory_digest}",
            inventory_digest=inventory.inventory_digest,
            inventory=inventory,
            backend_snapshot_id=f"tq-{inventory.inventory_digest}",
            prefixes=tuple(acknowledgements),
        )
        with self._lock:
            self._receipts[receipt_key] = receipt
            if len(self._receipts) > 256:
                self._receipts.pop(next(iter(self._receipts)))
        return receipt

    def _record_progress(
        self, states: list[_RequestCaptureState], *, deadline: float | None
    ) -> None:
        """Record the tokens each call decoded since its last record.

        One read at a single engine output boundary copies every call's new
        token IDs; their logprobs are then read outside it, from positions
        that read saw published. A call the engine has finished but whose
        consumer has not yet taken the output is read whole, with its finish
        reason. One the engine does not hold keeps what it has recorded.
        """
        pending = []
        for state in states:
            with state.lock:
                pending.append(
                    (
                        state,
                        state.call.model_call_id,
                        state.observed_generation_token_count,
                    )
                )
        if not pending:
            return
        try:
            read = self._parts.progress_reader.read(
                [(model_call_id, start) for _, model_call_id, start in pending],
                timeout_s=None if deadline is None else deadline - time.monotonic(),
            )
        except TimeoutError:
            LOGGER.warning(
                "generation cut could not read vLLM's progress before its deadline; "
                "%d calls are cut at the tokens they have recorded",
                len(pending),
            )
            return
        with self._lock:
            counts = self._counts
            counts["progress_reads"] += 1
            counts["progress_read_calls"] += len(pending)
            counts["progress_read_wait_us"] += round(read.wait_seconds * 1e6)
            counts["progress_read_hold_us"] += round(read.hold_seconds * 1e6)
        LOGGER.debug(
            "generation cut read %d calls from vLLM: waited %.2f ms, held %.2f ms",
            len(pending),
            read.wait_seconds * 1e3,
            read.hold_seconds * 1e3,
        )
        for state, model_call_id, start in pending:
            progress = read.progress.get(model_call_id)
            if progress is None:
                continue
            try:
                generated_logprobs = self._parts.progress_reader.logprobs(
                    progress.token_ids, progress.logprobs, start=start
                )
            except (RuntimeError, TypeError, ValueError) as error:
                with state.lock:
                    state.observation_error = f"{type(error).__name__}: {error}"
                continue
            with state.lock:
                state.observe(progress.token_ids, generated_logprobs, start=start)
                _record_terminal_reason(
                    state, progress.finish_reason, progress.stop_reason
                )

    def _failed_ack(self, prefix: Any, *, reason: str) -> Any:
        """Acknowledge one uncut call and count it under ``reason``."""
        from nemo_gym._checkpoint.generation_cut import GenerationCutPrefixAck

        if reason not in GENERATION_CUT_FAILURE_REASONS:
            raise ValueError(f"unknown generation-cut failure reason {reason!r}")
        self._count(f"failed_cuts_{reason}")
        return GenerationCutPrefixAck.failure(prefix)

    def _completed_ack(self, prefix: Any, checkpoint_id: str) -> Any:
        """Describe a call whose terminal path won the cut/abort race."""
        from nemo_gym._checkpoint.generation_cut import (
            GenerationCutPrefixAck,
        )

        with self._lock:
            completed = self._completed.get(prefix.model_call_id)
        if completed is not None:
            _require_inventory_identity(
                prefix, completed.coords.capture_key, call_kind="completed"
            )
        if completed is None:
            # Gym can name a call before its request reaches this worker, so
            # absence does not mean the call's finished result aged out.
            reason = "no_capture_or_completed_state"
            unusable_reason = "this worker has no record of it"
        elif completed.coords.disposition != "staged":
            reason = "completed_not_staged"
            unusable_reason = (
                f"its terminal write ended as {completed.coords.disposition!r}"
            )
        elif completed.effective_output_limit is None:
            reason = "completed_missing_output_limit"
            unusable_reason = "vLLM did not resolve its effective output limit"
        elif completed.terminal_finish_reason is None:
            reason = "completed_missing_finish_reason"
            unusable_reason = "it finished without a stop or length finish reason"
        else:
            self._count("completed_cuts_kept")
            return GenerationCutPrefixAck(
                **prefix.model_dump(mode="json"),
                disposition="durable_prefix",
                cut_kind="terminal_completion",
                frozen_buffer_id=f"terminal/{checkpoint_id}",
                staging_keys=(completed.coords.staging_key,),
                prefix_token_count=completed.generation_token_count,
                prefix_digest=completed.coords.digest,
                effective_output_limit=completed.effective_output_limit,
                terminal_finish_reason=completed.terminal_finish_reason,
                terminal_stop_reason=completed.terminal_stop_reason,
            )
        # The call regenerates from its previous boundary after a restore.
        LOGGER.warning(
            "cannot cut completed model call %s: %s",
            prefix.model_call_id,
            unusable_reason,
        )
        return self._failed_ack(prefix, reason=reason)

    def _cut_active(
        self,
        prefix: Any,
        state: _RequestCaptureState,
        checkpoint_id: str,
    ) -> Any | None:
        """Cut one live call, or return ``None`` if abort won the race."""
        from nemo_gym._checkpoint.generation_cut import (
            GenerationCutPrefixAck,
        )

        # checkpoint_cut checked both before cutting any call.
        capture = self._parts.capture
        sink = self._parts.sink

        # Abort and terminal completion use the same lifecycle lock. Once the
        # cut owns it, the call cannot be removed until its detached buffer is
        # staged or rolled back. If abort won first, use its terminal evidence.
        with state.lifecycle_lock:
            current = self._parts.registry.get_by_model_call_id(prefix.model_call_id)
            if current is not state or state.terminal_started:
                self._count("cut_lifecycle_races")
                return None

            _require_inventory_identity(
                prefix, state.call.capture_key, call_kind="active"
            )

            with state.lock:
                uncuttable_reason = state.uncuttable_reason
                observation_error = state.observation_error
                effective_output_limit = state.effective_output_limit
                terminal_finish_reason = state.terminal_finish_reason
                terminal_stop_reason = state.terminal_stop_reason
                if (
                    uncuttable_reason is None
                    and observation_error is None
                    and effective_output_limit is not None
                ):
                    (
                        frozen_buffer_id,
                        chunk_sequence,
                        chunk_token_ids,
                        chunk_logprobs,
                        generated_token_ids,
                        generated_logprobs,
                    ) = state.freeze_for_checkpoint(checkpoint_id)
            if uncuttable_reason is not None:
                # Expected for this request; the worker warned when it was set.
                LOGGER.debug(
                    "not cutting model call %s: %s",
                    prefix.model_call_id,
                    uncuttable_reason,
                )
                return self._failed_ack(prefix, reason="uncuttable_request")
            if observation_error is not None:
                # Its recorded tokens or logprobs are inconsistent with vLLM's.
                # Only this call regenerates; the rest of the inventory is cut.
                LOGGER.warning(
                    "cannot cut model call %s: %s",
                    prefix.model_call_id,
                    observation_error,
                )
                return self._failed_ack(prefix, reason="observation_error")
            if effective_output_limit is None:
                # Checked before freezing, so the call's buffer is untouched.
                LOGGER.warning(
                    "cannot cut model call %s: vLLM did not resolve its "
                    "effective output limit",
                    prefix.model_call_id,
                )
                return self._failed_ack(prefix, reason="missing_output_limit")
            inherited_prefix_token_count = (
                state.call.admission.generation_cut.prefix_token_count
                if state.call.admission.generation_cut is not None
                else 0
            )
            total_generation_token_count = inherited_prefix_token_count + len(
                generated_token_ids
            )
            unusable_prefix = True
            if total_generation_token_count == 0:
                # Expected for a call still prefilling; nothing to keep yet.
                LOGGER.debug(
                    "not cutting model call %s: it has no generated tokens yet",
                    prefix.model_call_id,
                )
            elif generated_token_ids and _ends_inside_character(
                self._parts.tokenizer,
                state.resumed_generation_token_ids + generated_token_ids,
            ):
                LOGGER.warning(
                    "cannot cut model call %s: its %d generated tokens end inside "
                    "a multi-byte character",
                    prefix.model_call_id,
                    total_generation_token_count,
                )
            else:
                unusable_prefix = False
            if unusable_prefix:
                with state.lock:
                    state.rollback_frozen_cut(checkpoint_id)
                return self._failed_ack(
                    prefix,
                    reason="zero_generated_tokens"
                    if total_generation_token_count == 0
                    else "incomplete_utf8_boundary",
                )
            staged_key = None
            try:
                cumulative_record = capture.build_prefix_record(
                    state.call,
                    prompt_token_ids=state.prompt_token_ids,
                    generated_token_ids=generated_token_ids,
                    generated_logprobs=generated_logprobs,
                )
                if chunk_token_ids:
                    chunk_record = (
                        cumulative_record
                        if not state.generation_cut_staging_keys
                        else capture.build_generation_chunk_record(
                            state.call,
                            generated_token_ids=chunk_token_ids,
                            generated_logprobs=chunk_logprobs,
                        )
                    )
                    result = sink.stage_generation_prefix(
                        chunk_record,
                        checkpoint_id=checkpoint_id,
                        chunk_sequence=chunk_sequence,
                    )
                    staged_key = result.staging_key
                    if not result.ok:
                        raise RuntimeError(
                            "generation-prefix staging failed for "
                            f"{prefix.model_call_id!r}: {result.error}"
                        )
                    if staged_key is None:
                        raise RuntimeError(
                            "generation-prefix staging returned no staging key for "
                            f"{prefix.model_call_id!r}"
                        )
                with state.lock:
                    candidate_staging_keys = tuple(state.generation_cut_staging_keys)
                    if staged_key is not None:
                        candidate_staging_keys += (staged_key,)
                    acknowledgement = GenerationCutPrefixAck(
                        **prefix.model_dump(mode="json"),
                        disposition="durable_prefix",
                        cut_kind="active_prefix",
                        frozen_buffer_id=frozen_buffer_id,
                        staging_keys=candidate_staging_keys,
                        prefix_token_count=total_generation_token_count,
                        prefix_digest=cumulative_record.digest,
                        effective_output_limit=effective_output_limit,
                        terminal_finish_reason=terminal_finish_reason,
                        terminal_stop_reason=terminal_stop_reason,
                    )
                    # Validate the complete acknowledgement before adopting the
                    # staged chunk. A failed acknowledgement must leave the
                    # frozen buffer available for rollback.
                    state.seal_frozen_cut(checkpoint_id)
                    if staged_key is not None:
                        state.generation_cut_staging_keys.append(staged_key)
            except Exception:
                with state.lock:
                    state.rollback_frozen_cut_if_owned(checkpoint_id)
                if staged_key is not None:
                    try:
                        sink.clear([staged_key])
                    except Exception:  # noqa: BLE001 — preserve the cut failure
                        LOGGER.exception(
                            "Failed to clear rejected generation-prefix row %s",
                            staged_key,
                        )
                raise
            return acknowledgement

    def _rebuild_prefix(self, admission: Any, prefix_token_ids: list[int]) -> Any:
        """Fetch an admission's staged prefix chunks and rebuild the cumulative prefix."""
        continuation = admission.generation_cut
        # Gym requires at least one staging key per continuation.
        staging_keys = list(continuation.staging_keys)
        snapshots = self._parts.staging_source.fetch(staging_keys)
        if len(snapshots) != len(staging_keys):
            raise RuntimeError("generation-cut fetch did not return every staged chunk")
        weight_versions = [snapshot.weight_version for snapshot in snapshots]
        if weight_versions != sorted(weight_versions):
            raise RuntimeError(
                "generation-cut chunk policy versions are not monotonically non-decreasing"
            )
        current_weight_version = self._parts.weight_version()
        if weight_versions[-1] > current_weight_version:
            raise RuntimeError(
                "generation-cut prefix contains policy version "
                f"{weight_versions[-1]}, newer than the current rollout version "
                f"{current_weight_version}"
            )
        token_ids_delta = [
            token_id for snapshot in snapshots for token_id in snapshot.token_ids_delta
        ]
        token_mask_delta = [
            mask for snapshot in snapshots for mask in snapshot.token_mask_delta
        ]
        generation_logprobs_delta = [
            logprob
            for snapshot in snapshots
            for logprob in snapshot.generation_log_probs_delta
        ]
        if any(
            mask == 0.0
            for snapshot in snapshots[1:]
            for mask in snapshot.token_mask_delta
        ):
            raise RuntimeError(
                "only the first generation-cut chunk may contain prompt tokens"
            )

        # Deferred: nemo_gym is an optional extra absent in non-Gym runs.
        from nemo_gym.token_id_capture.staging.digest import (
            compute_chain_hash,
            compute_extras_digest,
            hash_token_ids,
        )
        from nemo_gym.token_id_capture.staging.records import StagedCallRecord

        delta_len = len(token_ids_delta)
        # A cut chain may span retries performed after later weight updates.
        # Its token-level behavior logprobs remain attached to their original
        # chunks. The cumulative snapshot uses the oldest contributing version
        # so downstream replay-buffer staleness checks remain conservative.
        # Gym's begin_call checks this snapshot's digest and generated-token
        # count against the admission before the call is served.
        snapshot = StagedCallRecord.from_components(
            capture_key=continuation.source_capture_key,
            model_call_id=continuation.source_model_call_id,
            parent_call_id=admission.parent_call_id,
            mode=admission.mode,
            prev_len=admission.prev_len,
            delta_len=delta_len,
            cum_len=admission.prev_len + delta_len,
            weight_version=weight_versions[0],
            token_ids_delta=token_ids_delta,
            token_mask_delta=token_mask_delta,
            generation_log_probs_delta=generation_logprobs_delta,
            extras=None,
            extras_digest=compute_extras_digest(None),
            chain_hash=compute_chain_hash(admission.parent_chain_hash, token_ids_delta),
            cumulative_hash=hash_token_ids(prefix_token_ids + token_ids_delta),
        )
        # Lets a consumer confirm the exact generated tokens it later receives
        # (e.g. a trained row) without access to the staging store.
        generated_prefix_ids = [
            token_id
            for token_id, mask in zip(token_ids_delta, token_mask_delta, strict=True)
            if mask == 1.0
        ]
        prefix_ids_sha256 = hashlib.sha256(
            ",".join(map(str, generated_prefix_ids)).encode()
        ).hexdigest()
        LOGGER.info(
            "generation prefix restored: rollout_id=%s model_call_id=%s "
            "source_model_call_id=%s prefix_tokens=%d prefix_digest=%s "
            "weight_version_span=[%d,%d] prefix_ids_sha256=%s",
            admission.capture_key,
            admission.model_call_id,
            continuation.source_model_call_id,
            continuation.prefix_token_count,
            continuation.prefix_digest,
            weight_versions[0],
            weight_versions[-1],
            prefix_ids_sha256,
        )
        return snapshot


class NoPrefixRecovery:
    """Stand-in when prefix cuts are off: nothing is cut, restored, or held."""

    control_token: str | None = None

    def resolve_restore(
        self, admission: Any, prefix_token_ids: list[int]
    ) -> tuple[Any, Any | None]:
        # Gym only sends a restored cut to a worker that acknowledged one, so
        # this is a misconfiguration; regenerating the call is always valid.
        if admission.generation_cut is None:
            return admission, None
        LOGGER.warning(
            "generation-prefix cuts are off; regenerating model call %s",
            admission.model_call_id,
        )
        return admission.model_copy(update={"generation_cut": None}), None

    def on_begin(self, admission: Any) -> None:
        pass

    def on_sampling_params(self, request: Any, sampling_params: Any) -> None:
        pass

    def observe(self, request: Any, request_output: Any) -> None:
        pass

    def restore_reply_text(
        self, request: Any, request_output: Any, *, tokenizer: Any
    ) -> None:
        pass

    def on_terminal(
        self, model_call_id: str, coords: Any, generation_token_count: int, **_: Any
    ) -> None:
        pass

    def final_write(self) -> contextlib.AbstractContextManager[None]:
        return contextlib.nullcontext()

    def retire_rows(self, model_call_id: str, staging_keys: tuple[str, ...]) -> None:
        pass

    def retire_rows_later(
        self, model_call_id: str, staging_keys: tuple[str, ...]
    ) -> None:
        pass

    async def hold_final_writes(self) -> None:
        pass

    def release_final_writes(self) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def counts(self) -> dict[str, int]:
        return _count_report(0, 0, initial_generation_cut_counts())

    async def run_cut(self, inventory: Any) -> Any:
        raise RuntimeError("generation-prefix cuts are disabled")


def _count_report(
    restored_calls: int, restored_tokens: int, cut_counts: dict[str, int]
) -> dict[str, int]:
    return {
        "restored_calls": restored_calls,
        "restored_prefix_tokens": restored_tokens,
        **cut_counts,
        "failed_cuts": sum(
            cut_counts[f"failed_cuts_{reason}"]
            for reason in GENERATION_CUT_FAILURE_REASONS
        ),
    }
