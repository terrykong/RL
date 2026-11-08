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

"""Per-call capture state shared by the worker's terminal path and prefix recovery."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class _RequestCaptureState:
    """Append-only generated tokens for one in-flight captured request.

    Tokens are recorded lazily: a checkpoint cut reads the tokens vLLM decoded
    since the last record, and the finished output records the rest. ``observe``
    keeps each position once whichever source reaches it first. A cut freezes
    the recorded length under ``lock`` and performs its blocking TQ write after
    releasing it.

    Each successful cut stages only the tokens past ``staged_token_count``.
    The ordered TQ keys remain in Gym lineage; the full list is kept locally to
    compute the cumulative digest advertised by the latest cut.
    """

    call: Any
    prompt_token_ids: list[int]
    # Backend-owned opaque state that must survive until the terminal row is
    # staged. vLLM uses this for its processed-media snapshot; prefix recovery
    # is currently setup-guarded to text-only, but ordinary terminal capture
    # still shares this lifecycle container.
    media: Any | None = None
    resumed_generation_token_ids: list[int] = field(default_factory=list)
    effective_output_limit: int | None = None
    terminal_finish_reason: Literal["stop", "length"] | None = None
    terminal_stop_reason: str | int | None = None
    observation_error: str | None = None
    # Set when the request's sampling settings would not carry across a cut,
    # so the call regenerates after a restore instead of resuming its prefix.
    uncuttable_reason: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    # Serializes lifecycle-changing TQ operations for this call. Token
    # recording deliberately uses ``lock`` instead so a finished output can be
    # recorded while a detached generation chunk is staged.
    lifecycle_lock: threading.Lock = field(default_factory=threading.Lock)
    terminal_started: bool = False
    generated_token_ids: list[int] = field(default_factory=list)
    generated_logprobs: list[float] = field(default_factory=list)
    # Tokens already written to TQ by sealed cuts.
    staged_token_count: int = 0
    generation_cut_staging_keys: list[str] = field(default_factory=list)
    # The cut in progress: its checkpoint and the token count it froze.
    frozen_cut: tuple[str, int] | None = None
    # Names each frozen chunk, so a rolled-back cut never reuses a TQ key.
    next_chunk_sequence: int = 0

    @property
    def observed_generation_token_count(self) -> int:
        return len(self.generated_token_ids)

    def observe(
        self,
        generated_token_ids: list[int],
        generated_logprobs: list[float],
        *,
        start: int | None = None,
    ) -> None:
        """Record generated tokens read at position ``start`` (default: the end).

        A cut and the finished output both read from vLLM, so positions already
        recorded by the other are skipped rather than appended twice. Callers
        read ``start`` from this state, and tokens are never removed, so it
        never lies past the recorded end.
        """
        recorded = len(self.generated_token_ids)
        skip = 0 if start is None else recorded - start
        self.generated_token_ids.extend(generated_token_ids[skip:])
        self.generated_logprobs.extend(generated_logprobs[skip:])

    def freeze_for_checkpoint(
        self, checkpoint_id: str
    ) -> tuple[str, int, list[int], list[float], list[int], list[float]]:
        """Freeze the current length; return the unstaged chunk and the prefix."""
        if self.frozen_cut is not None:
            raise RuntimeError(
                "cannot start a generation cut while another cut is in progress"
            )
        frozen = len(self.generated_token_ids)
        sequence = self.next_chunk_sequence
        self.next_chunk_sequence += 1
        self.frozen_cut = (checkpoint_id, frozen)
        staged = self.staged_token_count
        return (
            f"active/{checkpoint_id}/{sequence}",
            sequence,
            self.generated_token_ids[staged:frozen],
            self.generated_logprobs[staged:frozen],
            self.generated_token_ids[:frozen],
            self.generated_logprobs[:frozen],
        )

    def seal_frozen_cut(self, checkpoint_id: str) -> None:
        """Count a successfully staged cut's tokens as written to TQ."""
        self.staged_token_count = self._require_frozen_cut(checkpoint_id)
        self.frozen_cut = None

    def rollback_frozen_cut_if_owned(self, checkpoint_id: str) -> None:
        """Abandon ``checkpoint_id``'s cut if it froze this call's tokens."""
        if self.frozen_cut is not None and self.frozen_cut[0] == checkpoint_id:
            self.frozen_cut = None

    def rollback_frozen_cut(self, checkpoint_id: str) -> None:
        """Abandon a failed cut; its tokens stay unstaged for the next one."""
        self._require_frozen_cut(checkpoint_id)
        self.frozen_cut = None

    def _require_frozen_cut(self, checkpoint_id: str) -> int:
        if self.frozen_cut is None or self.frozen_cut[0] != checkpoint_id:
            raise RuntimeError(
                f"generation cut {checkpoint_id!r} does not own the frozen cut"
            )
        return self.frozen_cut[1]


@dataclass(frozen=True)
class _CompletedCaptureState:
    """Time-bounded terminal evidence retained across the response/cut race."""

    coords: Any
    generation_token_count: int
    effective_output_limit: int | None
    terminal_finish_reason: Literal["stop", "length"] | None
    terminal_stop_reason: str | int | None
    completed_at_monotonic: float


class _TokenCaptureSnapshotGate:
    """Drain active terminal writes and block new ones across a TQ snapshot.

    Each fence takes an epoch before its close is queued. ``reopen`` releases
    every epoch issued so far, so a close that runs after the driver already
    timed out and released its fence is a no-op instead of a stuck gate.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._open = True
        self._active = 0
        self._epoch = 0
        self._released_epoch = 0

    def enter(self) -> None:
        with self._condition:
            while not self._open:
                self._condition.wait()
            self._active += 1

    def exit(self) -> None:
        with self._condition:
            self._active -= 1
            if self._active == 0:
                self._condition.notify_all()

    def begin_epoch(self) -> int:
        with self._condition:
            self._epoch += 1
            return self._epoch

    def close_and_wait(self, epoch: int) -> None:
        with self._condition:
            if epoch <= self._released_epoch:
                return
            self._open = False
            while self._active and self._released_epoch < epoch:
                self._condition.wait()

    def reopen(self) -> None:
        with self._condition:
            self._released_epoch = self._epoch
            self._open = True
            self._condition.notify_all()


class CaptureRegistry:
    """In-flight captured calls, indexed by request object and by model call ID.

    The worker's terminal path uses the first index; a checkpoint cut uses the
    second, because Gym's inventory names calls by model call ID.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self._by_request: dict[int, _RequestCaptureState] = {}
        self._by_model_call_id: dict[str, _RequestCaptureState] = {}

    def add(self, request: Any, state: _RequestCaptureState) -> None:
        with self.lock:
            if state.call.model_call_id in self._by_model_call_id:
                raise RuntimeError(
                    f"model call {state.call.model_call_id!r} is already active "
                    "in token capture"
                )
            self._by_request[id(request)] = state
            self._by_model_call_id[state.call.model_call_id] = state

    def get(self, request: Any) -> _RequestCaptureState | None:
        with self.lock:
            return self._by_request.get(id(request))

    def __contains__(self, request: Any) -> bool:
        return id(request) in self._by_request

    def pop(self, request: Any) -> _RequestCaptureState | None:
        with self.lock:
            state = self._by_request.pop(id(request), None)
            if state is not None:
                self._by_model_call_id.pop(state.call.model_call_id, None)
            return state

    def get_by_model_call_id(self, model_call_id: str) -> _RequestCaptureState | None:
        with self.lock:
            return self._by_model_call_id.get(model_call_id)
