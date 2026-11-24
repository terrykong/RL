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

"""Serve a NeMo RL Megatron engine through Dynamo's backend SDK.

``DynamoMegatronAdapter`` is the Megatron backend of
``engine_adapter.DynamoEngineAdapter``. It runs on rank zero of a prefill or
decode engine, on the worker's inference loop, and calls the engine through the
worker's own ``InferenceClient``, the one ``generate_async`` uses. Refit, pause
and the engine's lifetime stay the worker's own.

The KV handoff follows Megatron-LM's Dynamo adapter
(``megatron/inference/integrations/dynamo/llm_engine.py``): prefill pins the
prompt's KV and returns where it is; decode claims it, reads it, and releases it.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Awaitable
from typing import TYPE_CHECKING, Any, Optional, TypeVar

from dynamo._core import Context
from dynamo.common.backend import EngineConfig, LlmRegistration
from dynamo.common.backend.publisher import KvEventSource, PushSource
from dynamo.common.constants import DisaggregationMode
from dynamo.llm import KvEventPublisher
from megatron.core.inference.engine_endpoint import InferenceEngineCapabilities
from megatron.core.inference.inference_request import DynamicInferenceRequest
from megatron.core.inference.sampling_params import SamplingParams
from megatron.inference.integrations.dynamo.handoff_recovery import HandoffRelease

from nemo_rl.models.generation.dynamo.engine_adapter import (
    DynamoEngineAdapter,
    Handoff,
    ServedRollout,
)
from nemo_rl.models.generation.megatron.dynamo_config import MegatronDynamoRole
from nemo_rl.models.generation.rollout_result import RolloutResult

if TYPE_CHECKING:
    from nemo_rl.models.generation.megatron.megatron_worker import (
        MegatronGenerationMixin,
    )

LOGGER = logging.getLogger(__name__)

_T = TypeVar("_T")

_MODE_BY_ROLE: dict[str, DisaggregationMode] = {
    "prefill": DisaggregationMode.PREFILL,
    "decode": DisaggregationMode.DECODE,
}


class DynamoMegatronAdapter(DynamoEngineAdapter):
    """The Megatron backend, over a running NeMo RL Megatron engine it does not own."""

    def __init__(
        self,
        host: "MegatronGenerationMixin",
        *,
        leader_worker_idx: int,
        role: MegatronDynamoRole,
        model_name: str,
    ):
        super().__init__(
            leader_worker_idx=leader_worker_idx,
            disaggregation_mode=_MODE_BY_ROLE[role],
        )
        self._host = host
        self._model_name = model_name
        # Dynamo's request ID -> the task serving it, so abort can cancel it.
        self._requests: dict[str, asyncio.Task] = {}
        # Decode only: who claims prefill KV. Fresh per engine launch, as
        # Megatron-LM's adapter makes it.
        self._handoff_owner = uuid.uuid4().hex
        # Decode only: claimed handoffs whose read has stopped, so their
        # prefill KV may be released.
        self._releasable: set[HandoffRelease] = set()
        self._published_block_hashes: set[int] = set()

    async def start(self, worker_id: int) -> EngineConfig:
        del worker_id
        capabilities = InferenceEngineCapabilities.from_engine(
            self._host.dynamic_inference_engine
        )
        return EngineConfig(
            model=self._model_name,
            served_model_name=self._model_name,
            llm=LlmRegistration(
                context_length=capabilities.context_length,
                kv_cache_block_size=capabilities.kv_cache_block_size,
                total_kv_blocks=capabilities.total_kv_blocks,
                max_num_seqs=capabilities.max_num_seqs,
                max_num_batched_tokens=capabilities.max_num_batched_tokens,
            ),
        )

    async def serve_rollout(
        self,
        prompt_token_ids: list[int],
        *,
        greedy: bool,
        stop_strings: Optional[list[str]],
        request_id: str,
        handoff: Optional[Handoff] = None,
    ) -> ServedRollout:
        params = self._sampling_params(greedy, stop_strings)
        if handoff is None:
            # The native step generate_async runs for each sample.
            (request,) = await self._tracked(
                request_id,
                self._host._generate_with_persistent_engine(
                    [prompt_token_ids], [None], [params]
                ),
            )
        else:
            request = await self._tracked(
                request_id, self._decode_from(prompt_token_ids, params, handoff)
            )
        return _served(request, params, num_prompt_tokens=len(prompt_token_ids))

    async def serve_prefill(
        self,
        prompt_token_ids: list[int],
        *,
        greedy: bool,
        stop_strings: Optional[list[str]],
        request_id: str,
    ) -> Handoff:
        params = self._sampling_params(greedy, stop_strings)
        params.do_kv_handoff = True
        params.num_tokens_to_generate = 0
        client = self._host.inference_client
        engine_request_id, future = client.add_request_with_id(prompt_token_ids, params)
        try:
            request = await self._tracked(request_id, asyncio.shield(future))
        except asyncio.CancelledError:
            # The engine still pins the KV once this prefill finishes; free it
            # then instead of waiting for the unclaimed offer to expire.
            future.add_done_callback(self._release_unsent_prefill)
            raise
        handoff = dict(request.disaggregated_params or {})
        if not handoff:
            raise RuntimeError(
                "Megatron prefill returned no handoff; the engine needs "
                "setup_kv_transfer(role='prefill')"
            )
        if not client.coordinator_instance_id:
            raise RuntimeError("A Dynamo handoff needs coordinator instance IDs")
        # What decode claims and releases, in Megatron-LM's own format.
        handoff["release"] = {
            "coordinator_addr": self._host.coordinator_addr,
            "coordinator_instance_id": client.coordinator_instance_id,
            "request_id": int(handoff.get("request_id", engine_request_id)),
        }
        return handoff

    async def release_handoff(self, handoff: Handoff) -> None:
        release = HandoffRelease.from_metadata(handoff["release"])
        if release not in self._releasable:
            # Not claimed, or decode may still be reading it: leave it pinned.
            return
        self._releasable.discard(release)
        raise NotImplementedError(
            "Send RELEASE_KV to the prefill coordinator, as Megatron-LM's "
            "MegatronLLMEngine._release_handoff_from_meta_async does "
            "(megatron/inference/integrations/dynamo/llm_engine.py:708). It is a "
            "private method; it needs to become a function this adapter can call."
        )

    async def abort(self, context: Context) -> None:
        task = self._requests.pop(context.id(), None)
        if task is not None:
            # The native step aborts its Megatron request when cancelled.
            task.cancel()

    async def kv_event_sources(self) -> list[KvEventSource]:
        # Only prefill engines compute prompt KV the KV router can reuse.
        if self.disaggregation_mode is not DisaggregationMode.PREFILL:
            return []
        if not self._host.dynamic_inference_engine.context.enable_prefix_caching:
            return []
        return [PushSource(on_ready=self._start_publishing, dp_rank=0)]

    def _sampling_params(
        self, greedy: bool, stop_strings: Optional[list[str]]
    ) -> SamplingParams:
        """The SamplingParams generate_async builds for one sample."""
        stop_words = self._host._merge_stop_strings(
            [stop_strings] if stop_strings else None
        )
        return self._host._build_sampling_params(greedy, stop_words)

    async def _tracked(self, request_id: str, step: Awaitable[_T]) -> _T:
        """Run ``step`` as a task abort can cancel by Dynamo's request ID."""
        task = asyncio.ensure_future(step)
        self._requests[request_id] = task
        try:
            return await task
        finally:
            self._requests.pop(request_id, None)

    async def _decode_from(
        self, prompt_token_ids: list[int], params: SamplingParams, handoff: Handoff
    ) -> DynamicInferenceRequest:
        """Claim the prefill's KV, read it into this engine, and generate."""
        release = HandoffRelease.from_metadata(handoff["release"])
        await self._claim(release)
        client = self._host.inference_client
        stream = client.add_request_with_kv_handoff_streaming(
            prompt_token_ids, params, handoff["kv_meta"], list(handoff["block_ids"])
        )
        try:
            async for item in stream:
                if "final" in item:
                    self._releasable.add(release)
                    return item["final"]
            raise RuntimeError("Megatron decode stream ended without a result")
        except BaseException:
            # Release only once the engine confirms it stopped reading.
            if await asyncio.shield(client.abort_request_and_wait(stream.request_id)):
                self._releasable.add(release)
            raise

    async def _claim(self, release: HandoffRelease) -> None:
        """Take ownership of the prefill's KV before reading it."""
        raise NotImplementedError(
            f"Send CLAIM_KV for owner {self._handoff_owner} to "
            f"{release.coordinator_addr}, as Megatron-LM's "
            "MegatronLLMEngine._claim_remote_handoff does "
            "(megatron/inference/integrations/dynamo/llm_engine.py:609). It is a "
            "private method; it needs to become a function this adapter can call."
        )

    def _release_unsent_prefill(self, future: asyncio.Future) -> None:
        """Free a finished prefill's pinned KV that no decode will ever claim."""
        if future.cancelled() or future.exception() is not None:
            return
        handoff = future.result().disaggregated_params
        if handoff:
            self._host.inference_client.release_handoff(int(handoff["request_id"]))

    def _start_publishing(self, publisher: KvEventPublisher) -> None:
        """Forward the engine's KV events to Dynamo's KV router from now on.

        The engine calls listeners from its own loop, the inference loop this
        adapter runs on.
        """
        context = self._host.dynamic_inference_engine.context
        context.dynamo_helper.add_kv_event_listener(
            lambda kind, payload: self._publish_kv_event(publisher, kind, payload)
        )

    def _publish_kv_event(
        self, publisher: KvEventPublisher, kind: str, payload: dict[str, Any]
    ) -> None:
        """Publish one engine KV event, as Megatron-LM's adapter does."""
        if kind == "stored":
            publisher.publish_stored(**payload)
            self._published_block_hashes.update(payload["block_hashes"])
        elif kind == "removed":
            publisher.publish_removed(**payload)
            self._published_block_hashes.difference_update(payload["block_hashes"])
        elif kind == "cleared":
            # Dynamo's publisher has no clear-all: remove what was published.
            if self._published_block_hashes:
                publisher.publish_removed(
                    block_hashes=sorted(self._published_block_hashes)
                )
                self._published_block_hashes.clear()
        else:
            LOGGER.warning("Ignoring unknown Megatron KV event %s", kind)


def _served(
    request: DynamicInferenceRequest, params: SamplingParams, *, num_prompt_tokens: int
) -> ServedRollout:
    """The rollout ``request`` produced, as generate_async reads it."""
    token_ids = list(request.generated_tokens)
    truncated = len(token_ids) >= params.num_tokens_to_generate
    result = RolloutResult(
        token_ids=token_ids,
        logprobs=list(request.generated_log_probs),
        truncated=truncated,
        routed_experts=None,
        r3_stats=None,
    )
    return ServedRollout(
        result,
        finish_reason="length" if truncated else "stop",
        usage={
            "prompt_tokens": num_prompt_tokens,
            "completion_tokens": len(token_ids),
            "total_tokens": num_prompt_tokens + len(token_ids),
        },
    )
