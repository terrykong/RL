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

"""Serve a NeMo RL generation worker's own engine through Dynamo's backend SDK.

``DynamoEngineAdapter`` is the Dynamo side, written once for every backend: the
request kinds served, how errors reach the driver, and the endpoint's
registration. A backend implements four methods over its worker's engine
(``serve_rollout``, ``start``, ``abort`` and ``kv_event_sources``);
``vllm.adapter.DynamoVllmAdapter`` is the vLLM one. A backend that splits
prefill and decode across engines also implements ``serve_prefill`` and
``release_handoff``, and its decode ``serve_rollout`` takes the prefill's handoff. The engine only ever receives
token IDs: NeMo RL's rollouts on ``/v1/completions``, and NeMo-Gym's chat calls
once NeMo RL's chat engine in the frontend has rendered them. This module imports no generation backend.
"""

from __future__ import annotations

import asyncio
import json
import os
from abc import abstractmethod
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass
from typing import Any, Optional

from dynamo._core import Context, fetch_model
from dynamo.common.backend import (
    EngineConfig,
    GenerateChunk,
    GenerateRequest,
    LLMEngine,
    Worker,
    WorkerConfig,
)
from dynamo.common.backend.disagg import require_prefill_result
from dynamo.common.backend.publisher import KvEventSource
from dynamo.common.constants import DisaggregationMode
from dynamo.llm.exceptions import InvalidArgument

from nemo_rl.models.generation.dynamo.managed_runtime import (
    DYNAMO_COMPONENT,
    DYNAMO_ENDPOINT,
)
from nemo_rl.models.generation.dynamo.rollout_client import ROLLOUT_ANNOTATION
from nemo_rl.models.generation.rollout_result import (
    RolloutResult,
    rollout_result_to_wire,
)

# Serves one request kind: the request, its annotation's value, and the context.
RequestHandler = Callable[
    [GenerateRequest, str, Context], AsyncGenerator[GenerateChunk, None]
]

# What a prefill engine hands its decode peer: Dynamo's ``disaggregated_params``,
# which only the backend reads.
Handoff = dict[str, Any]


@dataclass(frozen=True)
class ServedRollout:
    """A backend's answer to one rollout."""

    result: RolloutResult
    # The OpenAI finish reason the frontend reports.
    finish_reason: str
    # The OpenAI completion usage; None when no new token fit.
    usage: Optional[dict[str, Any]]


def _annotation_value(request: GenerateRequest, prefix: str) -> Optional[str]:
    """The value of the first ``<prefix><value>`` annotation, or None."""
    for annotation in request.get("annotations") or ():
        if annotation.startswith(prefix):
            return annotation[len(prefix) :]
    return None


class DynamoEngineAdapter(LLMEngine):
    """Serves one NeMo RL generation worker through Dynamo. The worker owns the engine."""

    def __init__(
        self,
        *,
        leader_worker_idx: int,
        disaggregation_mode: DisaggregationMode = DisaggregationMode.AGGREGATED,
    ):
        # The worker's index in its group; rollouts report it, as
        # generate_async results do.
        self._leader_worker_idx = leader_worker_idx
        # AGGREGATED serves whole rollouts; PREFILL and DECODE each serve one
        # half, and start_dynamo_endpoint registers the engine as that half.
        self.disaggregation_mode = disaggregation_mode

    @classmethod
    async def from_args(cls, argv=None):
        raise NotImplementedError("constructed in-process by the generation worker")

    def _handlers(self) -> dict[str, RequestHandler]:
        """The request kinds served, keyed by the annotation prefix that marks them."""
        if self.disaggregation_mode is DisaggregationMode.PREFILL:
            return {ROLLOUT_ANNOTATION: self._prefill_rollout}
        return {ROLLOUT_ANNOTATION: self._generate_rollout}

    async def generate(
        self, request: GenerateRequest, context: Context
    ) -> AsyncGenerator[GenerateChunk, None]:
        handlers = self._handlers()
        for prefix, handler in handlers.items():
            value = _annotation_value(request, prefix)
            if value is not None:
                break
        else:
            raise InvalidArgument(
                "NeMo RL's engines serve only NeMo RL's own requests through Dynamo "
                f"(requests annotated {', '.join(map(repr, handlers))})."
            )
        try:
            async for chunk in handler(request, value, context):
                yield chunk
        except RuntimeError as error:
            # A 5xx reaches the driver as "Internal server error" and is re-sent;
            # a 4xx keeps the cause, and the rollout retry policy still applies.
            raise InvalidArgument(f"{type(error).__name__}: {error}") from error

    async def _generate_rollout(
        self, request: GenerateRequest, rollout_spec: str, context: Context
    ) -> AsyncGenerator[GenerateChunk, None]:
        """One NeMo RL rollout, served by the backend's ``serve_rollout``.

        One chunk carries the result, routed experts included, in ``engine_data``;
        the driver assembles the same output ``generate_async`` returns.
        """
        rollout = json.loads(rollout_spec)
        handoff: Optional[Handoff] = None
        if self.disaggregation_mode is DisaggregationMode.DECODE:
            prefill = require_prefill_result(request, DisaggregationMode.DECODE)
            handoff = prefill["disaggregated_params"]
        try:
            served = await self.serve_rollout(
                list(request["token_ids"]),
                greedy=rollout["greedy"],
                stop_strings=rollout["stop_strings"],
                request_id=context.id(),
                handoff=handoff,
            )
        finally:
            if handoff is not None:
                await self.release_handoff(handoff)
        chunk: GenerateChunk = {
            "token_ids": served.result.token_ids,
            "index": 0,
            "finish_reason": served.finish_reason,
            "engine_data": {
                "rollout": rollout_result_to_wire(served.result),
                "gen_leader_worker_idx": self._leader_worker_idx,
            },
        }
        if served.usage is not None:
            chunk["completion_usage"] = served.usage
        yield chunk

    async def _prefill_rollout(
        self, request: GenerateRequest, rollout_spec: str, context: Context
    ) -> AsyncGenerator[GenerateChunk, None]:
        """The prefill half of one rollout: its handoff, for Dynamo to pass to decode."""
        rollout = json.loads(rollout_spec)
        prompt_token_ids = list(request["token_ids"])
        handoff = await self.serve_prefill(
            prompt_token_ids,
            greedy=rollout["greedy"],
            stop_strings=rollout["stop_strings"],
            request_id=context.id(),
        )
        num_prompt_tokens = len(prompt_token_ids)
        yield {
            "token_ids": [],
            "index": 0,
            # Dynamo stops on any other finish reason without reading the
            # handoff, so decode would never run.
            "finish_reason": "length",
            "completion_usage": {
                "prompt_tokens": num_prompt_tokens,
                "completion_tokens": 0,
                "total_tokens": num_prompt_tokens,
            },
            "disaggregated_params": handoff,
        }

    @abstractmethod
    async def serve_rollout(
        self,
        prompt_token_ids: list[int],
        *,
        greedy: bool,
        stop_strings: Optional[list[str]],
        request_id: str,
        handoff: Optional[Handoff] = None,
    ) -> ServedRollout:
        """Serve one rollout exactly as the worker's native ``generate_async`` does.

        On a decode engine ``handoff`` is what the prefill engine's
        ``serve_prefill`` returned: the rollout reads the prompt's KV from there
        instead of computing it. It is None on every other engine.
        """

    async def serve_prefill(
        self,
        prompt_token_ids: list[int],
        *,
        greedy: bool,
        stop_strings: Optional[list[str]],
        request_id: str,
    ) -> Handoff:
        """Prefill engines only: compute the prompt's KV, keep it, and say where it is.

        The KV stays held until the decode engine's ``release_handoff``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not serve the prefill half of a rollout"
        )

    async def release_handoff(self, handoff: Handoff) -> None:
        """Decode engines only: let the prefill engine free what ``handoff`` holds.

        Called once per handoff, after ``serve_rollout`` returned or raised. A
        backend whose prefill KV frees itself keeps this default.
        """

    @abstractmethod
    async def start(self, worker_id: int) -> EngineConfig:
        """Register the engine's context length, KV block size and capacity."""

    @abstractmethod
    async def abort(self, context: Context) -> None:
        """Cancel the request ``context`` names; Dynamo calls it on disconnect."""

    @abstractmethod
    async def kv_event_sources(self) -> list[KvEventSource]:
        """Where the engine publishes the KV-cache events Dynamo's KV router indexes."""

    async def cleanup(self) -> None:
        """Nothing to release: the worker owns its engine and shuts it down."""


async def weight_free_model_path(model: str) -> str:
    """A local directory with ``model``'s config and tokenizer files.

    ``model`` itself when it is a local path; for a Hub ID, Dynamo's snapshot
    of it without the weights.
    """
    if os.path.exists(model):
        return model
    return await fetch_model(model, ignore_weights=True)


def start_dynamo_endpoint(
    adapter: DynamoEngineAdapter,
    *,
    model_name: str,
    namespace: str,
    engine_env: dict[str, str],
    component: str = DYNAMO_COMPONENT,
) -> asyncio.Task:
    """Register ``adapter``'s engine with Dynamo and serve it on the running loop.

    ``model_name`` is the engine's model, a local path or a Hub ID, and the name
    it is served under. Dynamo registers a Hub ID from its config and tokenizer
    files alone. Chat calls are rendered by NeMo RL's chat engine in the
    frontend, not Dynamo's renderer, so no template or parser is registered.
    The engine registers as ``adapter.disaggregation_mode`` under ``component``;
    prefill engines take a component of their own, so the frontend's prefill
    router can find them.

    The Rust Worker serves until the process gets SIGTERM. It installs a chained
    handler, so Ray's actor handler still runs and the actor exits as usual;
    discovery drops the engine with its process.
    """
    # The Rust runtime reads these when the first Worker is constructed.
    os.environ.update(engine_env)
    config = WorkerConfig(
        namespace=namespace,
        component=component,
        endpoint=DYNAMO_ENDPOINT,
        model_name=model_name,
        served_model_name=model_name,
        # NeMo RL's rollouts go to /v1/completions; NeMo-Gym calls chat.
        endpoint_types="chat,completions",
        discovery_backend=engine_env["DYN_DISCOVERY_BACKEND"],
        request_plane=engine_env["DYN_REQUEST_PLANE"],
        event_plane=engine_env["DYN_EVENT_PLANE"],
        disaggregation_mode=adapter.disaggregation_mode,
        # This is NeMo RL's process: Dynamo's setup would replace its root
        # handler and level, and vLLM's handlers.
        configure_logging=False,
    )
    return asyncio.get_running_loop().create_task(Worker(adapter, config).run())
