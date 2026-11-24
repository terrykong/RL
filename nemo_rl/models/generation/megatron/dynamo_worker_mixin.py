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
"""Worker-side hooks for a Megatron engine served through Dynamo."""

import asyncio
from concurrent.futures import Future as ConcurrentFuture
from typing import TYPE_CHECKING, Any, Mapping, Optional

from nemo_rl.distributed.virtual_cluster import _get_node_ip_local
from nemo_rl.models.generation.megatron.dynamo_config import DYNAMO_COMPONENT_BY_ROLE

if TYPE_CHECKING:
    from megatron.core.inference.contexts.dynamic_context import (
        DynamicInferenceContext,
    )
    from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine
    from megatron.core.inference.text_generation_controllers.text_generation_controller import (
        TextGenerationController,
    )


def require_megatron_dynamo_integration() -> None:
    """Fail before model setup if this Megatron-LM cannot serve an engine behind Dynamo.

    The engine is only built at the first refit, after the model has loaded, so
    a missing integration would otherwise surface late.
    """
    try:
        import megatron.inference.integrations.dynamo.dynamic_engine  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "A Megatron engine behind Dynamo needs a Megatron-LM with the Dynamo "
            "integration (megatron.inference.integrations.dynamo)"
        ) from e


class MegatronDynamoGenerationMixin:
    """Prefill/decode engine hooks for a Megatron worker served through Dynamo.

    Hosted by ``MegatronDynamoPolicyWorker``, the worker class the policy picks
    when ``generation._dynamo_role`` is set, so every override here applies
    unconditionally; a worker without a role is a plain ``MegatronPolicyWorker``
    and never sees this class. Compose it before ``MegatronGenerationMixin`` so
    the overrides win. ``MegatronDynamoGeneration`` sets the role per engine;
    it picks the state-handoff engine variant and its KV transfer agents.

    Borrowed from ``MegatronGenerationMixin``:

     - cfg: policy config (TypedDict).
     - rank: global rank (used for logging).
     - dynamic_inference_engine: the engine, once the first refit built it.
     - coordinator_addr: the engine's coordinator address once it listens;
       prefill handoffs name it so decode can release them.
     - _inference_loop: the engine's asyncio loop.
     - _inference_engine_initialized: whether the engine is built.
    """

    cfg: dict[str, Any]
    rank: int
    dynamic_inference_engine: Optional["DynamicInferenceEngine"]
    coordinator_addr: Optional[str]
    _inference_loop: Optional[asyncio.AbstractEventLoop]
    _inference_engine_initialized: bool

    @property
    def _dynamo_role(self) -> str:
        """``"prefill"`` or ``"decode"``: which half of the handoff this engine is."""
        role = self.cfg["generation"].get("_dynamo_role")
        if role is None:
            raise RuntimeError(
                "MegatronDynamoPolicyWorker needs policy.generation._dynamo_role; "
                "MegatronDynamoGeneration sets it per engine"
            )
        return role

    def _init_inference_engine_state(self) -> None:
        super()._init_inference_engine_state()
        # Rank zero only: the task serving this engine's Dynamo endpoint, once
        # start_dynamo_endpoint scheduled it on the inference loop.
        self._dynamo_endpoint_future: Optional[ConcurrentFuture] = None

    def _inference_config_overrides(self) -> dict[str, Any]:
        # Prefill and decode engines are separate process groups, so the
        # handoff must cross them; NIXL is the only transport that does. The
        # role sizes role-specific buffers: a decode context drops the Mamba
        # prefix cache that prefix_caching_mamba_gb gives prefill.
        return {
            "kv_transport_backend": "nixl",
            "disaggregation_role": self._dynamo_role,
        }

    def _build_dynamic_inference_engine(
        self,
        controller: "TextGenerationController",
        context: "DynamicInferenceContext",
    ) -> "DynamicInferenceEngine":
        # Prefill/decode behind Dynamo: the state-handoff engine pins prefill
        # blocks until decode releases them and imports them on decode.
        # Deferred: the Dynamo integration is optional (absent from older
        # Megatron-LM) and only Dynamo-attached engines load it.
        from megatron.inference.integrations.dynamo.dynamic_engine import (
            DynamoDynamicInferenceEngine,
        )

        engine = DynamoDynamicInferenceEngine(controller, context)
        # Collective over the engine's model-parallel ranks. The Dynamo adapter
        # claims every handoff before decode reads it, so unclaimed prefill
        # offers may expire instead of staying pinned forever.
        engine.setup_kv_transfer(role=self._dynamo_role, require_handoff_claim=True)
        return engine

    def _coordinator_kwargs(self) -> dict[str, Any]:
        # The peer role's engines connect to this coordinator from other
        # processes and nodes.
        return {"hostname": _get_node_ip_local()}

    def start_dynamo_endpoint(
        self,
        *,
        namespace: str,
        engine_env: Mapping[str, str],
        model_name: str,
        leader_worker_idx: int,
    ) -> None:
        """Serve this engine through Dynamo from this rank.

        Runs on rank zero once the engine is built (the first refit builds it).
        ``DynamoMegatronAdapter`` calls the engine in-process, through this
        worker's own ``InferenceClient`` on the inference loop, and
        ``engine_adapter.start_dynamo_endpoint`` registers it as this engine's
        role, as the vLLM worker registers its engine. A repeat call while the
        endpoint serves does nothing; once it stopped, failed or exited (see
        ``dynamo_endpoint_error``) it cannot be started again in this process
        (Dynamo's runtime serves one ``Worker`` per process).

        Args:
            namespace: Dynamo namespace shared with the frontend.
            engine_env: The managed runtime's environment for engines.
            model_name: The served model; also the name it is served under.
            leader_worker_idx: The engine's index in the fleet; rollouts it
                serves report it.

        Raises:
            RuntimeError: The engine is not built yet, or its endpoint already
                stopped serving in this process.
        """
        endpoint_error = self.dynamo_endpoint_error()
        if endpoint_error is not None:
            raise RuntimeError(
                f"This engine's Dynamo endpoint stopped serving ({endpoint_error}); "
                "Dynamo's runtime serves one Worker per process, so restart the "
                "engine instead"
            )
        if self._dynamo_endpoint_future is not None:
            return
        if not self._inference_engine_initialized:
            raise RuntimeError(
                "start_dynamo_endpoint needs the inference engine to be built "
                "(the first refit builds it)"
            )
        # Deferred: ai-dynamo is imported only by engines that serve through
        # the frontend.
        from nemo_rl.models.generation.dynamo.engine_adapter import (
            start_dynamo_endpoint,
        )
        from nemo_rl.models.generation.dynamo.megatron.adapter import (
            DynamoMegatronAdapter,
        )

        role = self._dynamo_role
        adapter = DynamoMegatronAdapter(
            self, leader_worker_idx=leader_worker_idx, role=role, model_name=model_name
        )

        async def _serve() -> None:
            # On the inference loop, where the worker's InferenceClient lives.
            await start_dynamo_endpoint(
                adapter,
                model_name=model_name,
                namespace=namespace,
                engine_env=dict(engine_env),
                component=DYNAMO_COMPONENT_BY_ROLE[role],
            )

        self._dynamo_endpoint_future = asyncio.run_coroutine_threadsafe(
            _serve(), self._inference_loop
        )
        print(f"[Rank {self.rank}] Dynamo {role} endpoint started", flush=True)

    def dynamo_endpoint_error(self) -> Optional[str]:
        """Why this engine's Dynamo endpoint stopped serving; None while it serves.

        Returns:
            None before ``start_dynamo_endpoint`` or while the endpoint task
            runs; ``"cancelled"`` after ``stop_dynamo_endpoint``; ``"exited"``
            when ``Worker.run`` returned; otherwise the ``repr`` of the
            exception the task raised.
        """
        future = self._dynamo_endpoint_future
        if future is None or not future.done():
            return None
        if future.cancelled():
            return "cancelled"
        error = future.exception()
        return "exited" if error is None else repr(error)

    def stop_dynamo_endpoint(self) -> None:
        """Cancel the Dynamo endpoint task.

        Cancelling drops Dynamo's ``Worker.run`` without its graceful path: no
        request drain and no adapter cleanup. The engine's discovery entry
        lapses when this process exits, so callers stop the engine next. The
        stopped task stays recorded, so a later start is refused.
        """
        future = self._dynamo_endpoint_future
        if future is not None:
            future.cancel()
