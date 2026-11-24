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
import asyncio
import sys
import threading
import time
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nemo_rl.models.generation.megatron import dynamo_worker_mixin
from nemo_rl.models.generation.megatron.dynamo_worker_mixin import (
    MegatronDynamoGenerationMixin,
    require_megatron_dynamo_integration,
)
from nemo_rl.models.generation.megatron.megatron_worker import MegatronGenerationMixin

pytestmark = pytest.mark.mcore


class _Worker(MegatronDynamoGenerationMixin, MegatronGenerationMixin):
    """MegatronDynamoPolicyWorker's generation bases, in the order it composes them."""


def _bare_worker(role, cls=_Worker):
    worker = object.__new__(cls)
    worker.cfg = {"generation": {}}
    if role is not None:
        worker.cfg["generation"]["_dynamo_role"] = role
    return worker


@pytest.fixture
def engine_modules(monkeypatch):
    """Fake the plain and the Dynamo handoff engine classes the hooks pick between."""
    plain = MagicMock(name="DynamicInferenceEngine")
    handoff = MagicMock(name="DynamoDynamicInferenceEngine")
    monkeypatch.setattr(
        "megatron.core.inference.engines.dynamic_engine.DynamicInferenceEngine", plain
    )
    for name in (
        "megatron.inference",
        "megatron.inference.integrations",
        "megatron.inference.integrations.dynamo",
    ):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    module = types.ModuleType("megatron.inference.integrations.dynamo.dynamic_engine")
    module.DynamoDynamicInferenceEngine = handoff
    monkeypatch.setitem(
        sys.modules, "megatron.inference.integrations.dynamo.dynamic_engine", module
    )
    monkeypatch.setattr(dynamo_worker_mixin, "_get_node_ip_local", lambda: "10.0.0.5")
    return plain, handoff


def test_plain_worker_builds_the_plain_engine(engine_modules):
    plain, handoff = engine_modules
    worker = _bare_worker(role=None, cls=MegatronGenerationMixin)
    controller, context = MagicMock(), MagicMock()

    assert worker._inference_config_overrides() == {}
    assert worker._coordinator_kwargs() == {}
    assert worker._build_dynamic_inference_engine(controller, context) is (
        plain.return_value
    )
    plain.assert_called_once_with(controller, context)
    handoff.assert_not_called()


def test_dynamo_worker_needs_its_role():
    worker = _bare_worker(role=None)

    with pytest.raises(RuntimeError, match="_dynamo_role"):
        worker._dynamo_role


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_dynamo_worker_builds_the_handoff_engine(engine_modules, role):
    plain, handoff = engine_modules
    worker = _bare_worker(role=role)
    controller, context = MagicMock(), MagicMock()

    assert worker._dynamo_role == role
    # The peer role lives in other processes: NIXL carries the handoff and the
    # coordinator must listen on a routable address. The role reaches
    # Megatron-LM, which drops the Mamba prefix cache for decode.
    assert worker._inference_config_overrides() == {
        "kv_transport_backend": "nixl",
        "disaggregation_role": role,
    }
    assert worker._coordinator_kwargs() == {"hostname": "10.0.0.5"}

    engine = worker._build_dynamic_inference_engine(controller, context)

    assert engine is handoff.return_value
    handoff.assert_called_once_with(controller, context)
    engine.setup_kv_transfer.assert_called_once_with(
        role=role, require_handoff_claim=True
    )
    plain.assert_not_called()


def test_engine_state_reset_clears_the_endpoint():
    worker = _bare_worker(role="prefill")
    worker._dynamo_endpoint_future = MagicMock()

    worker._init_inference_engine_state()

    assert worker._dynamo_endpoint_future is None
    assert worker.dynamic_inference_engine is None


# ---------------------------------------------------------------------------
# Rank zero serves the engine through DynamoMegatronAdapter, in-process
# ---------------------------------------------------------------------------


class _LoopThread:
    """A running asyncio loop on a thread, standing in for the worker's inference loop."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self._thread.start()

    def close(self) -> None:
        async def _drain() -> None:
            # Run every cancelled or still-running task to completion, so none is
            # garbage-collected pending (and logged) during a later test.
            tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

        asyncio.run_coroutine_threadsafe(_drain(), self.loop).result(timeout=5)
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=5)
        self.loop.close()


ENGINE_ENV = {
    "DYN_DISCOVERY_BACKEND": "etcd",
    "DYN_REQUEST_PLANE": "tcp",
    "DYN_EVENT_PLANE": "nats",
    "DYN_SYSTEM_PORT": "0",
}


@pytest.fixture
def dynamo_endpoint_env(monkeypatch):
    """A bare rank-zero worker whose engine is built, and a fake Dynamo ``Worker``.

    The adapter and ``engine_adapter.start_dynamo_endpoint`` are the real ones;
    only Dynamo's Rust-backed ``Worker`` is faked. Yields a namespace with the
    worker, the fake's records, and a knob that makes ``Worker.run`` return
    (``finish``) or raise (``fail``).
    """
    pytest.importorskip("dynamo._core")
    from nemo_rl.models.generation.dynamo import engine_adapter

    records = SimpleNamespace(workers=[], run_loop=None, run_cancelled=False)
    control = SimpleNamespace(finish=None, fail=None)

    class _DynamoWorker:
        def __init__(self, engine, config):
            self.engine = engine
            self.config = config
            records.workers.append(self)

        async def run(self):
            records.run_loop = asyncio.get_running_loop()
            try:
                while control.finish is None and control.fail is None:
                    await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                records.run_cancelled = True
                raise
            if control.fail is not None:
                raise control.fail

    monkeypatch.setattr(engine_adapter, "Worker", _DynamoWorker)
    for key, value in ENGINE_ENV.items():
        # Registered here so the variables the engine exports are restored.
        monkeypatch.setenv(key, value)

    loop_thread = _LoopThread()
    worker = _bare_worker(role="prefill")
    worker.rank = 0
    worker._inference_engine_initialized = True
    worker._dynamo_endpoint_future = None
    worker._inference_loop = loop_thread.loop
    yield SimpleNamespace(worker=worker, records=records, control=control)
    worker.stop_dynamo_endpoint()
    loop_thread.close()


def _start(env, role="prefill"):
    env.worker.cfg["generation"]["_dynamo_role"] = role
    return env.worker.start_dynamo_endpoint(
        namespace="ns",
        engine_env=ENGINE_ENV,
        model_name="Qwen/Qwen3-0.6B",
        leader_worker_idx=2,
    )


def _wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


@pytest.mark.parametrize(
    ("role", "component"), [("prefill", "prefill"), ("decode", "backend")]
)
def test_start_dynamo_endpoint_serves_the_adapter_on_the_inference_loop(
    dynamo_endpoint_env, role, component
):
    from dynamo.common.constants import DisaggregationMode

    from nemo_rl.models.generation.dynamo.megatron.adapter import (
        DynamoMegatronAdapter,
    )

    env = dynamo_endpoint_env
    assert _start(env, role=role) is None
    _wait_until(lambda: env.records.run_loop is not None)

    (dynamo_worker,) = env.records.workers
    # NeMo RL's own adapter, over this worker's engine, registered as its role.
    adapter = dynamo_worker.engine
    assert isinstance(adapter, DynamoMegatronAdapter) and adapter._host is env.worker
    assert adapter._leader_worker_idx == 2
    config = dynamo_worker.config
    assert config.disaggregation_mode is DisaggregationMode[role.upper()]
    assert config.component == component
    assert (config.namespace, config.model_name) == ("ns", "Qwen/Qwen3-0.6B")
    assert config.configure_logging is False
    # The runtime's environment as is: the OS picks the system port.
    import os

    assert os.environ["DYN_SYSTEM_PORT"] == "0"
    # It runs on the inference loop, where the worker's InferenceClient lives.
    assert env.records.run_loop is env.worker._inference_loop

    # A repeat start while it serves starts nothing new.
    _start(env, role=role)
    assert len(env.records.workers) == 1


def test_stop_dynamo_endpoint_cancels_the_adapter_task(dynamo_endpoint_env):
    env = dynamo_endpoint_env
    _start(env)
    future = env.worker._dynamo_endpoint_future
    _wait_until(lambda: env.records.run_loop is not None)

    env.worker.stop_dynamo_endpoint()

    # Worker.run itself is cancelled on the inference loop.
    _wait_until(lambda: env.records.run_cancelled)
    assert env.worker.dynamo_endpoint_error() == "cancelled"
    # The stopped task stays recorded: Dynamo's runtime serves one Worker per
    # process, so a second start in this process is refused.
    assert env.worker._dynamo_endpoint_future is future
    with pytest.raises(RuntimeError, match=r"\(cancelled\).*restart the engine"):
        _start(env)
    assert len(env.records.workers) == 1


def test_a_failed_endpoint_reports_its_error_and_refuses_a_restart(
    dynamo_endpoint_env,
):
    env = dynamo_endpoint_env
    _start(env)
    _wait_until(lambda: env.records.run_loop is not None)
    assert env.worker.dynamo_endpoint_error() is None

    env.control.fail = ConnectionError("etcd unreachable")
    _wait_until(env.worker._dynamo_endpoint_future.done)

    assert env.worker.dynamo_endpoint_error() == repr(env.control.fail)
    with pytest.raises(RuntimeError, match="etcd unreachable"):
        _start(env)
    assert len(env.records.workers) == 1


def test_an_endpoint_that_returned_reports_exited(dynamo_endpoint_env):
    env = dynamo_endpoint_env
    _start(env)
    env.control.finish = True
    _wait_until(env.worker._dynamo_endpoint_future.done)

    assert env.worker.dynamo_endpoint_error() == "exited"
    with pytest.raises(RuntimeError, match=r"\(exited\)"):
        _start(env)


def test_dynamo_endpoint_needs_a_built_engine(dynamo_endpoint_env):
    env = dynamo_endpoint_env
    env.worker._inference_engine_initialized = False
    with pytest.raises(RuntimeError, match="needs the inference engine"):
        _start(env)
    assert env.records.workers == []


def test_dynamo_worker_requires_the_megatron_dynamo_integration(monkeypatch):
    module = "megatron.inference.integrations.dynamo.dynamic_engine"
    monkeypatch.setitem(sys.modules, module, types.ModuleType(module))
    require_megatron_dynamo_integration()

    # None in sys.modules makes the import raise ImportError.
    monkeypatch.setitem(sys.modules, module, None)
    with pytest.raises(RuntimeError, match="megatron.inference.integrations.dynamo"):
        require_megatron_dynamo_integration()
