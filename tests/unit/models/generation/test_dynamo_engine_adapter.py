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
"""The prefill/decode hooks of the backend-neutral Dynamo adapter."""

import asyncio
import json
from typing import Any, Optional

import pytest

pytestmark = pytest.mark.vllm
pytest.importorskip("dynamo._core")

from dynamo.common.constants import DisaggregationMode  # noqa: E402
from dynamo.llm.exceptions import InvalidArgument  # noqa: E402

from nemo_rl.models.generation.dynamo import engine_adapter  # noqa: E402
from nemo_rl.models.generation.dynamo.rollout_client import (  # noqa: E402
    ROLLOUT_ANNOTATION,
)
from nemo_rl.models.generation.rollout_result import RolloutResult  # noqa: E402

HANDOFF = {"request_id": 7, "block_ids": [3, 4]}


class _Context:
    def id(self) -> str:
        return "req-1"


class _Adapter(engine_adapter.DynamoEngineAdapter):
    """Records what the base hands each hook."""

    def __init__(self, mode: DisaggregationMode, *, rollout_error=None) -> None:
        super().__init__(leader_worker_idx=3, disaggregation_mode=mode)
        self.calls: list[tuple[str, Any]] = []
        self._rollout_error = rollout_error

    async def serve_rollout(
        self,
        prompt_token_ids: list[int],
        *,
        greedy: bool,
        stop_strings: Optional[list[str]],
        request_id: str,
        handoff: Optional[engine_adapter.Handoff] = None,
    ) -> engine_adapter.ServedRollout:
        self.calls.append(("serve_rollout", handoff))
        if self._rollout_error is not None:
            raise self._rollout_error
        return engine_adapter.ServedRollout(
            RolloutResult([9], [-0.5], False, None, None),
            finish_reason="stop",
            usage=None,
        )

    async def serve_prefill(
        self,
        prompt_token_ids: list[int],
        *,
        greedy: bool,
        stop_strings: Optional[list[str]],
        request_id: str,
    ) -> engine_adapter.Handoff:
        self.calls.append(("serve_prefill", (prompt_token_ids, greedy, request_id)))
        return dict(HANDOFF)

    async def release_handoff(self, handoff: engine_adapter.Handoff) -> None:
        self.calls.append(("release_handoff", handoff))

    async def start(self, worker_id: int) -> engine_adapter.EngineConfig:
        raise AssertionError("not started in these tests")

    async def abort(self, context) -> None:
        raise AssertionError("not aborted in these tests")

    async def kv_event_sources(self) -> list:
        return []


def _request(prompt: list[int], **extra) -> dict:
    rollout = {"greedy": True, "stop_strings": None}
    return {
        "token_ids": prompt,
        "annotations": [ROLLOUT_ANNOTATION + json.dumps(rollout)],
        **extra,
    }


def _collect(adapter, request) -> list[dict]:
    async def run():
        return [chunk async for chunk in adapter.generate(request, _Context())]

    return asyncio.run(run())


def test_an_aggregated_engine_serves_whole_rollouts_with_no_handoff() -> None:
    adapter = _Adapter(DisaggregationMode.AGGREGATED)

    (chunk,) = _collect(adapter, _request([1, 2]))

    assert adapter.calls == [("serve_rollout", None)]
    assert chunk["token_ids"] == [9]
    assert chunk["engine_data"]["gen_leader_worker_idx"] == 3


def test_a_prefill_engine_returns_the_handoff_for_decode() -> None:
    adapter = _Adapter(DisaggregationMode.PREFILL)

    (chunk,) = _collect(adapter, _request([1, 2, 3]))

    assert adapter.calls == [("serve_prefill", ([1, 2, 3], True, "req-1"))]
    assert chunk["disaggregated_params"] == HANDOFF
    # Dynamo reads the handoff only off a "length" chunk with no tokens.
    assert chunk["token_ids"] == [] and chunk["finish_reason"] == "length"
    assert chunk["completion_usage"]["prompt_tokens"] == 3


@pytest.mark.parametrize("rollout_error", [None, RuntimeError("engine died")])
def test_a_decode_engine_reads_the_handoff_and_releases_it_once(rollout_error):
    adapter = _Adapter(DisaggregationMode.DECODE, rollout_error=rollout_error)
    request = _request([1, 2], prefill_result={"disaggregated_params": HANDOFF})

    if rollout_error is None:
        _collect(adapter, request)
    else:
        with pytest.raises(InvalidArgument, match="engine died"):
            _collect(adapter, request)

    assert adapter.calls == [
        ("serve_rollout", HANDOFF),
        ("release_handoff", HANDOFF),
    ]


def test_a_decode_engine_refuses_a_rollout_prefill_never_saw() -> None:
    adapter = _Adapter(DisaggregationMode.DECODE)

    with pytest.raises(ValueError, match="prefill_result"):
        _collect(adapter, _request([1, 2]))
    assert adapter.calls == []


def test_a_backend_without_prefill_refuses_prefill_requests() -> None:
    class _NoPrefill(_Adapter):
        serve_prefill = engine_adapter.DynamoEngineAdapter.serve_prefill

    adapter = _NoPrefill(DisaggregationMode.PREFILL)

    with pytest.raises(InvalidArgument, match="does not serve the prefill half"):
        _collect(adapter, _request([1]))


@pytest.mark.parametrize(
    ("mode", "component"),
    [
        (DisaggregationMode.AGGREGATED, "backend"),
        (DisaggregationMode.PREFILL, "prefill"),
        (DisaggregationMode.DECODE, "backend"),
    ],
)
def test_engines_register_as_their_half_under_their_component(
    monkeypatch, mode, component
) -> None:
    configs = []

    class _Worker:
        def __init__(self, engine, config):
            configs.append(config)

        async def run(self):
            return None

    monkeypatch.setattr(engine_adapter, "Worker", _Worker)
    env = {
        "DYN_DISCOVERY_BACKEND": "etcd",
        "DYN_REQUEST_PLANE": "tcp",
        "DYN_EVENT_PLANE": "nats",
    }
    for key, value in env.items():
        # Registered here so the variables the engine exports are restored.
        monkeypatch.setenv(key, value)

    async def start():
        await engine_adapter.start_dynamo_endpoint(
            _Adapter(mode),
            model_name="model",
            namespace="ns",
            engine_env=env,
            component=component,
        )

    asyncio.run(start())

    (config,) = configs
    assert config.disaggregation_mode is mode
    assert config.component == component
