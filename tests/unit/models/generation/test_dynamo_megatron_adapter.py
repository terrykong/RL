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
"""The Megatron Dynamo adapter over a fake InferenceClient.

The host is a real ``MegatronGenerationMixin`` built without its constructor,
so rollouts run the native worker's own sampling and generate steps.
"""

import asyncio
import inspect
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.mcore
pytest.importorskip("dynamo._core")

from dynamo.common.backend.publisher import PushSource  # noqa: E402
from dynamo.common.constants import DisaggregationMode  # noqa: E402
from megatron.core.inference.inference_client import InferenceClient  # noqa: E402

from nemo_rl.models.generation.dynamo.megatron.adapter import (  # noqa: E402
    DynamoMegatronAdapter,
)
from nemo_rl.models.generation.megatron.megatron_worker import (  # noqa: E402
    MegatronGenerationMixin,
)

MAX_NEW_TOKENS = 4
EOD = 2


class _Context:
    def __init__(self, request_id: str = "req-1") -> None:
        self._id = request_id

    def id(self) -> str:
        return self._id


class _Client:
    """The InferenceClient calls the adapter makes, with their real signatures."""

    def __init__(self) -> None:
        self.next_request_id = 0
        self.coordinator_instance_id = "coord-1"
        self.submitted: list[tuple[list[int], object]] = []
        self.futures: dict[int, asyncio.Future] = {}
        self.aborted: list[int] = []
        self.released: list[int] = []

    def add_request_with_id(
        self, prompt, sampling_params, *, multi_modal_data=None, offload_params=None
    ):
        request_id = self.next_request_id
        self.next_request_id += 1
        self.submitted.append((prompt, sampling_params))
        future = asyncio.get_running_loop().create_future()
        self.futures[request_id] = future
        return request_id, future

    def add_request_with_kv_handoff_streaming(
        self, prompt, sampling_params, kv_meta, src_block_ids
    ):
        raise AssertionError("decode must claim the handoff before reading it")

    def abort_request(self, request_id) -> None:
        self.aborted.append(request_id)

    def release_handoff(self, request_id) -> None:
        self.released.append(request_id)


def test_the_fake_client_matches_the_real_one() -> None:
    for name in (
        "add_request_with_id",
        "add_request_with_kv_handoff_streaming",
        "abort_request",
        "release_handoff",
    ):
        fake = list(inspect.signature(getattr(_Client, name)).parameters)
        real = list(inspect.signature(getattr(InferenceClient, name)).parameters)
        assert fake == real, name


def _engine(*, prefix_caching: bool):
    context = SimpleNamespace(
        max_sequence_length=64,
        block_size_tokens=16,
        kv_block_allocator=SimpleNamespace(pool_size=101),
        max_requests=8,
        max_tokens=512,
        enable_prefix_caching=prefix_caching,
        config=SimpleNamespace(prefix_caching_coordinator_policy=None),
        dynamo_helper=SimpleNamespace(listeners=[]),
    )
    context.dynamo_helper.add_kv_event_listener = context.dynamo_helper.listeners.append
    return SimpleNamespace(
        context=context, controller=SimpleNamespace(tokenizer=SimpleNamespace(bos=1))
    )


def _host(monkeypatch, *, prefix_caching: bool = True) -> MegatronGenerationMixin:
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    host = object.__new__(MegatronGenerationMixin)
    host.cfg = {
        "generation": {
            "temperature": 1.0,
            "top_k": None,
            "top_p": 1.0,
            "max_new_tokens": MAX_NEW_TOKENS,
            "stop_strings": ["</s>"],
        }
    }
    host.megatron_tokenizer = SimpleNamespace(eod=EOD)
    host.inference_client = _Client()
    host.coordinator_addr = "tcp://10.0.0.5:6000"
    host.dynamic_inference_engine = _engine(prefix_caching=prefix_caching)
    return host


def _adapter(host, role: str) -> DynamoMegatronAdapter:
    return DynamoMegatronAdapter(
        host, leader_worker_idx=0, role=role, model_name="Qwen/Qwen3-0.6B"
    )


async def _submitted(client: _Client, count: int = 1) -> None:
    """Let the adapter's tasks run until ``count`` requests reached the client."""
    while len(client.futures) < count:
        await asyncio.sleep(0)


def _finished(tokens, *, disaggregated_params=None) -> SimpleNamespace:
    return SimpleNamespace(
        generated_tokens=tokens,
        generated_log_probs=[-0.25] * len(tokens),
        disaggregated_params=disaggregated_params,
    )


@pytest.mark.parametrize(
    ("tokens", "finish_reason"),
    [([5, 6], "stop"), ([5, 6, 7, 8], "length")],
)
def test_rollouts_run_the_native_generate_step(monkeypatch, tokens, finish_reason):
    host = _host(monkeypatch)
    adapter = _adapter(host, "decode")
    client = host.inference_client

    async def run():
        serve = asyncio.ensure_future(
            adapter.serve_rollout(
                [1, 2, 3], greedy=True, stop_strings=["\n"], request_id="req-1"
            )
        )
        await _submitted(client)
        client.futures[0].set_result(_finished(tokens))
        return await serve

    served = asyncio.run(run())

    ((prompt, params),) = client.submitted
    # The SamplingParams generate_async builds for this sample.
    expected = host._build_sampling_params(True, host._merge_stop_strings([["\n"]]))
    assert prompt == [1, 2, 3]
    assert params.serialize() == expected.serialize()
    assert served.result.token_ids == tokens
    assert served.result.logprobs == [-0.25] * len(tokens)
    assert served.finish_reason == finish_reason
    assert served.result.truncated is (finish_reason == "length")
    assert served.usage["total_tokens"] == 3 + len(tokens)


def test_abort_cancels_the_rollout_and_its_megatron_request(monkeypatch):
    host = _host(monkeypatch)
    adapter = _adapter(host, "decode")

    async def run():
        serve = asyncio.ensure_future(
            adapter.serve_rollout(
                [1, 2], greedy=False, stop_strings=None, request_id="req-1"
            )
        )
        await _submitted(host.inference_client)
        await adapter.abort(_Context("req-1"))
        with pytest.raises(asyncio.CancelledError):
            await serve

    asyncio.run(run())

    # The native step aborts the request it submitted when cancelled.
    assert host.inference_client.aborted == [0]


def test_prefill_pins_the_kv_and_says_where_decode_releases_it(monkeypatch):
    host = _host(monkeypatch)
    adapter = _adapter(host, "prefill")
    assert adapter.disaggregation_mode is DisaggregationMode.PREFILL
    client = host.inference_client
    pinned = {"request_id": 11, "block_ids": [4, 5], "kv_meta": {"layout": "x"}}

    async def run():
        serve = asyncio.ensure_future(
            adapter.serve_prefill(
                [1, 2, 3], greedy=True, stop_strings=None, request_id="req-1"
            )
        )
        await _submitted(client)
        client.futures[0].set_result(_finished([], disaggregated_params=pinned))
        return await serve

    handoff = asyncio.run(run())

    ((_, params),) = client.submitted
    assert params.do_kv_handoff is True and params.num_tokens_to_generate == 0
    assert handoff == {
        **pinned,
        "release": {
            "coordinator_addr": "tcp://10.0.0.5:6000",
            "coordinator_instance_id": "coord-1",
            "request_id": 11,
        },
    }


def test_a_cancelled_prefill_frees_its_kv_once_it_finishes(monkeypatch):
    host = _host(monkeypatch)
    adapter = _adapter(host, "prefill")
    client = host.inference_client

    async def run():
        serve = asyncio.ensure_future(
            adapter.serve_prefill([1], greedy=True, stop_strings=None, request_id="r")
        )
        await _submitted(client)
        await adapter.abort(_Context("r"))
        with pytest.raises(asyncio.CancelledError):
            await serve
        # Prefill still runs to the end, and pins its KV for no one.
        assert client.released == []
        client.futures[0].set_result(
            _finished([], disaggregated_params={"request_id": 11})
        )
        await asyncio.sleep(0)

    asyncio.run(run())

    assert client.released == [11]


def test_decode_claims_the_handoff_before_reading_it(monkeypatch):
    host = _host(monkeypatch)
    adapter = _adapter(host, "decode")
    handoff = {
        "request_id": 11,
        "block_ids": [4, 5],
        "kv_meta": {},
        "release": {
            "coordinator_addr": "tcp://10.0.0.6:6000",
            "coordinator_instance_id": "coord-1",
            "request_id": 11,
        },
    }

    async def run():
        with pytest.raises(NotImplementedError, match="_claim_remote_handoff"):
            await adapter.serve_rollout(
                [1, 2], greedy=True, stop_strings=None, request_id="r", handoff=handoff
            )
        # Never claimed, so this engine has nothing to release.
        await adapter.release_handoff(handoff)

    asyncio.run(run())


def test_start_registers_the_engines_kv_capacity(monkeypatch):
    adapter = _adapter(_host(monkeypatch), "prefill")

    config = asyncio.run(adapter.start(worker_id=5))

    assert config.model == config.served_model_name == "Qwen/Qwen3-0.6B"
    assert config.llm.context_length == 64
    assert config.llm.kv_cache_block_size == 16
    # The allocator's last block is a dummy, not KV storage.
    assert config.llm.total_kv_blocks == 100
    assert (config.llm.max_num_seqs, config.llm.max_num_batched_tokens) == (8, 512)


class _Publisher:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def publish_stored(self, **payload) -> None:
        self.events.append(("stored", payload))

    def publish_removed(self, **payload) -> None:
        self.events.append(("removed", payload))


@pytest.mark.parametrize(
    ("role", "prefix_caching", "publishes"),
    [("prefill", True, True), ("prefill", False, False), ("decode", True, False)],
)
def test_only_prefill_engines_publish_kv_events(
    monkeypatch, role, prefix_caching, publishes
):
    host = _host(monkeypatch, prefix_caching=prefix_caching)
    adapter = _adapter(host, role)

    sources = asyncio.run(adapter.kv_event_sources())

    if not publishes:
        assert sources == []
        return
    (source,) = sources
    assert isinstance(source, PushSource) and source.dp_rank == 0
    publisher = _Publisher()
    source.on_ready(publisher)
    (listener,) = host.dynamic_inference_engine.context.dynamo_helper.listeners
    listener("stored", {"block_hashes": [1, 2], "parent_hash": None})
    listener("removed", {"block_hashes": [1]})
    listener("cleared", {})
    assert publisher.events == [
        ("stored", {"block_hashes": [1, 2], "parent_hash": None}),
        ("removed", {"block_hashes": [1]}),
        # Dynamo has no clear-all: what is still published is removed.
        ("removed", {"block_hashes": [2]}),
    ]
