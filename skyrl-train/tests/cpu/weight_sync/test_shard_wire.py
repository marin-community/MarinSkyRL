"""Shard metadata semantics, including the real vLLM codec when installed."""

import asyncio
from dataclasses import replace
import importlib
import json
from types import SimpleNamespace

import pytest
import torch

from skyrl_train.weight_sync.shard_preparation import PreparationOptions, _fresh, plan_shard_preparation
from skyrl_train.weight_sync.shard_wire import (
    SHARD_METHODS,
    call_all_shard_workers,
    decode_shard_metadata,
    encode_shard_metadata,
)
from tests.cpu.weight_sync.test_shard_preparation import endpoint, metadata_fixture
from tests.cpu.weight_sync.test_shard_session_routes import actual_methods


WorkerMethods = actual_methods("inference_engines/vllm/vllm_engine.py", "WorkerWrap", {"shard_metadata_rpc"})


def test_complete_plan_preserves_hash_memberships_and_byte_oracle():
    geometry, policy, receivers = metadata_fixture()
    options = PreparationOptions(1024, 64, 128, 0)
    copied = decode_shard_metadata(encode_shard_metadata((geometry, options, policy, receivers)))
    plan = plan_shard_preparation(*copied[2:], *copied[:2], endpoint)
    expected = plan_shard_preparation(policy, receivers, geometry, options, endpoint)
    received = decode_shard_metadata(encode_shard_metadata(plan))
    assert received == expected
    assert received.plan_id == expected.plan_id
    assert dict(received.expected_receiver_bytes) == {rank: 176 for rank in range(8, 14)}
    assert all(type(group.members) is tuple for group in received.endpoints)


@pytest.mark.parametrize(
    "value", [torch.zeros(1), object(), {True: 1}, replace(metadata_fixture()[0], policy_ranks=True)]
)
def test_wire_rejects_tensor_storage_and_invalid_metadata(value):
    with pytest.raises((TypeError, ValueError)):
        decode_shard_metadata(encode_shard_metadata(value))


@pytest.mark.parametrize("fault", ["schema", "unknown_type", "extra_field", "tuple", "boolean_dimension"])
def test_wire_rejects_corrupt_geometry(fault):
    envelope = json.loads(encode_shard_metadata(metadata_fixture()[0]))
    fields = envelope["metadata"][2]
    if fault == "schema":
        envelope["schema"] = True
    elif fault == "unknown_type":
        envelope["metadata"][1] = "LocalPreparation"
    elif fault == "extra_field":
        fields["extra"] = 1
    elif fault == "tuple":
        fields["layers_by_pp"][0] = "list"
    else:
        fields["policy_ranks"] = True
    with pytest.raises(ValueError):
        decode_shard_metadata(json.dumps(envelope).encode())


@pytest.fixture
def pinned_codec():
    # This test also runs in the pinned native image without allocating a GPU.
    codec = pytest.importorskip("vllm.v1.serial_utils")
    assert codec.envs.VLLM_ALLOW_INSECURE_SERIALIZATION is False
    engine_types = importlib.import_module("vllm.v1.engine")
    return codec, engine_types.UtilityOutput


@pytest.mark.parametrize("dp_size", [1, 2])
def test_actual_utility_codec_worker_route_preserves_all_shard_metadata(pinned_codec, dp_size):
    codec, UtilityOutput = pinned_codec
    geometry, policy, receivers = metadata_fixture()
    options = PreparationOptions(1024, 64, 128, 0)
    plan = plan_shard_preparation(policy, receivers, geometry, options, endpoint)
    worker = WorkerMethods()
    seen = []

    def invoke(*arguments):
        _fresh(worker, "fixture", arguments[0])
        assert arguments == (geometry, options, plan, tuple(policy), tuple(receivers))
        seen.append(arguments)
        return {"geometry": arguments[0], "plan": arguments[2], "bindings": tuple(receivers)}

    for method in SHARD_METHODS:
        setattr(worker, method, invoke)

    class Transport:
        engine_ranks_managed = [0]
        core_engines = [b"\x00\x00"]

        async def _call_utility_async(self, utility, *args, engine=None):
            encoded = codec.MsgpackEncoder().encode((0, 17, utility, args))
            _, call_id, utility, arguments = codec.MsgpackDecoder().decode(encoded)
            assert utility == "collective_rpc"
            method, timeout, args, kwargs = arguments
            assert timeout is None and kwargs is None
            result = [getattr(worker, method)(*args)]
            encoded = codec.MsgpackEncoder().encode(UtilityOutput(call_id, result=codec.UtilityResult(result)))
            return codec.MsgpackDecoder(UtilityOutput).decode(encoded).result.result

    core = Transport()

    async def collective_rpc(method, *, args, kwargs):
        return await core._call_utility_async("collective_rpc", method, None, args, kwargs)

    engine = SimpleNamespace(
        engine_core=core,
        collective_rpc=collective_rpc,
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(
                data_parallel_size=dp_size,
                data_parallel_size_local=1,
                local_engines_only=True,
                data_parallel_index=0,
                data_parallel_rank_local=0,
            )
        ),
    )
    for method in SHARD_METHODS:
        rows = asyncio.run(
            call_all_shard_workers(engine, method, args=(geometry, options, plan, tuple(policy), tuple(receivers)))
        )
        assert rows[0]["plan"].plan_id == plan.plan_id
        assert rows[0]["geometry"] == geometry
        assert rows[0]["bindings"] == tuple(receivers)
        assert ("receiver_transport" in rows[0]) == (dp_size > 1)
    assert len(seen) == len(SHARD_METHODS)


def test_actual_utility_codec_original_failure(pinned_codec):
    codec, _ = pinned_codec
    geometry = metadata_fixture()[0]
    message = (0, 17, "collective_rpc", ("collect_shard_receiver_preparation", None, ("fixture", geometry, 0), None))
    decoded = codec.MsgpackDecoder().decode(codec.MsgpackEncoder().encode(message))
    with pytest.raises(AttributeError, match="has no attribute 'validate'"):
        _fresh(SimpleNamespace(), "fixture", decoded[3][2][1])
