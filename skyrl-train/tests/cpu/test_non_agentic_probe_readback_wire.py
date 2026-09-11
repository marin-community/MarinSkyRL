"""The probe's readback request must survive vLLM's strict msgspec utility codec.

The pinned fork's `vllm.v1.serial_utils` cannot be imported without `vllm._C`,
so the encoder below reproduces its enc_hook: without
VLLM_ALLOW_INSECURE_SERIALIZATION every object msgpack cannot encode natively
raises TypeError. Tensors and multimodal items never appear in these requests.
"""

import asyncio
from types import SimpleNamespace

import msgspec
import pytest
import torch

from skyrl_train.entrypoints.non_agentic_probe_worker import (
    PROBE_WORKER_EXTENSION,
    WORKER_MEMORY,
    ProbeWorkerExtension,
    resolve_readback,
    worker_memory,
)
from skyrl_train.weight_sync.receiver_readback_rpc import read_all_receiver_workers
from tests.cpu.test_non_agentic_native_probe_engine import native_probe  # noqa: F401


def strict_enc_hook(obj):
    raise TypeError(f"Object of type {type(obj)} is not serializable")


ENCODER = msgspec.msgpack.Encoder(enc_hook=strict_enc_hook)
DECODER = msgspec.msgpack.Decoder()


def capturing_engine(cores: int):
    """Fake engine whose cores record the (utility, args) handed to _call_utility_async."""
    requests = []

    class CoreBoundary:
        core_engines = [bytes([index]) for index in range(cores)]

        async def _call_utility_async(self, utility, *utility_args, engine):
            requests.append((utility, utility_args))
            index = engine[0]
            return [{"pid": index + 1, "data_parallel_rank": index, "gpu_uuid": f"gpu-{index}"}]

    engine = SimpleNamespace(
        engine_core=CoreBoundary(),
        vllm_config=SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_size=cores)),
    )
    return engine, requests


def captured_utility_requests(method, args=()):
    engine, requests = capturing_engine(2)
    asyncio.run(read_all_receiver_workers(engine, method, args))
    return requests


def test_probe_readback_request_encodes_and_resolves_on_the_worker(native_probe):  # noqa: F811
    engine, requests = capturing_engine(8)
    asyncio.run(native_probe.all_worker_memory(engine))
    assert len(requests) == 8
    for utility, args in requests:
        # core_client._call_utility_async encodes (client_index, call_id, method, args)
        wire = DECODER.decode(ENCODER.encode((0, 1, utility, args)))
        assert wire == [0, 1, "collective_rpc", ["read_probe_worker", None, [WORKER_MEMORY], None]]
    assert resolve_readback(wire[3][2][0]) is worker_memory
    assert PROBE_WORKER_EXTENSION.rsplit(".", 1)[1] == ProbeWorkerExtension.__name__


def test_callable_readback_is_refused_before_the_codec():
    with pytest.raises(TypeError, match="worker method name"):
        captured_utility_requests(worker_memory)


def test_unknown_readback_spec_is_rejected_on_the_worker():
    with pytest.raises(ValueError, match="Unknown probe readback"):
        resolve_readback("skyrl_train.entrypoints.non_agentic_probe_worker:resolve_readback")


def test_worker_memory_result_is_plain_data(monkeypatch):
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (11, 97))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 3)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 4)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 5)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(uuid="GPU-0"))
    worker = SimpleNamespace(
        device="cuda:0", vllm_config=SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_rank=6))
    )
    result = ProbeWorkerExtension.read_probe_worker(worker, WORKER_MEMORY)
    # UtilityResult encodes as (None, result) without insecure serialization.
    assert DECODER.decode(ENCODER.encode((None, result)))[1] == result
    assert result["data_parallel_rank"] == 6 and result["free"] == 11 and result["gpu_uuid"] == "GPU-0"
