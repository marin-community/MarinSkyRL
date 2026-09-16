import asyncio
import threading
from types import SimpleNamespace

import pytest

from skyrl_train.inference_engines.vllm.numa import set_async_worker_numa_affinity, set_sync_worker_numa_affinity


@pytest.mark.asyncio
async def test_async_collective_rpc_completes_before_numa_initialization_returns():
    affinity_applied = asyncio.Event()

    async def collective_rpc(method):
        assert method == "set_numa_affinity"
        await asyncio.sleep(0)
        affinity_applied.set()
        return "applied"

    result = await set_async_worker_numa_affinity(collective_rpc)

    assert result == "applied"
    assert affinity_applied.is_set()


@pytest.mark.asyncio
async def test_sync_collective_rpc_runs_off_the_event_loop():
    event_loop_thread = threading.get_ident()
    rpc_thread = None

    def collective_rpc(method):
        nonlocal rpc_thread
        assert method == "set_numa_affinity"
        rpc_thread = threading.get_ident()
        return "applied"

    result = await set_sync_worker_numa_affinity(collective_rpc)

    assert result == "applied"
    assert rpc_thread != event_loop_thread


@pytest.mark.asyncio
async def test_sync_engine_reports_worker_hosts_before_weight_sync():
    pytest.importorskip("vllm")
    from skyrl_train.inference_engines.vllm.vllm_engine import VLLMInferenceEngine

    event_loop_thread = threading.get_ident()
    calls = []

    def collective_rpc(method):
        calls.append((method, threading.get_ident()))
        return ["gpu-node"]

    engine = object.__new__(VLLMInferenceEngine)
    engine.llm = SimpleNamespace(collective_rpc=collective_rpc)

    assert await engine.report_engine_hosts() == ["gpu-node"]
    assert len(calls) == 1
    assert calls[0][0] == "report_host"
    assert calls[0][1] != event_loop_thread
