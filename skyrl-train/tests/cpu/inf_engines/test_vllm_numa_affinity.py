import asyncio

import pytest

from skyrl_train.inference_engines.vllm.numa import set_async_worker_numa_affinity


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
