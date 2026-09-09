"""Retain every DP worker's read-only diagnostic utility response."""

import asyncio


async def read_all_receiver_workers(engine, method: str):
    """Avoid DPLBAsyncMPClient's deliberate first-core-only utility return.

    The pinned vLLM client dispatches collective_rpc to all cores but its public
    call_utility_async returns result[0]. Readbacks need each original result.
    This uses the same per-core dispatch without changing any worker operation.
    """
    dp_size = engine.vllm_config.parallel_config.data_parallel_size
    if dp_size == 1:
        return await engine.collective_rpc(method)
    core = engine.engine_core
    if len(core.core_engines) != dp_size:
        raise ValueError("Receiver readback requires every configured DP core")
    per_core = await asyncio.gather(
        *[
            core._call_utility_async("collective_rpc", method, None, (), None, engine=identity)
            for identity in core.core_engines
        ]
    )
    if any(not workers for workers in per_core):
        raise ValueError("Receiver core returned no worker readback")
    return [worker for workers in per_core for worker in workers]
