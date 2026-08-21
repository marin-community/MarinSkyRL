import asyncio
from collections.abc import Awaitable, Callable

import pytest
import ray
from omegaconf import OmegaConf

from skyrl_train.trajectory_runners.harbor.rollout_dispatcher import (
    RolloutCoordinatorRPCTimeoutError,
    RolloutDispatcher,
)


class _RemoteMethod:
    def __init__(self, call: Callable[[], Awaitable[dict]]):
        self._call = call

    def remote(self, *_args):
        return self._call()


class _Coordinator:
    def __init__(self, call: Callable[[], Awaitable[dict]]):
        self.run_shard = _RemoteMethod(call)


@ray.remote
class _BlockingCoordinator:
    def __init__(self):
        self._release = asyncio.Event()
        self._finished = asyncio.Event()
        self._cancelled = False

    async def run_shard(self, *_args):
        try:
            await self._release.wait()
            return {"response_ids": [[1]], "rollout_metrics": {}}
        except asyncio.CancelledError:
            self._cancelled = True
            raise
        finally:
            self._finished.set()

    async def release(self):
        self._release.set()

    async def wait_for_completion(self):
        await self._finished.wait()
        return self._cancelled


def _dispatcher(call: Callable[[], Awaitable[dict]], *, timeout: float) -> RolloutDispatcher:
    dispatcher = RolloutDispatcher(
        cfg=OmegaConf.create({}),
        trajectory_runner_cfg=OmegaConf.create({}),
        terminal_bench_cfg=OmegaConf.create({}),
        num_coordinators=1,
        cpus_per_coordinator=1,
        coordinator_rpc_timeout=timeout,
    )
    dispatcher._actors = [_Coordinator(call)]
    return dispatcher


@pytest.mark.asyncio
async def test_coordinator_rpc_returns_trajectory_batch():
    expected = {"response_ids": [[1]], "rollout_metrics": {}}

    async def completed_rpc():
        return expected

    dispatcher = _dispatcher(completed_rpc, timeout=1)

    assert await dispatcher.run({"prompts": ["task"]}) is expected


@pytest.mark.asyncio
async def test_coordinator_rpc_timeout_does_not_cancel_remote_work(ray_init):
    actor = _BlockingCoordinator.remote()
    dispatcher = _dispatcher(lambda: asyncio.sleep(0), timeout=0.1)
    dispatcher._actors = [actor]

    with pytest.raises(RolloutCoordinatorRPCTimeoutError):
        await dispatcher.run({"prompts": ["task"]})

    await actor.release.remote()
    assert await actor.wait_for_completion.remote() is False


@pytest.mark.asyncio
async def test_coordinator_rpc_preserves_remote_timeout_error():
    remote_error = TimeoutError("remote post-processing timed out")

    async def failed_rpc():
        raise remote_error

    dispatcher = _dispatcher(failed_rpc, timeout=1)

    with pytest.raises(TimeoutError) as raised:
        await dispatcher.run({"prompts": ["task"]})

    assert raised.value is remote_error
