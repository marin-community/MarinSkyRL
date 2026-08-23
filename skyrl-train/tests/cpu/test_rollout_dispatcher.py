import ast
import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from unittest.mock import Mock

import pytest
import ray
from omegaconf import OmegaConf

from skyrl_train.trajectory_runners.harbor.rollout_dispatcher import (
    RETAINED_RUNNER_NAME,
    RolloutCoordinatorRPCTimeoutError,
    RolloutDispatcher,
)


async def _noop_rpc() -> dict:
    return {"response_ids": [[1]], "rollout_metrics": {}}


async def _completed(batch: dict) -> dict:
    return batch


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


def _dispatcher(actor: object, *, timeout: float) -> RolloutDispatcher:
    dispatcher = RolloutDispatcher(
        cfg=OmegaConf.create({}),
        trajectory_runner_cfg=OmegaConf.create({}),
        terminal_bench_cfg=OmegaConf.create({}),
        num_coordinators=1,
        cpus_per_coordinator=1,
        coordinator_rpc_timeout=timeout,
    )
    dispatcher._actors = [actor]
    return dispatcher


@pytest.mark.asyncio
async def test_coordinator_rpc_returns_trajectory_batch():
    expected = {"response_ids": [[1]], "rollout_metrics": {}}

    async def completed_rpc():
        return expected

    dispatcher = _dispatcher(_Coordinator(completed_rpc), timeout=1)

    assert await dispatcher.run({"prompts": ["task"]}) is expected


@pytest.mark.asyncio
async def test_coordinator_rpc_timeout_does_not_cancel_remote_work(ray_init):
    actor = _BlockingCoordinator.remote()
    dispatcher = _dispatcher(actor, timeout=0.1)

    with pytest.raises(RolloutCoordinatorRPCTimeoutError):
        await dispatcher.run({"prompts": ["task"]})

    await actor.release.remote()
    assert await actor.wait_for_completion.remote() is False


@pytest.mark.asyncio
async def test_coordinator_rpc_preserves_remote_timeout_error():
    remote_error = TimeoutError("remote post-processing timed out")

    async def failed_rpc():
        raise remote_error

    dispatcher = _dispatcher(_Coordinator(failed_rpc), timeout=1)

    with pytest.raises(TimeoutError) as raised:
        await dispatcher.run({"prompts": ["task"]})

    assert raised.value is remote_error


@pytest.mark.asyncio
async def test_the_sink_binds_to_the_runner_that_produces_trajectories():
    """``runner_name`` is stamped on every retained trajectory, so it must not name this proxy."""
    sink = Mock()
    dispatcher = _dispatcher(_Coordinator(_noop_rpc), timeout=1)

    dispatcher.set_trajectory_sink(sink)

    sink.bind_runner.assert_called_once_with(RETAINED_RUNNER_NAME)


@pytest.mark.asyncio
async def test_run_retains_the_batch_its_coordinator_returned():
    expected = {"response_ids": [[1]], "rollout_metrics": {}}
    sink = Mock()
    sink.retain.return_value = {}
    dispatcher = _dispatcher(_Coordinator(lambda: _completed(expected)), timeout=1)
    dispatcher.set_trajectory_sink(sink)
    input_batch = {"prompts": ["task"], "batch_metadata": None}

    assert await dispatcher.run(input_batch) is expected
    sink.retain.assert_called_once_with(input_batch, expected)


@pytest.mark.asyncio
async def test_retention_metrics_reach_the_returned_batch():
    expected = {"response_ids": [[1]], "rollout_metrics": {"generate/existing": 1.0}}
    sink = Mock()
    sink.retain.return_value = {"generate/retained_trajectories": 4.0}
    dispatcher = _dispatcher(_Coordinator(lambda: _completed(expected)), timeout=1)
    dispatcher.set_trajectory_sink(sink)

    returned = await dispatcher.run({"prompts": ["task"], "batch_metadata": None})

    assert returned["rollout_metrics"] == {
        "generate/existing": 1.0,
        "generate/retained_trajectories": 4.0,
    }


def test_the_retained_runner_name_still_matches_a_real_class():
    """The name is a literal because harbor does not import off Linux; guard it against a rename."""
    runner = Path(__file__).resolve().parents[2] / "skyrl_train" / "trajectory_runners" / "harbor" / "runner.py"
    classes = {node.name for node in ast.walk(ast.parse(runner.read_text())) if isinstance(node, ast.ClassDef)}
    assert RETAINED_RUNNER_NAME in classes, f"{RETAINED_RUNNER_NAME} is not a class in {runner.name}: {sorted(classes)}"
