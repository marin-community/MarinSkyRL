import asyncio
from collections.abc import Awaitable, Callable

import pytest
import ray
from skyrl_train.trajectory_runners.harbor.execution import (
    ExecutionEnvironment,
    HarborRunnerSpec,
    ProcessPoolResources,
    TrajectoryWorkload,
    build_harbor_trajectory_runner,
)
from skyrl_train.trajectory_runners.harbor.rollout_dispatcher import (
    RolloutCoordinatorRPCTimeoutError,
    RolloutDispatcher,
)
from skyrl_train.trajectory_runners.types import TrajectoryID


class _RemoteMethod:
    def __init__(self, call: Callable[..., Awaitable[dict]]):
        self._call = call

    def remote(self, *args):
        return self._call(*args)


class _Coordinator:
    def __init__(self, call: Callable[..., Awaitable[dict]]):
        self.run_shard = _RemoteMethod(call)


@ray.remote
class _BlockingCoordinator:
    def __init__(self):
        self._started = asyncio.Event()

    async def run_shard(self, *_args):
        self._started.set()
        await asyncio.Event().wait()

    async def wait_for_start(self):
        await self._started.wait()


def _request(ids: list[TrajectoryID]) -> dict:
    return {
        "prompts": [f"prompt-{trajectory_id.to_string()}" for trajectory_id in ids],
        "env_classes": ["terminal" for _ in ids],
        "env_extras": [{} for _ in ids],
        "sampling_params": {},
        "trajectory_ids": ids,
        "batch_metadata": None,
    }


def _output(ids: list[TrajectoryID]) -> dict:
    values = [trajectory_id.repetition_id + (100 if trajectory_id.instance_id == "b" else 0) for trajectory_id in ids]
    return {
        "prompt_token_ids": [[value] for value in values],
        "response_ids": [[value] for value in values],
        "rewards": [float(value) for value in values],
        "loss_masks": [[1] for _ in values],
        "rollout_metrics": {},
        "rollout_logprobs": None,
        "trajectory_ids": ids,
        "actual_global_step": 7,
    }


def _dispatcher(actors: list[object], harbor_runner_spec: HarborRunnerSpec, *, timeout: float = 1) -> RolloutDispatcher:
    dispatcher = RolloutDispatcher(
        spec=harbor_runner_spec,
        resources=ProcessPoolResources(
            num_coordinators=len(actors),
            cpus_per_coordinator=1,
            executor_workers=1,
            rpc_timeout_seconds=timeout,
        ),
    )
    dispatcher._actors = actors
    return dispatcher


def test_production_harbor_workload_selects_process_isolation_before_trainer_construction(harbor_runner_spec):
    resources = ProcessPoolResources(2, 1, 4, 30)

    runner = build_harbor_trajectory_runner(
        spec=harbor_runner_spec,
        workload=TrajectoryWorkload(ExecutionEnvironment.PRODUCTION),
        tokenizer=object(),
        resources=resources,
    )

    assert isinstance(runner, RolloutDispatcher)
    assert runner._num_coordinators == 2


def test_development_harbor_workload_selects_in_process_execution():
    sentinel = object()

    class _LocalSpec:
        def build(self, tokenizer):
            assert tokenizer == "tokenizer"
            return sentinel

    runner = build_harbor_trajectory_runner(
        spec=_LocalSpec(),
        workload=TrajectoryWorkload(ExecutionEnvironment.DEVELOPMENT),
        tokenizer="tokenizer",
        resources=ProcessPoolResources(2, 1, 4, 30),
    )

    assert runner is sentinel


@pytest.mark.asyncio
async def test_dispatcher_partitions_complete_groups_and_restores_request_order(harbor_runner_spec):
    calls: list[list[str]] = []

    async def run_group(input_batch, _global_step):
        ids = input_batch["trajectory_ids"]
        calls.append([trajectory_id.to_string() for trajectory_id in ids])
        return _output(list(reversed(ids)))

    ids = [TrajectoryID("a", 0), TrajectoryID("b", 0), TrajectoryID("a", 1), TrajectoryID("b", 1)]
    dispatcher = _dispatcher([_Coordinator(run_group), _Coordinator(run_group)], harbor_runner_spec)

    result = await dispatcher.run(_request(ids))

    assert calls == [["a_0", "a_1"], ["b_0", "b_1"]]
    assert result["trajectory_ids"] == ids
    assert result["response_ids"] == [[0], [100], [1], [101]]


@pytest.mark.asyncio
async def test_dispatcher_concatenates_fully_excluded_group_without_logprobs(harbor_runner_spec):
    harbor_runner_spec.config.trainer.algorithm.use_tis = True

    async def run_group(input_batch, _global_step):
        ids = input_batch["trajectory_ids"]
        output = _output(ids)
        if ids[0].instance_id == "masked":
            output["loss_masks"] = [[0] for _ in ids]
            output["exclude_from_baseline"] = [True for _ in ids]
            output["rollout_logprobs"] = None
        else:
            output["exclude_from_baseline"] = [False for _ in ids]
            output["rollout_logprobs"] = [[-0.5] for _ in ids]
        return output

    ids = [
        TrajectoryID("trainable", 0),
        TrajectoryID("masked", 0),
        TrajectoryID("trainable", 1),
        TrajectoryID("masked", 1),
    ]
    dispatcher = _dispatcher([_Coordinator(run_group), _Coordinator(run_group)], harbor_runner_spec)

    result = await dispatcher.run(_request(ids))

    assert result["trajectory_ids"] == ids
    assert result["loss_masks"] == [[1], [0], [1], [0]]


@pytest.mark.asyncio
async def test_dispatcher_rejects_output_from_the_wrong_group(harbor_runner_spec):
    async def wrong_group(_input_batch, _global_step):
        return _output([TrajectoryID("other", 0)])

    dispatcher = _dispatcher([_Coordinator(wrong_group)], harbor_runner_spec)

    with pytest.raises(ValueError, match="identity mismatch"):
        await dispatcher.run(_request([TrajectoryID("a", 0)]))


@pytest.mark.asyncio
async def test_coordinator_rpc_returns_one_group_unchanged(harbor_runner_spec):
    expected = _output([TrajectoryID("a", 0)])

    async def completed_rpc(_input_batch, _global_step):
        return expected

    dispatcher = _dispatcher([_Coordinator(completed_rpc)], harbor_runner_spec)

    assert await dispatcher.run(_request([TrajectoryID("a", 0)])) is expected


@pytest.mark.asyncio
async def test_coordinator_rpc_timeout_cancels_remote_work(ray_init, harbor_runner_spec, monkeypatch):
    actor = _BlockingCoordinator.remote()
    dispatcher = _dispatcher([actor], harbor_runner_spec, timeout=0.1)
    cancelled_refs = []
    original_cancel = ray.cancel

    def capture_cancel(ref, *, force, recursive):
        cancelled_refs.append(ref)
        original_cancel(ref, force=force, recursive=recursive)

    monkeypatch.setattr(ray, "cancel", capture_cancel)

    run = asyncio.create_task(dispatcher.run(_request([TrajectoryID("a", 0)])))
    await actor.wait_for_start.remote()
    with pytest.raises(RolloutCoordinatorRPCTimeoutError):
        await run

    assert len(cancelled_refs) == 1
    with pytest.raises(ray.exceptions.TaskCancelledError):
        await asyncio.to_thread(ray.get, cancelled_refs[0], timeout=1)


@pytest.mark.asyncio
async def test_coordinator_rpc_preserves_remote_timeout_error(harbor_runner_spec):
    remote_error = TimeoutError("remote post-processing timed out")

    async def failed_rpc(_input_batch, _global_step):
        raise remote_error

    dispatcher = _dispatcher([_Coordinator(failed_rpc)], harbor_runner_spec)

    with pytest.raises(TimeoutError) as raised:
        await dispatcher.run(_request([TrajectoryID("a", 0)]))

    assert raised.value is remote_error
