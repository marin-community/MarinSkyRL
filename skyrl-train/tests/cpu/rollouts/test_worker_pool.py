import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import pytest
import ray
from omegaconf import DictConfig, OmegaConf

from skyrl_train.rollouts.buffer import RolloutLease, RolloutTask
from skyrl_train.rollouts.workers import (
    RolloutWorkerPool,
    RolloutWorkerResources,
    RolloutWorkerStalledError,
    WorkerShard,
)
from skyrl_train.trajectory_runners.types import BatchMetadata, TrainingPhase, TrajectoryID


@dataclass(frozen=True)
class _Spec:
    config: DictConfig

    def build(self, tokenizer, shard: WorkerShard):
        raise AssertionError("these tests supply already-started workers")


@pytest.fixture
def spec() -> _Spec:
    return _Spec(OmegaConf.create({}))


class _RemoteMethod:
    def __init__(self, call: Callable[..., Awaitable[dict]]):
        self._call = call

    def remote(self, *args, **kwargs):
        return self._call(*args, **kwargs)


class _Worker:
    def __init__(self, call: Callable[..., Awaitable[dict]]):
        self.run = _RemoteMethod(call)


class _SessionWorker(_Worker):
    def __init__(self, name: str, calls: list[tuple[str, str]]):
        async def run(input_batch):
            phase = input_batch["batch_metadata"].training_phase
            calls.append((name, phase))
            return _output(input_batch["trajectory_ids"])

        async def run_task(task, _writer):
            calls.append((name, f"task {task.prompt['uid']}"))

        async def start_eval_session(**_kwargs):
            calls.append((name, "start_eval"))

        async def stop_eval_session():
            calls.append((name, "stop_eval"))

        super().__init__(run)
        self.run_task = _RemoteMethod(run_task)
        self.start_eval_session = _RemoteMethod(start_eval_session)
        self.stop_eval_session = _RemoteMethod(stop_eval_session)


@ray.remote
class _BlockingWorker:
    def __init__(self):
        self._started = asyncio.Event()

    async def run(self, *_args):
        self._started.set()
        await asyncio.Event().wait()

    async def wait_for_start(self):
        await self._started.wait()


def _request(ids: list[TrajectoryID], phase: TrainingPhase | None = None) -> dict:
    return {
        "prompts": [f"prompt-{trajectory_id.to_string()}" for trajectory_id in ids],
        "env_classes": ["terminal" for _ in ids],
        "env_extras": [{} for _ in ids],
        "sampling_params": {},
        "trajectory_ids": ids,
        "batch_metadata": BatchMetadata(global_step=1, training_phase=phase) if phase is not None else None,
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
    }


def _pool(workers: list[object], spec: _Spec, *, timeout: float = 1) -> RolloutWorkerPool:
    pool = RolloutWorkerPool(
        spec,
        RolloutWorkerResources(
            num_workers=len(workers), cpus_per_worker=1, executor_threads=1, progress_timeout_seconds=timeout
        ),
    )
    pool._actors = list(workers)
    return pool


@pytest.mark.asyncio
async def test_training_tasks_avoid_the_eval_worker(spec):
    calls: list[tuple[str, str]] = []
    pool = _pool(
        [_SessionWorker("eval", calls), _SessionWorker("train", calls)],
        spec,
    )
    task = RolloutTask(RolloutLease("lease", 1), {"uid": "a"}, _request([TrajectoryID("a", 0)], "train"))

    await pool.start_eval_session(run_name="run", eval_step=0)
    await pool.run_task(task, writer=None)
    await pool.stop_eval_session()

    assert calls == [("eval", "start_eval"), ("train", "task a"), ("eval", "stop_eval")]


@pytest.mark.asyncio
async def test_pool_isolates_eval_from_concurrent_training(spec):
    calls: list[tuple[str, str]] = []
    pool = _pool(
        [_SessionWorker("eval", calls), _SessionWorker("train", calls)],
        spec,
    )

    await pool.start_eval_session(run_name="run", eval_step=0)
    await pool.run(_request([TrajectoryID("heldout", 0)], "eval"))
    await pool.run(_request([TrajectoryID("training", 0)], "train"))
    await pool.stop_eval_session()

    assert calls == [
        ("eval", "start_eval"),
        ("eval", "eval"),
        ("train", "train"),
        ("eval", "stop_eval"),
    ]


@pytest.mark.asyncio
async def test_a_single_worker_defers_training_until_the_eval_session_ends(spec):
    calls: list[tuple[str, str]] = []
    pool = _pool([_SessionWorker("only", calls)], spec)
    task = RolloutTask(RolloutLease("lease", 1), {"uid": "a"}, _request([TrajectoryID("a", 0)], "train"))

    await pool.start_eval_session(run_name="run", eval_step=0)
    training = asyncio.create_task(pool.run_task(task, writer=None))
    await pool.run(_request([TrajectoryID("heldout", 0)], "eval"))
    await pool.stop_eval_session()
    await training

    assert calls == [("only", "start_eval"), ("only", "eval"), ("only", "stop_eval"), ("only", "task a")]


@pytest.mark.asyncio
async def test_pool_returns_one_group_unchanged(spec):
    expected = _output([TrajectoryID("a", 0)])

    async def completed_request(_input_batch):
        return expected

    pool = _pool([_Worker(completed_request)], spec)

    assert await pool.run(_request([TrajectoryID("a", 0)])) is expected


@pytest.mark.asyncio
async def test_progress_deadline_resets_when_the_same_worker_completes_a_request(spec, monkeypatch):
    slow_result = None
    slow_task = None

    class _FakeDeadline:
        def __init__(self):
            self._expired = False

        async def __aenter__(self):
            nonlocal slow_task
            current_task = asyncio.current_task()
            if slow_task is None:
                slow_task = current_task
            elif current_task is slow_task and not slow_result.done():
                slow_result.set_result(_output([TrajectoryID("a", 0)]))
            return self

        async def __aexit__(self, exception_type, _exception, _traceback):
            if exception_type is asyncio.CancelledError:
                self._expired = True
                raise TimeoutError
            return False

        def expired(self):
            return self._expired

    class _TimedRemoteMethod:
        def remote(self, input_batch):
            nonlocal slow_result
            instance_id = input_batch["trajectory_ids"][0].instance_id
            result = asyncio.get_running_loop().create_future()
            if instance_id == "a":
                slow_result = result
                return result

            result.set_result(_output(input_batch["trajectory_ids"]))
            asyncio.get_running_loop().call_soon(slow_task.cancel)
            return result

    class _TimedWorker:
        run = _TimedRemoteMethod()

    pool = _pool([_TimedWorker()], spec, timeout=0.5)
    monkeypatch.setattr(asyncio, "timeout_at", lambda _deadline: _FakeDeadline())

    slow, fast = await asyncio.gather(
        pool.run(_request([TrajectoryID("a", 0)])), pool.run(_request([TrajectoryID("b", 0)]))
    )

    assert slow["trajectory_ids"] == [TrajectoryID("a", 0)]
    assert fast["trajectory_ids"] == [TrajectoryID("b", 0)]


@pytest.mark.asyncio
async def test_stalled_worker_request_is_cancelled_remotely(ray_init, spec, monkeypatch):
    actor = _BlockingWorker.remote()
    pool = _pool([actor], spec, timeout=0.1)
    cancel_calls = []
    original_cancel = ray.cancel

    def capture_cancel(ref, *, force, recursive):
        cancel_calls.append((ref, force, recursive))
        original_cancel(ref, force=force, recursive=recursive)

    monkeypatch.setattr(ray, "cancel", capture_cancel)

    run = asyncio.create_task(pool.run(_request([TrajectoryID("a", 0)])))
    await actor.wait_for_start.remote()
    with pytest.raises(RolloutWorkerStalledError):
        await run

    assert len(cancel_calls) == 1
    cancelled_ref, force, recursive = cancel_calls[0]
    assert isinstance(cancelled_ref, ray.ObjectRef)
    assert force is False
    assert recursive is True


@pytest.mark.asyncio
async def test_pool_preserves_a_remote_timeout_error(spec):
    remote_error = TimeoutError("remote post-processing timed out")

    async def failed_request(_input_batch):
        raise remote_error

    pool = _pool([_Worker(failed_request)], spec)

    with pytest.raises(TimeoutError) as raised:
        await pool.run(_request([TrajectoryID("a", 0)]))

    assert raised.value is remote_error
