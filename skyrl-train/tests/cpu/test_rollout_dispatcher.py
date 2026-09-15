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
from skyrl_train.trajectory_runners.types import BatchMetadata, TrainingPhase, TrajectoryID


class _RemoteMethod:
    def __init__(self, call: Callable[..., Awaitable[dict]]):
        self._call = call

    def remote(self, *args, **kwargs):
        return self._call(*args, **kwargs)


class _Coordinator:
    def __init__(self, call: Callable[..., Awaitable[dict]]):
        self.run_shard = _RemoteMethod(call)


class _SessionCoordinator(_Coordinator):
    def __init__(self, name: str, calls: list[tuple[str, str]]):
        self.eval_concurrency: list[int | None] = []

        async def run_shard(input_batch, _global_step):
            phase = input_batch["batch_metadata"].training_phase
            calls.append((name, phase))
            return _output(input_batch["trajectory_ids"])

        async def start_eval_session(*, n_concurrent_trials=None, **_kwargs):
            calls.append((name, "start_eval"))
            self.eval_concurrency.append(n_concurrent_trials)

        async def stop_eval_session():
            calls.append((name, "stop_eval"))

        super().__init__(run_shard)
        self.start_eval_session = _RemoteMethod(start_eval_session)
        self.stop_eval_session = _RemoteMethod(stop_eval_session)


@ray.remote
class _BlockingCoordinator:
    def __init__(self):
        self._started = asyncio.Event()

    async def run_shard(self, *_args):
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
        "actual_global_step": 7,
    }


def _dispatcher(
    actors: list[object],
    harbor_runner_spec: HarborRunnerSpec,
    *,
    timeout: float = 1,
    eval_spread: bool = False,
) -> RolloutDispatcher:
    dispatcher = RolloutDispatcher(
        spec=harbor_runner_spec,
        resources=ProcessPoolResources(
            num_coordinators=len(actors),
            cpus_per_coordinator=1,
            executor_workers=1,
            rpc_timeout_seconds=timeout,
            eval_spread_coordinators=eval_spread,
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
async def test_dispatcher_isolates_eval_from_concurrent_training(harbor_runner_spec):
    calls: list[tuple[str, str]] = []
    dispatcher = _dispatcher(
        [_SessionCoordinator("eval", calls), _SessionCoordinator("train", calls)],
        harbor_runner_spec,
    )

    await dispatcher.start_eval_session(run_name="run", eval_step=0)
    await dispatcher.run(_request([TrajectoryID("heldout", 0)], "eval"))
    await dispatcher.run(_request([TrajectoryID("training", 0)], "train"))
    await dispatcher.stop_eval_session()

    assert calls == [
        ("eval", "start_eval"),
        ("eval", "eval"),
        ("train", "train"),
        ("eval", "stop_eval"),
    ]


@pytest.mark.asyncio
async def test_dispatcher_preserves_global_eval_concurrency_when_training_is_sharded(harbor_runner_spec):
    calls: list[tuple[str, str]] = []
    eval_coordinator = _SessionCoordinator("eval", calls)
    harbor_runner_spec.terminal_bench_config.harbor = {"n_concurrent_trials": 32}
    dispatcher = _dispatcher(
        [eval_coordinator, _SessionCoordinator("train-1", calls), _SessionCoordinator("train-2", calls)],
        harbor_runner_spec,
    )

    await dispatcher.start_eval_session(run_name="run", eval_step=0)

    assert eval_coordinator.eval_concurrency == [32]


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
async def test_dispatcher_does_not_require_rollout_logprobs_during_eval(harbor_runner_spec):
    harbor_runner_spec.config.trainer.algorithm.use_tis = True
    calls: list[tuple[str, str]] = []
    ids = [TrajectoryID("a", 0), TrajectoryID("b", 0)]
    dispatcher = _dispatcher(
        [_SessionCoordinator("eval", calls), _SessionCoordinator("train", calls)],
        harbor_runner_spec,
    )

    await dispatcher.start_eval_session(run_name="run", eval_step=0)
    try:
        result = await dispatcher.run(_request(ids, "eval"))
    finally:
        await dispatcher.stop_eval_session()

    assert result["trajectory_ids"] == ids
    assert result["rollout_logprobs"] is None


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
async def test_coordinator_rpc_timeout_resets_on_same_actor_progress(harbor_runner_spec, monkeypatch):
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
        def remote(self, input_batch, _global_step):
            nonlocal slow_result
            instance_id = input_batch["trajectory_ids"][0].instance_id
            result = asyncio.get_running_loop().create_future()
            if instance_id == "a":
                slow_result = result
                return result

            result.set_result(_output(input_batch["trajectory_ids"]))
            asyncio.get_running_loop().call_soon(slow_task.cancel)
            return result

    class _TimedCoordinator:
        run_shard = _TimedRemoteMethod()

    ids = [TrajectoryID("a", 0), TrajectoryID("b", 0)]
    dispatcher = _dispatcher([_TimedCoordinator()], harbor_runner_spec, timeout=0.5)
    monkeypatch.setattr(asyncio, "timeout_at", lambda _deadline: _FakeDeadline())

    result = await dispatcher.run(_request(ids))

    assert result["trajectory_ids"] == ids


@pytest.mark.asyncio
async def test_coordinator_rpc_timeout_cancels_remote_work(ray_init, harbor_runner_spec, monkeypatch):
    actor = _BlockingCoordinator.remote()
    dispatcher = _dispatcher([actor], harbor_runner_spec, timeout=0.1)
    cancel_calls = []
    original_cancel = ray.cancel

    def capture_cancel(ref, *, force, recursive):
        cancel_calls.append((ref, force, recursive))
        original_cancel(ref, force=force, recursive=recursive)

    monkeypatch.setattr(ray, "cancel", capture_cancel)

    run = asyncio.create_task(dispatcher.run(_request([TrajectoryID("a", 0)])))
    await actor.wait_for_start.remote()
    with pytest.raises(RolloutCoordinatorRPCTimeoutError):
        await run

    assert len(cancel_calls) == 1
    cancelled_ref, force, recursive = cancel_calls[0]
    assert isinstance(cancelled_ref, ray.ObjectRef)
    assert force is False
    assert recursive is True


@pytest.mark.asyncio
async def test_coordinator_rpc_preserves_remote_timeout_error(harbor_runner_spec):
    remote_error = TimeoutError("remote post-processing timed out")

    async def failed_rpc(_input_batch, _global_step):
        raise remote_error

    dispatcher = _dispatcher([_Coordinator(failed_rpc)], harbor_runner_spec)

    with pytest.raises(TimeoutError) as raised:
        await dispatcher.run(_request([TrajectoryID("a", 0)]))

    assert raised.value is remote_error


class _KwargsRemoteMethod:
    """Remote-method stub that records calls and returns an awaitable."""

    def __init__(self, log: list, name: str):
        self._log = log
        self._name = name

    def remote(self, *args, **kwargs):
        self._log.append((self._name, args, kwargs))

        request = next(
            (a for a in list(args) + list(kwargs.values()) if isinstance(a, dict) and a.get("trajectory_ids")), None
        )

        async def _done():
            if request is not None:
                return _output(list(request["trajectory_ids"]))
            return {"response_ids": [[1]], "rollout_metrics": {}}

        return _done()


class _EvalCoordinator:
    def __init__(self):
        self.calls: list = []
        self.run_shard = _KwargsRemoteMethod(self.calls, "run_shard")
        self.start_eval_session = _KwargsRemoteMethod(self.calls, "start_eval_session")
        self.stop_eval_session = _KwargsRemoteMethod(self.calls, "stop_eval_session")


def _fanout_dispatcher(actors: list, harbor_runner_spec: HarborRunnerSpec) -> RolloutDispatcher:
    dispatcher = _dispatcher(actors, harbor_runner_spec, timeout=5)
    dispatcher._actors = actors
    return dispatcher


def test_dispatcher_advertises_concurrent_eval(harbor_runner_spec):
    """evaluate() may hand the dispatcher every eval chunk at once.

    Eval runs on shard 0 with the unscaled n_concurrent_trials (#514); issuing the chunks
    together is what keeps that concurrency saturated.
    """
    dispatcher = _fanout_dispatcher([_EvalCoordinator()], harbor_runner_spec)
    assert dispatcher.supports_concurrent_eval is True


class _GatedCoordinator:
    """Session-aware coordinator whose group RPCs can be held open or made to fail."""

    def __init__(
        self,
        name: str,
        log: list[tuple[str, str]],
        *,
        fail_start: BaseException | None = None,
        fail_stop: BaseException | None = None,
    ):
        self.eval_concurrency: list[int] = []
        self.group_sizes: list[int] = []
        self.gate = asyncio.Event()
        self.gate.set()
        self.fail_run: BaseException | None = None

        async def run_shard(input_batch, _global_step):
            log.append((name, input_batch["batch_metadata"].training_phase))
            self.group_sizes.append(len(input_batch["prompts"]))
            await self.gate.wait()
            if self.fail_run is not None:
                raise self.fail_run
            return _output(input_batch["trajectory_ids"])

        async def start_eval_session(*, n_concurrent_trials, **_kwargs):
            log.append((name, "start_eval"))
            if fail_start is not None:
                raise fail_start
            self.eval_concurrency.append(n_concurrent_trials)

        async def stop_eval_session():
            log.append((name, "stop_eval"))
            if fail_stop is not None:
                raise fail_stop

        self.run_shard = _RemoteMethod(run_shard)
        self.start_eval_session = _RemoteMethod(start_eval_session)
        self.stop_eval_session = _RemoteMethod(stop_eval_session)


async def _until(predicate: Callable[[], bool]) -> None:
    for _ in range(1000):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition was not reached")


def _group(instance_id: str, size: int, phase: TrainingPhase = "eval") -> dict:
    return _request([TrajectoryID(instance_id, repetition) for repetition in range(size)], phase)


def _session_calls(log: list[tuple[str, str]], call: str) -> list[str]:
    return sorted(name for name, entry in log if entry == call)


def test_process_pool_eval_spread_is_off_by_default_and_set_by_hydra_override():
    from hydra import compose, initialize_config_dir
    from skyrl_train.config.utils import CONFIG_DIR, DEFAULT_CONFIG_NAME

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        default = compose(config_name=DEFAULT_CONFIG_NAME)
        spread = compose(
            config_name=DEFAULT_CONFIG_NAME,
            overrides=["trajectory_runner.process_pool.eval_spread_coordinators=true"],
        )

    assert ProcessPoolResources.from_config(default).eval_spread_coordinators is False
    assert ProcessPoolResources.from_config(spread).eval_spread_coordinators is True


@pytest.mark.asyncio
async def test_eval_without_spread_stays_on_coordinator_zero(harbor_runner_spec):
    log: list[tuple[str, str]] = []
    coordinators = [_GatedCoordinator(f"c{index}", log) for index in range(3)]
    harbor_runner_spec.terminal_bench_config.harbor = {"n_concurrent_trials": 32}
    dispatcher = _dispatcher(coordinators, harbor_runner_spec)

    await dispatcher.start_eval_session(run_name="run", eval_step=0)
    eval_ids = [TrajectoryID(task, repetition) for task in ("a", "b", "c", "d") for repetition in range(2)]
    await dispatcher.run(_request(eval_ids, "eval"))
    for index in range(4):
        await dispatcher.run(_group(f"train-{index}", 1, "train"))
    await dispatcher.stop_eval_session()

    assert [coordinator.eval_concurrency for coordinator in coordinators] == [[32], [], []]
    assert [name for name, phase in log if phase == "eval"] == ["c0", "c0", "c0", "c0"]
    assert sorted({name for name, phase in log if phase == "train"}) == ["c1", "c2"]
    assert _session_calls(log, "start_eval") == ["c0"]
    assert _session_calls(log, "stop_eval") == ["c0"]
    assert dispatcher._eval_inflight_trials == [0, 0, 0]


@pytest.mark.asyncio
@pytest.mark.parametrize(("total", "caps"), [(32, [11, 11, 10]), (192, [64, 64, 64]), (2, [1, 1])])
async def test_spread_eval_session_splits_global_concurrency(harbor_runner_spec, total, caps):
    log: list[tuple[str, str]] = []
    coordinators = [_GatedCoordinator(f"c{index}", log) for index in range(3)]
    harbor_runner_spec.terminal_bench_config.harbor = {"n_concurrent_trials": total}
    dispatcher = _dispatcher(coordinators, harbor_runner_spec, eval_spread=True)

    await dispatcher.start_eval_session(run_name="run", eval_step=0)
    started = [coordinator.eval_concurrency for coordinator in coordinators]
    await dispatcher.stop_eval_session()

    assert started == [[cap] for cap in caps] + [[] for _ in range(3 - len(caps))]
    assert sum(cap for caps_started in started for cap in caps_started) == total
    assert _session_calls(log, "stop_eval") == [f"c{index}" for index in range(len(caps))]


@pytest.mark.asyncio
async def test_spread_eval_session_honours_explicit_concurrency(harbor_runner_spec):
    coordinators = [_GatedCoordinator(f"c{index}", []) for index in range(4)]
    dispatcher = _dispatcher(coordinators, harbor_runner_spec, eval_spread=True)

    await dispatcher.start_eval_session(run_name="run", eval_step=0, n_concurrent_trials=10)

    assert [coordinator.eval_concurrency for coordinator in coordinators] == [[3], [3], [2], [2]]


@pytest.mark.asyncio
async def test_spread_eval_routes_groups_to_fewest_in_flight_trials(harbor_runner_spec):
    log: list[tuple[str, str]] = []
    coordinators = [_GatedCoordinator(f"c{index}", log) for index in range(2)]
    for coordinator in coordinators:
        coordinator.gate.clear()
    dispatcher = _dispatcher(coordinators, harbor_runner_spec, eval_spread=True)
    await dispatcher.start_eval_session(run_name="run", eval_step=0)

    runs = []
    # Balancing by RPC count would send "c" to c0; by trial count c1 keeps taking groups until it
    # carries as many trials as c0, and the 4-4 tie then continues the round-robin at c0.
    for routed, (instance_id, size) in enumerate([("a", 4), ("b", 1), ("c", 1), ("d", 2), ("e", 1)], start=1):
        runs.append(asyncio.create_task(dispatcher.run(_group(instance_id, size))))
        await _until(lambda routed=routed: sum(phase == "eval" for _name, phase in log) == routed)

    assert coordinators[0].group_sizes == [4, 1]
    assert coordinators[1].group_sizes == [1, 1, 2]
    assert dispatcher._eval_inflight_trials == [5, 4]

    for coordinator in coordinators:
        coordinator.gate.set()
    await asyncio.gather(*runs)
    await dispatcher.stop_eval_session()

    assert dispatcher._eval_inflight_trials == [0, 0]
    assert dispatcher._actor_pending_rpcs == [0, 0]


@pytest.mark.asyncio
async def test_spread_eval_releases_trial_counts_on_error_and_cancellation(harbor_runner_spec):
    log: list[tuple[str, str]] = []
    coordinators = [_GatedCoordinator(f"c{index}", log) for index in range(2)]
    coordinators[0].fail_run = RuntimeError("trial group failed")
    dispatcher = _dispatcher(coordinators, harbor_runner_spec, eval_spread=True)
    await dispatcher.start_eval_session(run_name="run", eval_step=0)

    with pytest.raises(RuntimeError, match="trial group failed"):
        await dispatcher.run(_group("failed", 3))
    assert coordinators[0].group_sizes == [3]
    assert dispatcher._eval_inflight_trials == [0, 0]

    coordinators[1].gate.clear()
    cancelled = asyncio.create_task(dispatcher.run(_group("cancelled", 2)))
    await _until(lambda: coordinators[1].group_sizes == [2])
    assert dispatcher._eval_inflight_trials == [0, 2]
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    coordinators[1].gate.set()
    await asyncio.sleep(0)

    assert dispatcher._eval_inflight_trials == [0, 0]
    assert dispatcher._actor_pending_rpcs == [0, 0]
    await dispatcher.stop_eval_session()


@pytest.mark.asyncio
async def test_spread_eval_stop_clears_state_when_one_coordinator_stop_fails(harbor_runner_spec):
    log: list[tuple[str, str]] = []
    coordinators = [
        _GatedCoordinator("c0", log),
        _GatedCoordinator("c1", log, fail_stop=RuntimeError("stop failed")),
        _GatedCoordinator("c2", log),
    ]
    dispatcher = _dispatcher(coordinators, harbor_runner_spec, eval_spread=True)
    await dispatcher.start_eval_session(run_name="run", eval_step=0)

    with pytest.raises(RuntimeError, match="stop failed"):
        await dispatcher.stop_eval_session()

    assert _session_calls(log, "stop_eval") == ["c0", "c1", "c2"]
    assert dispatcher._eval_session_active is False
    assert dispatcher._eval_coordinators == ()
    await asyncio.wait_for(dispatcher.run(_group("train", 1, "train")), timeout=1)
    await dispatcher.start_eval_session(run_name="run", eval_step=1)
    assert dispatcher._eval_session_active is True


@pytest.mark.asyncio
async def test_spread_eval_start_failure_stops_the_sessions_that_started(harbor_runner_spec):
    log: list[tuple[str, str]] = []
    coordinators = [
        _GatedCoordinator("c0", log),
        _GatedCoordinator("c1", log),
        _GatedCoordinator("c2", log, fail_start=RuntimeError("start failed")),
    ]
    dispatcher = _dispatcher(coordinators, harbor_runner_spec, eval_spread=True)

    with pytest.raises(RuntimeError, match="start failed"):
        await dispatcher.start_eval_session(run_name="run", eval_step=0)

    assert _session_calls(log, "stop_eval") == ["c0", "c1"]
    assert dispatcher._eval_session_active is False
    assert dispatcher._eval_coordinators == ()
    await asyncio.wait_for(dispatcher.run(_group("train", 1, "train")), timeout=1)


@pytest.mark.asyncio
async def test_spread_eval_holds_training_until_the_session_stops(harbor_runner_spec):
    log: list[tuple[str, str]] = []
    coordinators = [_GatedCoordinator(f"c{index}", log) for index in range(2)]
    dispatcher = _dispatcher(coordinators, harbor_runner_spec, eval_spread=True)
    await dispatcher.start_eval_session(run_name="run", eval_step=0)

    training = asyncio.create_task(dispatcher.run(_group("train", 1, "train")))
    for _ in range(50):
        await asyncio.sleep(0)
    assert [entry for entry in log if entry[1] == "train"] == []

    await dispatcher.stop_eval_session()
    await asyncio.wait_for(training, timeout=1)

    phases = [entry for _name, entry in log]
    assert phases.index("train") > max(index for index, entry in enumerate(phases) if entry == "stop_eval")


@pytest.mark.asyncio
async def test_spread_eval_start_drains_training_on_every_coordinator(harbor_runner_spec):
    log: list[tuple[str, str]] = []
    coordinators = [_GatedCoordinator(f"c{index}", log) for index in range(2)]
    for coordinator in coordinators:
        coordinator.gate.clear()
    dispatcher = _dispatcher(coordinators, harbor_runner_spec, eval_spread=True)
    training = [asyncio.create_task(dispatcher.run(_group(f"train-{index}", 1, "train"))) for index in range(2)]
    await _until(lambda: len(log) == 2)
    assert sorted(name for name, _phase in log) == ["c0", "c1"]

    starting = asyncio.create_task(dispatcher.start_eval_session(run_name="run", eval_step=0))
    coordinators[0].gate.set()
    await _until(lambda: training[0].done() or training[1].done())
    for _ in range(50):
        await asyncio.sleep(0)
    assert _session_calls(log, "start_eval") == []

    coordinators[1].gate.set()
    await asyncio.wait_for(starting, timeout=1)
    await asyncio.gather(*training)
    assert _session_calls(log, "start_eval") == ["c0", "c1"]


@pytest.mark.asyncio
async def test_spread_eval_below_coordinator_count_leaves_training_a_coordinator(harbor_runner_spec):
    log: list[tuple[str, str]] = []
    coordinators = [_GatedCoordinator(f"c{index}", log) for index in range(3)]
    harbor_runner_spec.terminal_bench_config.harbor = {"n_concurrent_trials": 2}
    dispatcher = _dispatcher(coordinators, harbor_runner_spec, eval_spread=True)
    await dispatcher.start_eval_session(run_name="run", eval_step=0)

    for index in range(3):
        await asyncio.wait_for(dispatcher.run(_group(f"train-{index}", 1, "train")), timeout=1)
    await dispatcher.stop_eval_session()

    assert [name for name, phase in log if phase == "train"] == ["c2", "c2", "c2"]
