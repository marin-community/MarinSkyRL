from __future__ import annotations

import asyncio
import collections
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from skyrl_train import rollout_observability, telemetry as training_telemetry
from skyrl_train.async_rollout_state import GeneratedOutputGroup
from skyrl_train.dynamic_sampling import GroupSelectionResult
from skyrl_train.fully_async_trainer import (
    FullyAsyncRayPPOTrainer,
    GenerationStalledError,
    _AsyncStalenessManager,
    _GenerationQueues,
    _GroupFreshness,
)
from skyrl_train.group_admission import AdmissionDecision
from skyrl_train.trainer import ResumeMode


class RecordingInstrument:
    def __init__(self) -> None:
        self.records: list[tuple[float, dict[str, str]]] = []

    def add(self, value: float, *, attributes: dict[str, str]) -> None:
        self.records.append((value, attributes))

    def record(self, value: float, *, attributes: dict[str, str]) -> None:
        self.records.append((value, attributes))

    def set(self, value: float, *, attributes: dict[str, str]) -> None:
        self.records.append((value, attributes))


class PolicyModel:
    def __init__(self, failure: Exception | None = None) -> None:
        self.failure = failure

    async def async_run_method(self, *_args):
        if self.failure is not None:
            raise self.failure
        return "published"


class Control:
    should_evaluate = False
    should_training_stop = False

    def reset(self) -> None:
        pass


class CallbackHandler:
    async def call_event_async(self, _name, _state, control, **_kwargs):
        return control


class BufferCallback:
    def bind_queues(self, _queues) -> None:
        pass


def trainer_shell(*, global_step: int = 0, published_policy_version: int = 0) -> FullyAsyncRayPPOTrainer:
    trainer = object.__new__(FullyAsyncRayPPOTrainer)
    trainer.global_step = global_step
    trainer._published_policy_version = published_policy_version
    trainer.all_timings = {}
    trainer.all_startup_timings = {}
    trainer.admission_stall_timeout = 10
    trainer.policy_model = PolicyModel()
    trainer.inference_engine_client = object()

    async def drain() -> None:
        pass

    trainer._drain_policy_event_loops = drain
    return trainer


@pytest.mark.asyncio
@pytest.mark.parametrize("drained", [False, True])
async def test_shutdown_reports_actual_completed_buffer_depth(monkeypatch, drained):
    depth, capacity = RecordingInstrument(), RecordingInstrument()
    monkeypatch.setattr(training_telemetry, "rollout_queue_depth", depth)
    monkeypatch.setattr(training_telemetry, "rollout_capacity", capacity)
    trainer = trainer_shell()
    queue = asyncio.Queue(maxsize=4)
    queue.put_nowait(object())
    trainer._generation_queues = _GenerationQueues(queue, asyncio.Queue(), asyncio.Condition())
    training_telemetry.record_rollout_buffer(1, 4)
    if drained:
        queue.get_nowait()

    async def release_resources():
        pass

    trainer._teardown = release_resources
    await trainer.shutdown()

    assert depth.records[-1][0] == (0 if drained else 1)
    assert capacity.records[-1][0] == 4


@pytest.mark.asyncio
async def test_successful_weight_publication_installs_one_based_sampling_callback() -> None:
    trainer = trainer_shell(global_step=3, published_policy_version=2)
    trainer.resume_mode = ResumeMode.NONE
    trainer.total_training_steps = 4
    trainer.num_steps_per_epoch = 1
    trainer.init_weight_sync_state = lambda: None
    trainer._log_weight_update_completed = lambda **_kwargs: None
    trainer._log_startup_timings = lambda: None
    trainer._create_trainer_state = lambda **_kwargs: object()
    trainer._control = Control()
    trainer.callback_handler = CallbackHandler()
    trainer.eval_dataset = None
    trainer.cfg = OmegaConf.create({"trainer": {"epochs": 4}})
    trainer.max_buffered_groups = 1
    trainer.num_parallel_generation_workers = 0
    trainer.mini_batch_size = 1
    trainer._buffer_checkpoint_callback = BufferCallback()
    trainer._pending_buffer_restore_path = None
    trainer.trajectory_runner = SimpleNamespace(global_step_fn=None)

    with pytest.raises(GenerationStalledError, match="admitted=0/1"):
        await trainer._train_loop()

    assert trainer._published_policy_version == 3
    assert trainer.global_step == 4
    assert trainer.trajectory_runner.global_step_fn() == 4


@pytest.mark.asyncio
async def test_failed_weight_publication_keeps_last_successful_version() -> None:
    failure = RuntimeError("broadcast failed")
    trainer = trainer_shell(global_step=6, published_policy_version=5)
    trainer.policy_model = PolicyModel(failure)

    with pytest.raises(RuntimeError, match="broadcast failed"):
        await trainer.async_sync_policy_weights_to_inference_engines()

    assert trainer._published_policy_version == 5


@pytest.mark.asyncio
async def test_train_waits_for_producer_cancellation_before_shutdown_snapshot() -> None:
    trainer = trainer_shell()
    trainer._active_trajectory_tasks = []
    lifecycle: list[str] = []
    producer_started = asyncio.Event()
    cancellation_started = asyncio.Event()
    finish_cancellation = asyncio.Event()

    async def producer() -> None:
        producer_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellation_started.set()
            await finish_cancellation.wait()
            lifecycle.append("producer_cancelled")
            raise

    async def startup() -> None:
        pass

    async def train_loop() -> None:
        trainer._active_trajectory_tasks = [asyncio.create_task(producer())]
        await producer_started.wait()
        raise RuntimeError("training failed")

    async def shutdown() -> None:
        assert all(task.done() for task in trainer._active_trajectory_tasks)
        lifecycle.append("shutdown_snapshot")

    trainer._startup_trajectory_runner = startup
    trainer._train_loop = train_loop
    trainer.shutdown = shutdown

    training = asyncio.create_task(trainer.train())
    await cancellation_started.wait()
    assert lifecycle == []
    finish_cancellation.set()
    with pytest.raises(RuntimeError, match="training failed"):
        await training

    assert lifecycle == ["producer_cancelled", "shutdown_snapshot"]


@pytest.mark.asyncio
async def test_producer_failure_reaches_waiting_consumer() -> None:
    class ProducerFailure(RuntimeError):
        pass

    class FailingDataloader:
        async def get_next_non_consumed_data(self):
            raise ProducerFailure("generation input failed")

    trainer = trainer_shell()
    trainer.async_train_dataloader = FailingDataloader()
    queues = _GenerationQueues(
        completed=asyncio.Queue(),
        retries=asyncio.Queue(),
        condition=asyncio.Condition(),
        active_producers=1,
    )

    with pytest.raises(ProducerFailure, match="generation input failed"):
        await trainer._run_generate_for_a_group_loop(queues)

    assert queues.active_producers == 0
    with pytest.raises(GenerationStalledError, match="rollout producer failed") as exc_info:
        await trainer._get_admitted_generation_group_mini_batch(queues)
    assert isinstance(exc_info.value.__cause__, ProducerFailure)


@pytest.mark.asyncio
async def test_cancellation_after_enqueue_preserves_accepted_rollout_accounting() -> None:
    """A queued group remains accepted if cancellation lands during bookkeeping."""
    trainer = trainer_shell(global_step=1)
    trainer._async_observations_enabled = False
    trainer.cfg = OmegaConf.create(
        {
            "generator": {
                "n_samples_per_prompt": 1,
                "backend": "vllm",
                "sampling_params": {
                    "max_generate_length": 8,
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "top_k": -1,
                    "min_p": 0.0,
                    "logprobs": None,
                },
            },
            "environment": {"env_class": "unused"},
        }
    )

    class Dataloader:
        async def get_next_non_consumed_data(self):
            return [{"uid": "queued", "prompt": [1], "env_class": None, "env_extras": {}}]

    class Runner:
        async def run(self, _request, *, disable_tqdm):
            assert disable_tqdm
            return {"response_ids": [[2, 3]], "is_last_step": [True]}

    trainer.async_train_dataloader = Dataloader()
    trainer.trajectory_runner = Runner()
    trainer._classify_and_route_group = lambda _queues, _group: _GroupFreshness.FRESH
    manager = _AsyncStalenessManager(max_concurrent_generation_groups=1, mini_batch_size=1, max_staleness_steps=0)
    trainer._staleness_manager = manager
    acceptance_started = asyncio.Event()
    finish_acceptance = asyncio.Event()
    original_accept = manager.on_rollout_accepted

    async def paused_acceptance() -> None:
        acceptance_started.set()
        await finish_acceptance.wait()
        await original_accept()

    manager.on_rollout_accepted = paused_acceptance
    queues = _GenerationQueues(
        completed=asyncio.Queue(maxsize=1),
        retries=asyncio.Queue(),
        condition=asyncio.Condition(),
        active_producers=1,
    )
    producer = asyncio.create_task(trainer._run_generate_for_a_group_loop(queues))
    await acceptance_started.wait()
    assert queues.completed.qsize() == 1

    producer.cancel()
    done, _ = await asyncio.wait({producer}, timeout=0)
    assert producer not in done
    finish_acceptance.set()
    await producer

    assert manager._stat.running == 0
    assert manager._stat.submitted == 1
    assert manager._stat.accepted == 1


@pytest.mark.asyncio
async def test_group_terminal_outcomes_and_dwell_publish_once_across_rescan(monkeypatch) -> None:
    counts = RecordingInstrument()
    tokens = RecordingInstrument()
    dwell = RecordingInstrument()
    events = []
    monkeypatch.setattr(rollout_observability, "group_count", counts)
    monkeypatch.setattr(rollout_observability, "group_tokens", tokens)
    monkeypatch.setattr(rollout_observability, "buffer_dwell", dwell)
    monkeypatch.setattr(rollout_observability.time, "perf_counter", lambda: 14.0)
    monkeypatch.setattr(
        rollout_observability.telemetry,
        "event",
        lambda name, fields, *, attributes: events.append((name, fields, attributes)),
    )
    trainer = trainer_shell(global_step=8)
    trainer._async_observations_enabled = True
    trainer.mini_batch_size = 2
    trainer.all_metrics = {}
    trainer._groups_rejected_since_step = 0
    trainer._rejection_reasons_since_step = collections.Counter()
    trainer._groups_inspected_since_step = 0
    trainer._dynamic_sampling_type = None
    trainer._dynamic_sampling_max_candidate_groups = None
    trainer._group_admission_policy = SimpleNamespace(evaluate=lambda _group, global_step: AdmissionDecision())
    trainer._group_selection_policy = SimpleNamespace(evaluate=lambda _group: GroupSelectionResult.KEEP)
    trainer.data_tracker = SimpleNamespace(get_consumed_uids_in_epoch=lambda: set())

    class StalenessManager:
        async def on_rollouts_discarded(self, _count):
            pass

    trainer._staleness_manager = StalenessManager()
    queues = _GenerationQueues(
        completed=asyncio.Queue(),
        retries=asyncio.Queue(),
        condition=asyncio.Condition(),
        active_producers=1,
    )
    first = GeneratedOutputGroup(
        trajectory_batch={"response_ids": [[1, 2], [3]]},
        uid="group",
        earliest_model_step=7,
        source_prompts=[{"uid": "group"}],
        completed_at=10.0,
        telemetry_attempt_id="first-attempt",
    )
    duplicate = GeneratedOutputGroup(
        trajectory_batch={"response_ids": [[4]]},
        uid="group",
        earliest_model_step=7,
        source_prompts=[{"uid": "group"}],
        completed_at=11.0,
        telemetry_attempt_id="duplicate-attempt",
    )
    second = GeneratedOutputGroup(
        trajectory_batch={"response_ids": [[5, 6]]},
        uid="second",
        earliest_model_step=7,
        source_prompts=[{"uid": "second"}],
        completed_at=12.0,
        telemetry_attempt_id="second-attempt",
    )
    queues.completed.put_nowait(first)
    pending_batch = asyncio.create_task(trainer._get_admitted_generation_group_mini_batch(queues))
    done, _ = await asyncio.wait({pending_batch}, timeout=0)
    assert pending_batch not in done

    async with queues.condition:
        queues.completed.put_nowait(duplicate)
        queues.completed.put_nowait(second)
        queues.condition.notify_all()
    batch = await pending_batch
    assert batch == [first, second]

    first.admitted_at = 13.5
    trainer._record_group_terminal(first, "consumed")
    trainer._record_group_terminal(first, "consumed")

    duplicate_attributes = {"role": "trainer", "step": "8", "outcome": "duplicate"}
    consumed_attributes = {"role": "trainer", "step": "8", "outcome": "consumed"}
    assert counts.records == [(1, duplicate_attributes), (1, consumed_attributes)]
    assert tokens.records == [(1, duplicate_attributes), (3, consumed_attributes)]
    assert dwell.records == [(3.0, duplicate_attributes), (3.5, consumed_attributes)]
    assert [(name, body.fields, attributes) for name, body, attributes in events] == [
        ("rollout_group_outcome", {"call_id": "duplicate-attempt", "tokens": 1}, duplicate_attributes),
        ("rollout_group_outcome", {"call_id": "first-attempt", "tokens": 3}, consumed_attributes),
    ]
