import asyncio
import collections
from types import SimpleNamespace

import pytest
import torch
from torchdata.stateful_dataloader import StatefulDataLoader

from skyrl_train.fully_async_trainer import (
    FullyAsyncRayPPOTrainer,
    GeneratedOutputGroup,
    _AsyncDataloader,
    _AsyncStalenessManager,
    _GenerationQueues,
)
from skyrl_train.dynamic_sampling import DynamicSamplingType, GroupSelectionPolicy, resolve_dynamic_sampling_criteria
from skyrl_train.group_admission import (
    GroupAdmissionPolicy,
    GroupAdmissionStalledError,
    GroupAdvantageInvariant,
    TrainingGroupInvariantError,
)
from skyrl_train.trajectory_selection import BestOfNTrajectorySelector
from skyrl_train.trajectory_runners.base import TrajectoryID
from skyrl_train.utils.data_tracker import DataConsumptionTracker


@pytest.mark.parametrize("sync_phase", ["initial", "training_step"])
@pytest.mark.parametrize("offload_enabled", [False, True])
@pytest.mark.parametrize("first_token_admission", [False, True])
def test_async_weight_sync_respects_optimizer_offload_policy(sync_phase, offload_enabled, first_token_admission):
    trainer = object.__new__(FullyAsyncRayPPOTrainer)
    trainer.cfg = SimpleNamespace(trainer=SimpleNamespace(offload_optimizer_during_rollouts=offload_enabled))
    trainer.colocate_all = False
    trainer.all_startup_timings = {}
    trainer.all_timings = {}
    trainer.first_token_admission = first_token_admission
    trainer.global_step = 0 if sync_phase == "initial" else 3
    events = []

    class Policy:
        optimizer_on_gpu = True

        def offload_to_cpu(self, *, offload_optimizer, offload_model):
            assert offload_optimizer and not offload_model
            self.optimizer_on_gpu = False
            events.append("offload")

    class Engine:
        async def pause_generation(self):
            events.append("pause")

        async def resume_generation(self, policy_version=None):
            assert trainer.policy_model.optimizer_on_gpu != offload_enabled
            events.append(("resume", policy_version))

    async def sync_weights():
        events.append("sync")

    async def drain():
        events.append("drain")

    trainer.policy_model = Policy()
    trainer.inference_engine_client = Engine()
    trainer.async_sync_policy_weights_to_inference_engines = sync_weights
    trainer._drain_policy_event_loops = drain

    asyncio.run(trainer._sync_policy_weights_and_offload_optimizer(sync_phase=sync_phase))

    assert trainer.policy_model.optimizer_on_gpu != offload_enabled
    # Every sync, the initial one included, pauses before the copy and names the installed
    # version on resume only when first-token admission is on.
    assert events == ["pause"] + (["offload"] if offload_enabled else []) + [
        "sync",
        "drain",
        ("resume", trainer.global_step if first_token_admission else None),
    ]


def _trainer_at_step(global_step: int, *, first_token_admission: bool) -> FullyAsyncRayPPOTrainer:
    trainer = object.__new__(FullyAsyncRayPPOTrainer)
    trainer.global_step = global_step
    trainer.first_token_admission = first_token_admission
    return trainer


def _spans(*versions):
    return [[{"start": 0, "token_count": 1, "policy_version": version}] for version in versions]


def _completed_batch(rows, *, captured_step=4, response_ids=([1], [2])):
    batch = {"response_ids": list(response_ids), "actual_global_step": captured_step}
    if rows is not None:
        batch["behavior_policy_version_segments"] = rows
    return batch


def test_admission_step_is_the_captured_step_unless_first_token_admission_is_on():
    # Same completed group, same trainer state; only the flag differs. Off is the stamp the
    # runner captured; on is the oldest version that sampled the group plus one, because
    # version 1 is the policy after the first update and step 2 is the update that follows it.
    batch = _completed_batch(_spans(1, 3), captured_step=4)
    assert _trainer_at_step(4, first_token_admission=False)._admission_step(batch, fallback_step=4) == 4
    assert _trainer_at_step(4, first_token_admission=True)._admission_step(batch, fallback_step=4) == 2


def test_admission_step_falls_back_to_the_submission_step_when_nothing_was_captured():
    batch = _completed_batch(None, captured_step=None)
    assert _trainer_at_step(4, first_token_admission=False)._admission_step(batch, fallback_step=3) == 3


def test_a_response_continued_under_newer_weights_counts_from_its_oldest_span():
    rows = [[{"start": 0, "token_count": 1, "policy_version": 1}, {"start": 1, "token_count": 2, "policy_version": 3}]]
    batch = _completed_batch(rows, response_ids=([10, 11, 12],))
    assert _trainer_at_step(4, first_token_admission=True)._admission_step(batch, fallback_step=4) == 2


@pytest.mark.parametrize("rows", [None, _spans(None, 3)])
def test_first_token_admission_fails_loudly_when_a_sampled_group_carries_no_version(rows):
    trainer = _trainer_at_step(4, first_token_admission=True)
    with pytest.raises(RuntimeError, match="first_token_admission"):
        trainer._admission_step(_completed_batch(rows), fallback_step=4)


def test_first_token_admission_keeps_the_captured_step_for_a_group_that_sampled_nothing():
    trainer = _trainer_at_step(4, first_token_admission=True)
    batch = _completed_batch([[], []], captured_step=4, response_ids=([], []))
    assert trainer._admission_step(batch, fallback_step=3) == 4


def test_admission_step_rejects_a_version_newer_than_the_installed_policy():
    trainer = _trainer_at_step(2, first_token_admission=True)
    with pytest.raises(RuntimeError, match="newer than the installed policy"):
        trainer._admission_step(_completed_batch(_spans(3, 1)), fallback_step=1)


def _generated_group(
    uid: str,
    earliest_model_step: int,
    *,
    fully_masked: bool = False,
    rewards: list[float] | list[list[float]] | None = None,
    unshaped_rewards: list[float] | None = None,
) -> GeneratedOutputGroup:
    rewards = rewards or [0.0, 1.0]
    unshaped_rewards = unshaped_rewards or [0.0, 1.0]
    trajectory_batch = {
        "prompt_token_ids": [[1], [1]],
        "response_ids": [[2], [3]],
        "rewards": rewards,
        "unshaped_rewards": unshaped_rewards,
        "loss_masks": [[0], [0]] if fully_masked else [[1], [1]],
        "stop_reasons": ["stop", "stop"],
        "rollout_metrics": {},
        "rollout_logprobs": None,
        "trajectory_ids": [
            TrajectoryID(instance_id=uid, repetition_id=0),
            TrajectoryID(instance_id=uid, repetition_id=1),
        ],
        "is_last_step": [True, True],
        "exclude_from_baseline": [False, False],
    }
    return GeneratedOutputGroup(
        trajectory_batch=trajectory_batch,
        uid=uid,
        earliest_model_step=earliest_model_step,
        source_prompts=[{"uid": uid}],
    )


def _batch_assembly_state(
    mini_batch_size: int,
    accepted: int,
    *,
    dynamic_sampling_type: str | None = None,
    informative_on: str = "shaped",
    max_sample_batches: int = 30,
):
    trainer = object.__new__(FullyAsyncRayPPOTrainer)
    trainer.global_step = 10
    trainer.max_staleness_steps = 2
    trainer.mini_batch_size = mini_batch_size
    trainer.all_metrics = {}
    trainer._groups_rejected_since_step = 0
    trainer._rejection_reasons_since_step = collections.Counter()
    trainer._groups_inspected_since_step = 0
    trainer._group_selection_policy = GroupSelectionPolicy.for_fully_async(
        dynamic_sampling_type, criteria=resolve_dynamic_sampling_criteria(informative_on)
    )
    trainer._dynamic_sampling_type = trainer._group_selection_policy.sampling_type
    trainer._dynamic_sampling_max_sample_batches = max_sample_batches
    trainer._dynamic_sampling_max_candidate_groups = max_sample_batches * mini_batch_size
    trainer._step_time_history = collections.deque([1000.0], maxlen=5)
    trainer.group_admission_stall_timeout = 21_600
    trainer._active_generator_tasks = []
    trainer.trajectory_selector = None
    trainer._async_distillation_runtime = None
    trainer._async_distillation_tickets = {}
    trainer._staleness_manager = _AsyncStalenessManager(
        max_concurrent_generation_groups=accepted,
        mini_batch_size=mini_batch_size,
        max_staleness_steps=2,
    )
    trainer._staleness_manager._stat.submitted = accepted
    trainer._staleness_manager._stat.accepted = accepted
    trainer._group_admission_policy = GroupAdmissionPolicy(
        GroupAdvantageInvariant.exact_physical(physical_group_size=2),
        max_staleness_steps=2,
        rollout_logprobs_required=False,
    )
    trainer.data_tracker = DataConsumptionTracker(mini_batch_size=mini_batch_size, num_steps_per_epoch=1)
    queues = _GenerationQueues(
        completed=asyncio.Queue(),
        retries=asyncio.Queue(),
        condition=asyncio.Condition(),
        active_producers=1,
    )
    return trainer, queues


class _TeacherTicket:
    async def result(self):
        raise AssertionError("batch admission must not wait for teacher evidence")


class _RecordingAsyncDistillationRuntime:
    def __init__(self):
        self.submitted = asyncio.Queue()

    async def submit_before_batch_assembly(self, trajectory_batch):
        await self.submitted.put(trajectory_batch)
        return _TeacherTicket()


class _DatasetRows:
    def __init__(self, uids: list[str]):
        self._rows = [[{"uid": uid}] for uid in uids]

    def __iter__(self):
        return iter(self._rows)

    def __len__(self):
        return len(self._rows)

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        assert state == {}


@pytest.mark.asyncio
async def test_dapo_dataloader_resamples_discarded_prompts_after_finite_source_exhaustion():
    tracker = DataConsumptionTracker(mini_batch_size=2, num_steps_per_epoch=1)
    await tracker.mark_consumed(["accepted-1", "accepted-2"])
    source = StatefulDataLoader(
        [{"uid": "accepted-1"}, {"uid": "accepted-2"}, {"uid": "discarded"}],
        batch_size=1,
        shuffle=False,
        collate_fn=lambda rows: rows,
    )
    dataloader = _AsyncDataloader(
        source,
        mini_batch_size=2,
        data_tracker=tracker,
        dynamic_sampling_type=DynamicSamplingType.FILTER,
    )

    first = await dataloader.get_next_non_consumed_data()
    replacement = await dataloader.get_next_non_consumed_data()

    assert first[0]["uid"] == "discarded"
    assert replacement[0]["uid"] == "discarded"


@pytest.mark.asyncio
async def test_dapo_dataloader_stops_when_every_source_uid_was_consumed():
    tracker = DataConsumptionTracker(mini_batch_size=1, num_steps_per_epoch=1)
    await tracker.mark_consumed(["consumed"])
    dataloader = _AsyncDataloader(
        _DatasetRows(["consumed"]),
        mini_batch_size=1,
        data_tracker=tracker,
        dynamic_sampling_type=DynamicSamplingType.FILTER,
    )

    assert await dataloader.get_next_non_consumed_data() is None


@pytest.mark.asyncio
async def test_async_dataloader_without_dapo_stops_after_finite_source_exhaustion():
    tracker = DataConsumptionTracker(mini_batch_size=1, num_steps_per_epoch=1)
    dataloader = _AsyncDataloader(
        _DatasetRows(["only"]),
        mini_batch_size=1,
        data_tracker=tracker,
        dynamic_sampling_type=None,
    )

    assert (await dataloader.get_next_non_consumed_data())[0]["uid"] == "only"
    assert await dataloader.get_next_non_consumed_data() is None


@pytest.mark.asyncio
async def test_staleness_manager_blocks_work_beyond_capacity_until_training_advances():
    manager = _AsyncStalenessManager(
        max_concurrent_generation_groups=2,
        mini_batch_size=1,
        max_staleness_steps=0,
    )
    await manager.acquire_submission_slot()

    next_submission = asyncio.create_task(manager.acquire_submission_slot())
    done, _ = await asyncio.wait({next_submission}, timeout=0)
    assert next_submission not in done

    await manager.notify_capacity_change(new_global_step=2)
    await asyncio.wait_for(next_submission, timeout=1)

    await manager.on_rollout_accepted()
    await manager.on_rollout_accepted()


@pytest.mark.asyncio
async def test_batch_assembly_retries_stale_groups_from_entire_buffer():
    trainer, queues = _batch_assembly_state(mini_batch_size=2, accepted=4)
    for group in [
        _generated_group("stale-in-batch", earliest_model_step=7),
        _generated_group("fresh-1", earliest_model_step=10),
        _generated_group("fresh-2", earliest_model_step=9),
        _generated_group("stale-beyond-batch", earliest_model_step=6),
    ]:
        queues.completed.put_nowait(group)

    batch = await trainer._get_admitted_generation_group_mini_batch(queues)

    assert [group.uid for group in batch] == ["fresh-1", "fresh-2"]
    assert queues.completed.empty()
    assert [queues.retries.get_nowait()[0]["uid"] for _ in range(2)] == [
        "stale-in-batch",
        "stale-beyond-batch",
    ]
    assert trainer.all_metrics["async/rejected_count"] == 2
    assert trainer.all_metrics["async/rejected_rate"] == 0.5
    assert trainer.all_metrics["async/rejected_count/stale"] == 2


@pytest.mark.asyncio
async def test_batch_assembly_waits_for_fresh_replacement():
    trainer, queues = _batch_assembly_state(mini_batch_size=1, accepted=1)
    queues.completed.put_nowait(_generated_group("retry-me", earliest_model_step=7))

    pending_batch = asyncio.create_task(trainer._get_admitted_generation_group_mini_batch(queues))
    done, _ = await asyncio.wait({pending_batch}, timeout=0)
    assert pending_batch not in done
    assert queues.retries.get_nowait()[0]["uid"] == "retry-me"

    async with queues.condition:
        queues.completed.put_nowait(_generated_group("retry-me", earliest_model_step=10))
        queues.condition.notify_all()
    batch = await asyncio.wait_for(pending_batch, timeout=1)

    assert [group.uid for group in batch] == ["retry-me"]


@pytest.mark.asyncio
async def test_batch_assembly_submits_each_admitted_group_before_the_batch_is_complete():
    trainer, queues = _batch_assembly_state(mini_batch_size=2, accepted=2)
    runtime = _RecordingAsyncDistillationRuntime()
    trainer._async_distillation_runtime = runtime
    queues.completed.put_nowait(_generated_group("first", earliest_model_step=10))

    pending_batch = asyncio.create_task(trainer._get_admitted_generation_group_mini_batch(queues))
    first = await asyncio.wait_for(runtime.submitted.get(), timeout=1)

    assert first["trajectory_ids"][0].instance_id == "first"
    assert not pending_batch.done()

    async with queues.condition:
        queues.completed.put_nowait(_generated_group("second", earliest_model_step=10))
        queues.condition.notify_all()
    batch = await asyncio.wait_for(pending_batch, timeout=1)
    second = await asyncio.wait_for(runtime.submitted.get(), timeout=1)

    assert [group.uid for group in batch] == ["first", "second"]
    assert second["trajectory_ids"][0].instance_id == "second"


@pytest.mark.asyncio
async def test_batch_assembly_scores_only_learner_selected_rows():
    trainer, queues = _batch_assembly_state(mini_batch_size=1, accepted=1)
    runtime = _RecordingAsyncDistillationRuntime()
    trainer._async_distillation_runtime = runtime
    trainer.trajectory_selector = BestOfNTrajectorySelector(2)
    queues.completed.put_nowait(_generated_group("best", earliest_model_step=10, rewards=[0.25, 0.75]))

    batch = await trainer._get_admitted_generation_group_mini_batch(queues)
    submitted = await runtime.submitted.get()

    assert [group.uid for group in batch] == ["best"]
    assert submitted["response_ids"] == [[3]]
    assert submitted["trajectory_ids"][0].repetition_id == 1


@pytest.mark.asyncio
async def test_batch_assembly_skips_fully_masked_group_and_waits_for_fresh_prompt():
    trainer, queues = _batch_assembly_state(mini_batch_size=1, accepted=1)
    queues.completed.put_nowait(_generated_group("retry-me", earliest_model_step=10, fully_masked=True))

    pending_batch = asyncio.create_task(trainer._get_admitted_generation_group_mini_batch(queues))
    done, _ = await asyncio.wait({pending_batch}, timeout=0)
    assert pending_batch not in done
    assert queues.retries.empty()

    async with queues.condition:
        queues.completed.put_nowait(_generated_group("replacement", earliest_model_step=10))
        queues.condition.notify_all()
    batch = await asyncio.wait_for(pending_batch, timeout=1)

    assert [group.uid for group in batch] == ["replacement"]
    assert trainer.all_metrics["async/rejected_count/fully_masked"] == 1


@pytest.mark.asyncio
async def test_batch_assembly_fails_fast_on_structural_group_corruption():
    trainer, queues = _batch_assembly_state(mini_batch_size=1, accepted=1)
    trainer._group_admission_policy = GroupAdmissionPolicy(
        GroupAdvantageInvariant.exact_physical(physical_group_size=3),
        max_staleness_steps=2,
        rollout_logprobs_required=False,
    )
    queues.completed.put_nowait(_generated_group("wrong-size", earliest_model_step=10))

    with pytest.raises(TrainingGroupInvariantError, match="physical_group_size"):
        await trainer._get_admitted_generation_group_mini_batch(queues)


@pytest.mark.asyncio
async def test_batch_assembly_discards_insufficient_reward_spread_and_waits_for_fresh_prompt():
    trainer, queues = _batch_assembly_state(
        mini_batch_size=1, accepted=1, dynamic_sampling_type="filter", informative_on="unshaped"
    )
    queues.completed.put_nowait(
        _generated_group(
            "uniform",
            earliest_model_step=10,
            rewards=[-0.1, -0.1],
            unshaped_rewards=[0.0, 0.0],
        )
    )

    pending_batch = asyncio.create_task(trainer._get_admitted_generation_group_mini_batch(queues))
    done, _ = await asyncio.wait({pending_batch}, timeout=0)
    assert pending_batch not in done
    assert queues.retries.empty()

    async with queues.condition:
        queues.completed.put_nowait(
            _generated_group(
                "fresh",
                earliest_model_step=10,
                rewards=[[0.2, 0.3], [0.0, 1.0]],
            )
        )
        queues.condition.notify_all()
    batch = await asyncio.wait_for(pending_batch, timeout=1)

    assert [group.uid for group in batch] == ["fresh"]
    assert trainer.all_metrics["async/dynamic_sampling/discarded_count"] == 1
    assert trainer.all_metrics["async/dynamic_sampling/candidate_count"] == 2
    assert trainer.all_metrics["async/dynamic_sampling/candidate_trajectory_count"] == 4
    assert trainer.all_metrics["async/dynamic_sampling/candidate_optimization_reward_mean"] == pytest.approx(0.325)
    assert trainer.all_metrics["async/dynamic_sampling/candidate_outcome_reward_mean"] == pytest.approx(0.25)
    assert trainer.all_metrics["async/dynamic_sampling/candidate_pass_at_2"] == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_batch_assembly_routes_stale_and_uniform_groups_differently():
    trainer, queues = _batch_assembly_state(
        mini_batch_size=1, accepted=2, dynamic_sampling_type="filter", informative_on="unshaped"
    )
    queues.completed.put_nowait(_generated_group("stale", earliest_model_step=7))
    queues.completed.put_nowait(_generated_group("uniform", earliest_model_step=10, unshaped_rewards=[1.0, 1.0]))

    pending_batch = asyncio.create_task(trainer._get_admitted_generation_group_mini_batch(queues))
    done, _ = await asyncio.wait({pending_batch}, timeout=0)
    assert pending_batch not in done
    assert queues.retries.get_nowait()[0]["uid"] == "stale"
    assert queues.retries.empty()

    async with queues.condition:
        queues.completed.put_nowait(_generated_group("fresh", earliest_model_step=10))
        queues.condition.notify_all()
    batch = await asyncio.wait_for(pending_batch, timeout=1)

    assert [group.uid for group in batch] == ["fresh"]


@pytest.mark.asyncio
async def test_batch_assembly_fails_when_dynamic_sampling_exhausts_candidate_budget():
    trainer, queues = _batch_assembly_state(
        mini_batch_size=2,
        accepted=2,
        dynamic_sampling_type=DynamicSamplingType.FILTER,
        informative_on="unshaped",
        max_sample_batches=1,
    )
    queues.completed.put_nowait(_generated_group("uniform-1", earliest_model_step=10, unshaped_rewards=[0.0, 0.0]))
    queues.completed.put_nowait(_generated_group("uniform-2", earliest_model_step=10, unshaped_rewards=[1.0, 1.0]))

    with pytest.raises(RuntimeError, match="dynamic sampling limit"):
        await trainer._get_admitted_generation_group_mini_batch(queues)


@pytest.mark.asyncio
async def test_dapo_replacement_sampling_remains_bounded_by_candidate_budget():
    tracker = DataConsumptionTracker(mini_batch_size=1, num_steps_per_epoch=1)
    await tracker.mark_consumed(["trained"])
    dataloader = _AsyncDataloader(
        _DatasetRows(["trained", "discarded"]),
        mini_batch_size=1,
        data_tracker=tracker,
        dynamic_sampling_type=DynamicSamplingType.FILTER,
    )
    trainer, queues = _batch_assembly_state(
        mini_batch_size=1,
        accepted=2,
        dynamic_sampling_type="filter",
        informative_on="unshaped",
        max_sample_batches=2,
    )

    first_prompts = await dataloader.get_next_non_consumed_data()
    queues.completed.put_nowait(
        _generated_group(first_prompts[0]["uid"], earliest_model_step=10, unshaped_rewards=[0.0, 0.0])
    )
    pending_batch = asyncio.create_task(trainer._get_admitted_generation_group_mini_batch(queues))
    done, _ = await asyncio.wait({pending_batch}, timeout=0)
    assert pending_batch not in done

    replacement_prompts = await dataloader.get_next_non_consumed_data()
    assert replacement_prompts[0]["uid"] == first_prompts[0]["uid"]
    async with queues.condition:
        queues.completed.put_nowait(
            _generated_group(replacement_prompts[0]["uid"], earliest_model_step=10, unshaped_rewards=[1.0, 1.0])
        )
        queues.condition.notify_all()

    with pytest.raises(RuntimeError, match="dynamic sampling limit"):
        await asyncio.wait_for(pending_batch, timeout=1)


@pytest.mark.asyncio
async def test_batch_assembly_scans_rejections_and_preserves_accepted_surplus():
    trainer, queues = _batch_assembly_state(mini_batch_size=2, accepted=4)
    for group in [
        _generated_group("accepted-1", earliest_model_step=10),
        _generated_group("accepted-2", earliest_model_step=10),
        _generated_group("masked-beyond-batch", earliest_model_step=10, fully_masked=True),
        _generated_group("accepted-surplus", earliest_model_step=10),
    ]:
        queues.completed.put_nowait(group)

    batch = await trainer._get_admitted_generation_group_mini_batch(queues)

    assert [group.uid for group in batch] == ["accepted-1", "accepted-2"]
    assert queues.retries.empty()
    assert queues.completed.get_nowait().uid == "accepted-surplus"


@pytest.mark.asyncio
async def test_batch_assembly_discards_duplicate_uid_and_fills_the_batch():
    trainer, queues = _batch_assembly_state(mini_batch_size=2, accepted=3)
    queues.completed.put_nowait(_generated_group("duplicate", earliest_model_step=10))
    queues.completed.put_nowait(_generated_group("duplicate", earliest_model_step=10))
    queues.completed.put_nowait(_generated_group("unique", earliest_model_step=10))

    batch = await trainer._get_admitted_generation_group_mini_batch(queues)

    assert [group.uid for group in batch] == ["duplicate", "unique"]
    assert queues.retries.empty()
    assert queues.completed.empty()
    assert trainer.all_metrics["async/rejected_count/duplicate_uid"] == 1


@pytest.mark.asyncio
async def test_batch_assembly_discards_duplicate_uid_received_in_a_later_scan():
    trainer, queues = _batch_assembly_state(mini_batch_size=2, accepted=3)
    queues.completed.put_nowait(_generated_group("first", earliest_model_step=10))

    pending_batch = asyncio.create_task(trainer._get_admitted_generation_group_mini_batch(queues))
    done, _ = await asyncio.wait({pending_batch}, timeout=0)
    assert pending_batch not in done

    async with queues.condition:
        queues.completed.put_nowait(_generated_group("first", earliest_model_step=10))
        queues.completed.put_nowait(_generated_group("second", earliest_model_step=10))
        queues.condition.notify_all()

    batch = await asyncio.wait_for(pending_batch, timeout=1)

    assert [group.uid for group in batch] == ["first", "second"]
    assert trainer.all_metrics["async/rejected_count/duplicate_uid"] == 1


@pytest.mark.asyncio
async def test_batch_assembly_does_not_readmit_a_uid_consumed_by_an_earlier_step():
    trainer, queues = _batch_assembly_state(mini_batch_size=1, accepted=2)
    await trainer.data_tracker.mark_consumed(["trained"])
    queues.completed.put_nowait(_generated_group("trained", earliest_model_step=10))
    queues.completed.put_nowait(_generated_group("fresh", earliest_model_step=10))

    batch = await trainer._get_admitted_generation_group_mini_batch(queues)

    assert [group.uid for group in batch] == ["fresh"]
    assert trainer.all_metrics["async/rejected_count/duplicate_uid"] == 1


@pytest.mark.asyncio
async def test_batch_assembly_prefers_eligible_duplicate_without_scheduling_a_retry():
    trainer, queues = _batch_assembly_state(mini_batch_size=1, accepted=2)
    queues.completed.put_nowait(_generated_group("same", earliest_model_step=10, fully_masked=True))
    queues.completed.put_nowait(_generated_group("same", earliest_model_step=10))

    batch = await trainer._get_admitted_generation_group_mini_batch(queues)

    assert [group.uid for group in batch] == ["same"]
    assert queues.retries.empty()
    assert trainer.all_metrics["async/rejected_count/duplicate_uid"] == 1


@pytest.mark.asyncio
async def test_batch_assembly_skips_masked_duplicates_without_retrying_the_prompt():
    trainer, queues = _batch_assembly_state(mini_batch_size=1, accepted=3)
    queues.completed.put_nowait(_generated_group("masked", earliest_model_step=10, fully_masked=True))
    queues.completed.put_nowait(_generated_group("masked", earliest_model_step=10, fully_masked=True))
    queues.completed.put_nowait(_generated_group("replacement", earliest_model_step=10))

    batch = await trainer._get_admitted_generation_group_mini_batch(queues)

    assert [group.uid for group in batch] == ["replacement"]
    assert queues.retries.empty()
    assert trainer.all_metrics["async/rejected_count/fully_masked"] == 1
    assert trainer.all_metrics["async/rejected_count/duplicate_uid"] == 1


@pytest.mark.asyncio
async def test_resume_skips_uids_owned_by_restored_completed_groups_and_retries(tmp_path):
    tracker = DataConsumptionTracker(mini_batch_size=1, num_steps_per_epoch=4)
    await tracker.mark_consumed(["consumed"])
    dataloader = _AsyncDataloader(
        _DatasetRows(["consumed", "completed", "retry", "unscheduled"]),
        mini_batch_size=1,
        data_tracker=tracker,
    )
    dataloader.load_state_from_checkpoint()

    trainer, _ = _batch_assembly_state(mini_batch_size=1, accepted=0)
    trainer.async_train_dataloader = dataloader
    queues = _GenerationQueues(
        completed=asyncio.Queue(maxsize=4), retries=asyncio.Queue(), condition=asyncio.Condition()
    )
    torch_state = {
        "completed_groups": [
            {
                "trajectory_batch": dict(_generated_group("completed", 10).trajectory_batch),
                "uid": "completed",
                "earliest_model_step": 10,
                "source_prompts": [{"uid": "completed"}],
            }
        ],
        "retry_prompts": [[{"uid": "retry"}]],
    }
    torch.save(torch_state, tmp_path / "generation_buffer_state.pt")

    trainer._restore_buffer_from_checkpoint(queues, str(tmp_path))

    assert (await dataloader.get_next_non_consumed_data())[0]["uid"] == "unscheduled"


@pytest.mark.asyncio
async def test_restore_continues_a_partially_admitted_batch(tmp_path):
    trainer, _ = _batch_assembly_state(mini_batch_size=2, accepted=0)

    class _PendingUIDs:
        def __init__(self):
            self.reserved = set()

        def reserve_pending_uids(self, _uids):
            self.reserved.update(_uids)

    pending_uids = _PendingUIDs()
    trainer.async_train_dataloader = pending_uids
    queues = _GenerationQueues(
        completed=asyncio.Queue(maxsize=2), retries=asyncio.Queue(), condition=asyncio.Condition()
    )
    torch.save(
        {
            "completed_groups": [],
            "admitted_groups": [
                {
                    "trajectory_batch": dict(_generated_group("banked", 10).trajectory_batch),
                    "uid": "banked",
                    "earliest_model_step": 10,
                    "source_prompts": [{"uid": "banked"}],
                }
            ],
            "retry_prompts": [],
        },
        tmp_path / "generation_buffer_state.pt",
    )

    trainer._restore_buffer_from_checkpoint(queues, str(tmp_path))
    assert pending_uids.reserved == {"banked"}
    queues.completed.put_nowait(_generated_group("replacement", earliest_model_step=10))

    batch = await trainer._get_admitted_generation_group_mini_batch(queues)
    assert [group.uid for group in batch] == ["banked", "replacement"]


@pytest.mark.asyncio
async def test_batch_assembly_rejected_only_progress_terminates_instead_of_livelocking():
    trainer, queues = _batch_assembly_state(mini_batch_size=1, accepted=1)
    trainer.group_admission_stall_timeout = 1e-9
    queues.completed.put_nowait(_generated_group("always-masked", earliest_model_step=10, fully_masked=True))

    with pytest.raises(GroupAdmissionStalledError):
        await trainer._get_admitted_generation_group_mini_batch(queues)
