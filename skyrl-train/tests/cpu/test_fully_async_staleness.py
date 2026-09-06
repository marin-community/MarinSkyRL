import asyncio
import collections

import pytest
import torch
from torchdata.stateful_dataloader import StatefulDataLoader

from skyrl_train.fully_async_trainer import (
    FullyAsyncRayPPOTrainer,
    GenerationStalledError,
    GeneratedOutputGroup,
    _AsyncDataloader,
    _AsyncStalenessManager,
    _GenerationQueues,
)
from skyrl_train.dynamic_sampling import DynamicSamplingType, GroupSelectionPolicy, resolve_dynamic_sampling_criteria
from skyrl_train.group_admission import GroupAdmissionPolicy, GroupAdvantageInvariant
from skyrl_train.trajectory_runners.base import TrajectoryID
from skyrl_train.utils.data_tracker import DataConsumptionTracker


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
    trainer.admission_stall_timeout = 21_600
    trainer._active_generator_tasks = []
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

    await manager.notify_policy_weights_published(completed_step=1)
    await manager.notify_capacity_change(new_global_step=2)
    await asyncio.wait_for(next_submission, timeout=1)

    await manager.on_rollout_accepted()
    await manager.on_rollout_accepted()


@pytest.mark.asyncio
async def test_unpublished_update_cannot_admit_a_third_cohort():
    manager = _AsyncStalenessManager(
        max_concurrent_generation_groups=4,
        mini_batch_size=2,
        max_staleness_steps=1,
    )
    for _ in range(4):
        await manager.acquire_submission_slot()
        await manager.on_rollout_accepted()

    await manager.notify_capacity_change(new_global_step=2)
    third_cohort = asyncio.create_task(manager.acquire_submission_slot())
    try:
        done, _ = await asyncio.wait({third_cohort}, timeout=0)
        assert not done, "weights from update zero cannot supply a third eligible batch"
        await manager.notify_policy_weights_published(completed_step=2)
        await asyncio.wait_for(third_cohort, timeout=1)
        await manager.on_rollout_accepted()
    finally:
        third_cohort.cancel()
        await asyncio.gather(third_cohort, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("max_age", [0, 1, 3])
async def test_cadence_one_releases_exactly_one_batch_per_completed_update(max_age):
    manager = _AsyncStalenessManager(
        max_concurrent_generation_groups=16, mini_batch_size=2, max_staleness_steps=max_age
    )
    for _ in range((max_age + 1) * 2):
        await manager.acquire_submission_slot()
        await manager.on_rollout_accepted()

    pending = None
    try:
        for completed_step in range(1, 4):
            pending = asyncio.create_task(manager.acquire_submission_slot())
            done, _ = await asyncio.wait({pending}, timeout=0)
            assert not done
            await manager.notify_policy_weights_published(completed_step)
            done, _ = await asyncio.wait({pending}, timeout=0)
            assert not done, "publishing alone must preserve the cadence-one learner capacity"
            await manager.notify_capacity_change(new_global_step=completed_step + 1)
            await asyncio.wait_for(pending, timeout=1)
            await manager.on_rollout_accepted()
            await asyncio.wait_for(manager.acquire_submission_slot(), timeout=1)
            await manager.on_rollout_accepted()
        pending = asyncio.create_task(manager.acquire_submission_slot())
        done, _ = await asyncio.wait({pending}, timeout=0)
        assert not done
    finally:
        if pending is not None:
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("accepted_before_rejection", [False, True])
async def test_stale_attempt_releases_capacity_for_replacement_after_publication(accepted_before_rejection):
    manager = _AsyncStalenessManager(max_concurrent_generation_groups=3, mini_batch_size=1, max_staleness_steps=1)
    await manager.acquire_submission_slot()
    await manager.on_rollout_accepted()
    await manager.acquire_submission_slot()  # A slow update-zero rollout remains in flight.
    await manager.notify_capacity_change(new_global_step=2)
    await manager.notify_policy_weights_published(completed_step=1)  # Off-grid evaluation.
    await manager.acquire_submission_slot()
    await manager.on_rollout_accepted()  # Fresh work supplies the second training batch.
    await manager.notify_capacity_change(new_global_step=3)

    replacement = asyncio.create_task(manager.acquire_submission_slot())
    try:
        done, _ = await asyncio.wait({replacement}, timeout=0)
        assert not done
        if accepted_before_rejection:
            await manager.on_rollout_accepted()
            await manager.on_rollouts_discarded(1)
        else:
            await manager.cancel_submission_slot()
        await asyncio.wait_for(replacement, timeout=1)
        await manager.on_rollout_accepted()
    finally:
        replacement.cancel()
        await asyncio.gather(replacement, return_exceptions=True)


@pytest.mark.asyncio
async def test_resumed_capacity_waits_for_checkpoint_weights_to_be_installed():
    manager = _AsyncStalenessManager(max_concurrent_generation_groups=4, mini_batch_size=2, max_staleness_steps=1)
    manager.load_state_from_checkpoint(global_step=5)
    pending = asyncio.create_task(manager.acquire_submission_slot())
    try:
        done, _ = await asyncio.wait({pending}, timeout=0)
        assert not done
        await manager.notify_policy_weights_published(completed_step=4)
        await asyncio.wait_for(pending, timeout=1)
        await manager.on_rollout_accepted()
    finally:
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)


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
async def test_batch_assembly_retries_fully_masked_group_and_waits_for_replacement():
    trainer, queues = _batch_assembly_state(mini_batch_size=1, accepted=1)
    queues.completed.put_nowait(_generated_group("retry-me", earliest_model_step=10, fully_masked=True))

    pending_batch = asyncio.create_task(trainer._get_admitted_generation_group_mini_batch(queues))
    done, _ = await asyncio.wait({pending_batch}, timeout=0)
    assert pending_batch not in done
    assert queues.retries.get_nowait()[0]["uid"] == "retry-me"

    async with queues.condition:
        queues.completed.put_nowait(_generated_group("replacement", earliest_model_step=10))
        queues.condition.notify_all()
    batch = await asyncio.wait_for(pending_batch, timeout=1)

    assert [group.uid for group in batch] == ["replacement"]
    assert trainer.all_metrics["async/rejected_count/fully_masked"] == 1


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
    assert queues.retries.get_nowait()[0]["uid"] == "masked-beyond-batch"
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
async def test_batch_assembly_retries_duplicate_uid_at_most_once_when_all_copies_are_masked():
    trainer, queues = _batch_assembly_state(mini_batch_size=1, accepted=3)
    queues.completed.put_nowait(_generated_group("masked", earliest_model_step=10, fully_masked=True))
    queues.completed.put_nowait(_generated_group("masked", earliest_model_step=10, fully_masked=True))
    queues.completed.put_nowait(_generated_group("replacement", earliest_model_step=10))

    batch = await trainer._get_admitted_generation_group_mini_batch(queues)

    assert [group.uid for group in batch] == ["replacement"]
    assert queues.retries.qsize() == 1
    assert queues.retries.get_nowait()[0]["uid"] == "masked"
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
    trainer.admission_stall_timeout = 0.0
    queues.completed.put_nowait(_generated_group("always-masked", earliest_model_step=10, fully_masked=True))

    with pytest.raises(GenerationStalledError):
        await trainer._get_admitted_generation_group_mini_batch(queues)
