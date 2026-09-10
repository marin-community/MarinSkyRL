"""Frozen async cohorts preserve native DP order and checkpointed preparation."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from marinskyrl.runtime_options import R3Transport
from skyrl_train.async_rollout_state import (
    GeneratedOutputGroup,
    GenerationBufferState,
    PreparedAsyncCohort,
    consumed_token_version_metrics,
)
from skyrl_train.callbacks.builtin import BufferCheckpointCallback
from skyrl_train.distributed.dispatch import ActorInfo, DispatchSettings, MeshDispatch, MeshRank
from skyrl_train.training_batch import TrainingBatchIterator, TrainingInputBatch
from skyrl_train.fully_async_trainer import _GenerationQueues

import asyncio


def make_cohort():
    groups = [GeneratedOutputGroup({"response_ids": [[1]] * 4}, str(i), 1, [{"uid": str(i)}]) for i in range(128)]
    row = torch.arange(512).unsqueeze(1)
    mask = torch.arange(7).unsqueeze(0) <= row % 7
    batch = TrainingInputBatch(
        dict(
            sequences=row.clone(),
            action_log_probs=row.float() / -100,
            base_action_log_probs=None,
            rollout_logprobs=row.float() / -90,
            advantages=row.float() / 11,
            values=None,
            returns=None,
            loss_mask=mask,
            response_mask=mask.clone(),
            attention_mask=mask.clone(),
        )
    )
    batch.metadata = {"uids": [str(i) for i in range(128) for _ in range(4)], "response_length": 7}
    return PreparedAsyncCohort(batch, groups, admission_step=1, dp_size=4, mini_batch_groups=64, samples_per_prompt=4)


def dispatch_to_iterators(batch):
    # Only the remote actor boundary is replaced. Native dispatch and native
    # worker minibatch iteration choose all rows and tensor fields.
    actors = [
        ActorInfo(
            SimpleNamespace(ppo_train=SimpleNamespace(remote=lambda shard: list(TrainingBatchIterator(shard, 64)))),
            MeshRank(dp=rank, sp=0, tp=0, pp=0, world_size=4, dp_size=4, pp_size=1),
        )
        for rank in range(4)
    ]
    return MeshDispatch.dispatch(
        actors,
        "ppo_train",
        batch,
        settings=DispatchSettings(r3_transport=R3Transport.BY_VALUE, r3_dispatch_put_timeout_seconds=0),
    )


def test_cohort_partitions_match_native_sync_n2_dispatch_and_frozen_inputs():
    cohort = make_cohort()
    native_sync = dispatch_to_iterators(cohort.batch)
    seen, tokens = [], 0
    for update in range(2):
        part = cohort.partition(consume_step=1 + update)
        actual = dispatch_to_iterators(part)
        for rank in range(4):
            assert len(actual[rank]) == 1
            for field in ("sequences", "action_log_probs", "advantages", "rollout_logprobs", "loss_mask"):
                assert torch.equal(getattr(actual[rank][0], field), getattr(native_sync[rank][update], field))
        seen.extend(part["sequences"].flatten().tolist())
        tokens += part["loss_mask"].sum().item()
        assert part["rollout_age"].tolist() == [update] * 256
        assert part.metadata["async_cohort_admission_ages"] == [0] * 64
        assert part.metadata["async_cohort_update_index"] == update
        part["action_log_probs"].fill_(-999)
        part.metadata["uids"][0] = "mutated"
        assert cohort.batch["action_log_probs"][0].item() == 0
        assert cohort.batch.metadata["uids"][0] == "0"
        cohort = cohort.advanced()
    assert sorted(seen) == list(range(512))
    assert tokens == cohort.batch["loss_mask"].sum().item()
    assert cohort.pending_groups() == []


@pytest.mark.asyncio
async def test_real_buffer_checkpoint_preserves_second_partition_without_repreparation(tmp_path):
    original = make_cohort().advanced()
    state = GenerationBufferState([], [], prepared_cohort=original)
    callback = BufferCheckpointCallback()
    callback.bind_queues(SimpleNamespace(shutdown_snapshot=lambda: state))
    await callback.flush_to_checkpoint(str(tmp_path))
    restored = callback.load_buffer_state(str(tmp_path))
    assert restored.pending_uids() == {str(i) for i in original.group_indices()}
    assert len(restored.pending_uids()) == 64
    actual = restored.prepared_cohort.partition(consume_step=2)
    expected = original.partition(consume_step=2)
    assert actual.metadata == expected.metadata
    for key in actual:
        assert actual[key] is None if expected[key] is None else torch.equal(actual[key], expected[key])
    assert restored.prepared_cohort.next_update == 1
    with pytest.raises(ValueError, match="successful-update clock"):
        restored.prepared_cohort.partition(consume_step=3)


@pytest.mark.parametrize("changes", [{"dp_size": 3}, {"next_update": 3}, {"mini_batch_groups": 63}])
def test_invalid_prepared_geometry_rejects(changes):
    with pytest.raises(ValueError):
        replace(make_cohort(), **changes)


def test_shutdown_snapshot_replays_only_the_uncheckpointed_partition():
    queues = _GenerationQueues(asyncio.Queue(), asyncio.Queue(), asyncio.Condition())
    cohort = make_cohort()
    queues.install_prepared_cohort(cohort)
    queues.mark_prepared_partition_consumed()
    assert queues.snapshot().prepared_cohort.next_update == 1
    assert queues.shutdown_snapshot().prepared_cohort.next_update == 0
    queues.clear_admitted()
    queues.mark_prepared_partition_consumed()
    assert queues.snapshot().prepared_cohort is None
    assert queues.shutdown_snapshot().prepared_cohort.next_update == 1
    assert len(queues.shutdown_snapshot().pending_uids()) == 64
    queues.clear_admitted()
    assert queues.shutdown_snapshot().pending_uids() == set()


@pytest.mark.parametrize("step,fractions,newest_age", [(8, [0.25, 0.25, 0.25, 0.25], 0), (9, [0, 0.25, 0.25, 0.5], 1)])
def test_token_age_metrics_preserve_unknowns_and_ignore_masked_observations(step, fractions, newest_age):
    groups = [
        GeneratedOutputGroup(
            {
                "response_ids": [[1, 2, 3, 4]],
                "loss_masks": [[1, 0, 1, 1]],
                "rollout_versions": [[5, 100, 7, None]],
                "rollout_abort_counts": [2],
            },
            "a",
            5,
            [],
        ),
        GeneratedOutputGroup(
            {
                "response_ids": [[5, 6]],
                "loss_masks": [[1, 1]],
                "rollout_versions": [[6, 8]],
                "rollout_abort_counts": [None],
            },
            "b",
            6,
            [],
            latest_model_step=8,
        ),
    ]
    metrics = consumed_token_version_metrics(groups, step)
    assert metrics["async/token_version_known_tokens"] == 4
    assert metrics["async/token_version_unknown_tokens"] == 1
    assert metrics["async/token_version_known_fraction"] == 0.8
    assert metrics["async/token_version_complete_group_fraction"] == 0.5
    assert metrics["async/version_spread_mean"] == metrics["async/version_spread_max"] == 2
    assert metrics["async/staleness_newest_mean"] == newest_age
    assert [metrics[f"async/token_age_frac/{bucket}"] for bucket in (0, 1, 2, "3plus")] == fractions
    assert metrics["async/response_abort_count"] == 2
    assert metrics["async/response_abort_count_known_fraction"] == 0.5


def test_abort_metrics_preserve_counts_without_token_version_evidence():
    group = GeneratedOutputGroup(
        {"response_ids": [[1, 2]], "loss_masks": [[1, 1]], "rollout_abort_counts": [2]},
        "a",
        5,
        [],
    )
    metrics = consumed_token_version_metrics([group], 8)
    assert metrics["async/response_abort_count"] == 2
    assert metrics["async/response_abort_count_known_fraction"] == 1
    assert metrics["async/token_version_unknown_tokens"] == 2
    assert metrics["async/token_version_known_fraction"] == 0
    assert "async/staleness_newest_mean" not in metrics
    assert "async/token_age_frac/0" not in metrics


def test_known_token_versions_with_explicit_null_abort_channel_keep_unknown_coverage():
    group = GeneratedOutputGroup(
        {"response_ids": [[7]], "loss_masks": [[1]], "rollout_versions": [[1]], "rollout_abort_counts": None},
        "a",
        1,
        [],
    )
    metrics = consumed_token_version_metrics([group], 2)
    assert metrics["async/token_version_known_fraction"] == 1
    assert metrics["async/response_abort_count_known_fraction"] == 0
    assert "async/response_abort_count" not in metrics
