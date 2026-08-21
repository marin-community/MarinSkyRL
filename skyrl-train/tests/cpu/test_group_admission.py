import pytest

from skyrl_train.fully_async_trainer import GeneratedOutputGroup
from skyrl_train.dynamic_sampling import GroupSelectionPolicy, GroupSelectionResult
from skyrl_train.group_admission import (
    AdmissionRejection,
    GroupAdmissionPolicy,
    GroupAdvantageInvariant,
    TrainingGroupInvariantError,
    assert_training_groups_eligible,
    resolve_group_advantage_invariant,
)


def _group(
    *,
    loss_masks: list[list[int]],
    exclude_from_baseline: list[bool] | None = None,
    rollout_logprobs: list[list[float | None]] | None = None,
    earliest_model_step: int = 10,
) -> GeneratedOutputGroup:
    group_size = len(loss_masks)
    trajectory_batch = {
        "prompt_token_ids": [[1] for _ in range(group_size)],
        "response_ids": [[2] for _ in range(group_size)],
        "rewards": [0.0 for _ in range(group_size)],
        "loss_masks": loss_masks,
        "stop_reasons": ["stop" for _ in range(group_size)],
        "rollout_metrics": {},
        "rollout_logprobs": rollout_logprobs,
        "exclude_from_baseline": exclude_from_baseline,
    }
    return GeneratedOutputGroup(
        trajectory_batch=trajectory_batch,
        uid="group",
        earliest_model_step=earliest_model_step,
        source_prompts=[{"uid": "group"}],
    )


def test_exact_group_accepts_masked_trial_when_group_can_train():
    policy = GroupAdmissionPolicy(
        GroupAdvantageInvariant.exact_physical(physical_group_size=3),
        max_staleness_steps=2,
        rollout_logprobs_required=False,
    )

    decision = policy.evaluate(_group(loss_masks=[[1], [0], [1]]), global_step=10)

    assert decision.accepted


def test_fully_masked_group_is_rejected_independently_of_estimator():
    policy = GroupAdmissionPolicy(
        GroupAdvantageInvariant.no_group_advantage(physical_group_size=2),
        max_staleness_steps=2,
        rollout_logprobs_required=False,
    )

    decision = policy.evaluate(_group(loss_masks=[[0], [0]]), global_step=10)

    assert decision.primary_rejection is AdmissionRejection.FULLY_MASKED


def test_non_group_estimator_bypasses_physical_cardinality():
    policy = GroupAdmissionPolicy(
        GroupAdvantageInvariant.no_group_advantage(physical_group_size=4),
        max_staleness_steps=2,
        rollout_logprobs_required=False,
    )

    decision = policy.evaluate(_group(loss_masks=[[1]]), global_step=10)

    assert decision.accepted


def test_minimum_group_counts_baseline_members_independently_of_loss_mask():
    policy = GroupAdmissionPolicy(
        GroupAdvantageInvariant.minimum_baseline_eligible(physical_group_size=3, minimum_group_size=2),
        max_staleness_steps=2,
        rollout_logprobs_required=False,
    )

    decision = policy.evaluate(
        _group(loss_masks=[[1], [0], [0]], exclude_from_baseline=[False, False, True]),
        global_step=10,
    )

    assert decision.accepted


def test_minimum_group_treats_missing_exclusions_as_all_baseline_eligible():
    policy = GroupAdmissionPolicy(
        GroupAdvantageInvariant.minimum_baseline_eligible(physical_group_size=2, minimum_group_size=2),
        max_staleness_steps=2,
        rollout_logprobs_required=False,
    )

    decision = policy.evaluate(_group(loss_masks=[[1], [0]], exclude_from_baseline=None), global_step=10)

    assert decision.accepted


def test_minimum_group_rejects_cohort_below_floor():
    policy = GroupAdmissionPolicy(
        GroupAdvantageInvariant.minimum_baseline_eligible(physical_group_size=3, minimum_group_size=2),
        max_staleness_steps=2,
        rollout_logprobs_required=False,
    )

    decision = policy.evaluate(
        _group(loss_masks=[[1], [1], [0]], exclude_from_baseline=[False, True, True]),
        global_step=10,
    )

    assert decision.primary_rejection is AdmissionRejection.BELOW_MINIMUM_GROUP_SIZE


def test_required_logprobs_reject_missing_values_only_for_trainable_group():
    policy = GroupAdmissionPolicy(
        GroupAdvantageInvariant.exact_physical(physical_group_size=2),
        max_staleness_steps=2,
        rollout_logprobs_required=True,
    )

    decision = policy.evaluate(_group(loss_masks=[[1], [0]], rollout_logprobs=None), global_step=10)

    assert decision.primary_rejection is AdmissionRejection.MISSING_ROLLOUT_LOGPROBS


def test_required_logprobs_allow_placeholders_only_at_masked_tokens():
    policy = GroupAdmissionPolicy(
        GroupAdvantageInvariant.exact_physical(physical_group_size=2),
        max_staleness_steps=2,
        rollout_logprobs_required=True,
    )
    group = _group(loss_masks=[[1], [0]], rollout_logprobs=[[None], [None]])

    decision = policy.evaluate(group, global_step=10)

    assert decision.primary_rejection is AdmissionRejection.MISSING_ROLLOUT_LOGPROBS

    group.trajectory_batch["rollout_logprobs"] = [[-0.5], [None]]
    assert policy.evaluate(group, global_step=10).accepted


def test_malformed_group_fails_instead_of_being_retried():
    policy = GroupAdmissionPolicy(
        GroupAdvantageInvariant.exact_physical(physical_group_size=2),
        max_staleness_steps=2,
        rollout_logprobs_required=False,
    )
    group = _group(loss_masks=[[1], [1]])
    group.trajectory_batch["loss_masks"] = [[1]]

    with pytest.raises(ValueError, match="loss_masks"):
        policy.evaluate(group, global_step=10)


def test_training_group_error_reports_observed_and_expected_physical_counts():
    group = _group(loss_masks=[[1], [1], [1], [1]])

    with pytest.raises(TrainingGroupInvariantError) as exc_info:
        assert_training_groups_eligible(
            group.trajectory_batch,
            ["duplicate"] * 4,
            GroupAdvantageInvariant.exact_physical(physical_group_size=2),
        )

    assert exc_info.value.uid == "duplicate"
    assert exc_info.value.rejections == (AdmissionRejection.PHYSICAL_GROUP_SIZE,)
    assert exc_info.value.physical_count == 4
    assert exc_info.value.expected_physical_count == 2
    assert exc_info.value.row_indices == (0, 1, 2, 3)


def test_stepwise_group_counts_final_trials_but_checks_all_transitions_for_training():
    policy = GroupAdmissionPolicy(
        GroupAdvantageInvariant.exact_physical(physical_group_size=2),
        max_staleness_steps=2,
        rollout_logprobs_required=False,
    )
    group = _group(loss_masks=[[1], [0], [1], [0]])
    group.trajectory_batch["is_last_step"] = [False, True, False, True]

    decision = policy.evaluate(group, global_step=10)

    assert decision.accepted


@pytest.mark.parametrize("minimum_group_size", [None, 1, 5])
def test_rloo_n_group_floor_must_support_leave_one_out(minimum_group_size):
    with pytest.raises(ValueError):
        resolve_group_advantage_invariant(
            advantage_estimator="rloo_n",
            physical_group_size=4,
            minimum_group_size=minimum_group_size,
        )


def test_grpo_rejects_unused_group_floor():
    with pytest.raises(ValueError, match="set it to null"):
        resolve_group_advantage_invariant(
            advantage_estimator="grpo",
            physical_group_size=4,
            minimum_group_size=2,
        )


def test_dynamic_filter_uses_final_unshaped_outcomes():
    policy = GroupSelectionPolicy.for_fully_async("filter")
    group = _group(loss_masks=[[1], [1], [1], [1]])
    group.trajectory_batch.update(
        {
            "rewards": [0.0, 0.0, 0.0, -0.1],
            "unshaped_rewards": [0.0, 1.0, 0.0, 1.0],
            "is_last_step": [False, True, False, True],
        }
    )

    decision = policy.evaluate(group)

    assert decision is GroupSelectionResult.UNIFORM_OUTCOMES


def test_dynamic_filter_requires_unshaped_outcomes():
    policy = GroupSelectionPolicy.for_fully_async("filter")

    with pytest.raises(ValueError, match="requires unshaped_rewards"):
        policy.evaluate(_group(loss_masks=[[1], [1]]))


def test_fully_async_selection_rejects_replace_sampling():
    with pytest.raises(ValueError, match="supports dynamic_sampling.type=filter or null"):
        GroupSelectionPolicy.for_fully_async("replace")
