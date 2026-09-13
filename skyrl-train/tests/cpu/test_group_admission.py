import pytest

from skyrl_train.fully_async_trainer import GeneratedOutputGroup
from skyrl_train.dynamic_sampling import (
    GroupSelectionPolicy,
    GroupSelectionResult,
    resolve_dynamic_sampling_criteria,
)
from skyrl_train.group_admission import (
    AdmissionAction,
    AdmissionDecision,
    AdmissionProgressWatchdog,
    AdmissionRejection,
    GroupAdmissionPolicy,
    GroupAdvantageInvariant,
    TrainingGroupInvariantError,
    assert_training_groups_eligible,
    resolve_group_advantage_invariant,
)
from skyrl_train.trajectory_runners.trajectory_reward_shaping import shape_trajectory_rewards


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


@pytest.mark.parametrize(
    ("history", "expected"),
    [
        ([], 1800.0),
        ([100.0, 200.0, 300.0], 1000.0),
        ([1.0, 2.0, 3.0], 600.0),
    ],
)
def test_admission_watchdog_uses_shared_adaptive_timeout(history, expected):
    watchdog = AdmissionProgressWatchdog.start(now=10.0, recent_step_times=history, timeout_override=None)

    assert watchdog.timeout == expected
    assert watchdog.remaining(now=11.0) == expected - 1.0


def test_admission_watchdog_resets_only_when_admission_progresses():
    watchdog = AdmissionProgressWatchdog.start(now=10.0, recent_step_times=[], timeout_override=20.0)

    watchdog.observe(now=15.0, progressed=False)
    assert watchdog.elapsed(now=18.0) == 8.0

    watchdog.observe(now=18.0, progressed=True)
    assert watchdog.elapsed(now=20.0) == 2.0


@pytest.mark.parametrize(
    ("rejections", "expected"),
    [
        ((), AdmissionAction.ACCEPT),
        ((AdmissionRejection.STALE,), AdmissionAction.RETRY_PROMPT),
        ((AdmissionRejection.FULLY_MASKED,), AdmissionAction.REPLACE_PROMPT),
        ((AdmissionRejection.BELOW_MINIMUM_GROUP_SIZE,), AdmissionAction.REPLACE_PROMPT),
        ((AdmissionRejection.PHYSICAL_GROUP_SIZE,), AdmissionAction.FAIL),
        ((AdmissionRejection.MISSING_ROLLOUT_LOGPROBS,), AdmissionAction.FAIL),
    ],
)
def test_admission_decision_has_shared_queue_action(rejections, expected):
    assert AdmissionDecision(rejections).action is expected


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
    policy = GroupSelectionPolicy.for_fully_async("filter", criteria=resolve_dynamic_sampling_criteria("unshaped"))
    group = _group(loss_masks=[[1], [1], [1], [1]])
    group.trajectory_batch.update(
        {
            "rewards": [0.0, 0.0, 0.0, -0.1],
            "unshaped_rewards": [0.0, 1.0, 0.0, 1.0],
            "is_last_step": [False, True, False, True],
        }
    )

    decision = policy.evaluate(group)

    assert decision is GroupSelectionResult.INSUFFICIENT_REWARD_SPREAD


def test_dynamic_filter_requires_unshaped_outcomes():
    policy = GroupSelectionPolicy.for_fully_async("filter", criteria=resolve_dynamic_sampling_criteria("unshaped"))

    with pytest.raises(ValueError, match="requires unshaped_rewards"):
        policy.evaluate(_group(loss_masks=[[1], [1]]))


def test_dynamic_filter_can_admit_shaped_reward_variance():
    group = _group(loss_masks=[[1], [1], [1], [1]])
    group.trajectory_batch.update(
        {
            "rewards": [0.0, 0.25, 0.5, 0.0],
            "unshaped_rewards": [0.0, 0.0, 0.0, 0.0],
        }
    )

    shaped = GroupSelectionPolicy.for_fully_async("filter", criteria=resolve_dynamic_sampling_criteria("shaped"))
    unshaped = GroupSelectionPolicy.for_fully_async("filter", criteria=resolve_dynamic_sampling_criteria("unshaped"))

    assert shaped.evaluate(group) is GroupSelectionResult.KEEP
    assert unshaped.evaluate(group) is GroupSelectionResult.INSUFFICIENT_REWARD_SPREAD


def test_unshaped_dynamic_filter_ignores_passthrough_optimization_penalty():
    group = _group(loss_masks=[[1], [1]])
    group.trajectory_batch.update(
        {
            "rewards": [1.0, 1.0],
            "unshaped_rewards": [1.0, 1.0],
            "stop_reasons": ["turn_cap_exhausted", "complete"],
            "exception_types": ["TurnCapExhaustedError", None],
            "error_treatments": ["passthrough", None],
        }
    )
    shape_trajectory_rewards(
        group.trajectory_batch,
        {"enabled": True, "passthrough": {"penalty": 0.25}},
    )

    shaped = GroupSelectionPolicy.for_fully_async("filter", criteria=resolve_dynamic_sampling_criteria("shaped"))
    unshaped = GroupSelectionPolicy.for_fully_async("filter", criteria=resolve_dynamic_sampling_criteria("unshaped"))

    assert group.trajectory_batch["rewards"] == pytest.approx([0.75, 1.0])
    assert shaped.evaluate(group) is GroupSelectionResult.KEEP
    assert unshaped.evaluate(group) is GroupSelectionResult.INSUFFICIENT_REWARD_SPREAD


def test_dynamic_filter_applies_minimum_reward_std():
    group = _group(loss_masks=[[1], [1], [1], [1]])
    group.trajectory_batch["rewards"] = [0.05, 0.05, 0.10, 0.05]

    default_floor = GroupSelectionPolicy.for_fully_async("filter")
    high_floor = GroupSelectionPolicy.for_fully_async(
        "filter", criteria=resolve_dynamic_sampling_criteria("shaped", 0.1)
    )

    assert default_floor.evaluate(group) is GroupSelectionResult.KEEP
    assert high_floor.evaluate(group) is GroupSelectionResult.INSUFFICIENT_REWARD_SPREAD


def test_fully_async_selection_rejects_replace_sampling():
    with pytest.raises(ValueError, match="supports dynamic_sampling.type=filter or null"):
        GroupSelectionPolicy.for_fully_async("replace")
