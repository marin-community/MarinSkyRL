import pytest

from skyrl_train.trajectory_runners.harbor.identity_aware_reward import (
    IdentityAwareFallback,
    identity_aware_pass_ratios,
)
from skyrl_train.trajectory_runners.types import TrajectoryID, VerifierTestCollection


def _collection(trial: int, outcomes: dict[str, str], *, complete: bool = True) -> VerifierTestCollection:
    return {
        "parser": "test",
        "complete": complete,
        "tests": [
            {
                "record_id": f"trial-{trial}:{test_id}",
                "trial_id": TrajectoryID(instance_id="task", repetition_id=trial),
                "test_id": test_id,
                "outcome": outcome,
                "output": f"{test_id}: {outcome}",
            }
            for test_id, outcome in outcomes.items()
        ],
    }


def test_uniform_tests_are_ignored_before_pass_ratio_is_computed():
    collections = [
        _collection(0, {"always-pass": "passed", "mixed": "passed", "always-fail": "failed"}),
        _collection(1, {"always-pass": "passed", "mixed": "failed", "always-fail": "failed"}),
    ]

    result = identity_aware_pass_ratios(collections, [2 / 3, 1 / 3], [True, True])

    assert result.rewards == (1.0, 0.0)
    assert result.informative_test_count == 1
    assert result.fallback_reason is None


def test_group_with_no_non_uniform_tests_has_zero_reward_signal():
    collections = [
        _collection(0, {"pass": "passed", "fail": "failed"}),
        _collection(1, {"pass": "passed", "fail": "failed"}),
    ]

    result = identity_aware_pass_ratios(collections, [0.5, 0.5], [True, True])

    assert result.rewards == (0.0, 0.0)
    assert result.informative_test_count == 0


@pytest.mark.parametrize(
    ("collections", "reason"),
    [
        ([None, None], IdentityAwareFallback.INCOMPLETE_TESTS),
        (
            [_collection(0, {"a": "passed"}, complete=False), _collection(1, {"a": "failed"})],
            IdentityAwareFallback.INCOMPLETE_TESTS,
        ),
        (
            [_collection(0, {"a": "passed"}), _collection(1, {"b": "failed"})],
            IdentityAwareFallback.MISMATCHED_TESTS,
        ),
    ],
)
def test_unreliable_identity_falls_back_to_aggregate_reward(collections, reason):
    result = identity_aware_pass_ratios(collections, [0.75, 0.25], [True, True])

    assert result.rewards == (0.75, 0.25)
    assert result.fallback_reason is reason


def test_ineligible_trials_do_not_determine_whether_a_test_is_uniform():
    collections = [
        _collection(0, {"a": "passed"}),
        _collection(1, {"a": "passed"}),
        None,
    ]

    result = identity_aware_pass_ratios(collections, [1.0, 1.0, 0.0], [True, True, False])

    assert result.rewards == (0.0, 0.0, 0.0)
    assert result.fallback_reason is None


def test_exact_record_weights_and_post_hoc_filter_are_composable():
    collections = [
        _collection(0, {"a": "passed", "b": "failed"}),
        _collection(1, {"a": "failed", "b": "passed"}),
    ]

    weighted = identity_aware_pass_ratios(
        collections,
        [0.5, 0.5],
        [True, True],
        exact_weights={"trial-0:a": 3.0},
    )
    filtered = identity_aware_pass_ratios(
        collections,
        [0.5, 0.5],
        [True, True],
        weight_filter=lambda record: record["test_id"] == "b",
    )

    assert weighted.rewards == (0.75, 0.5)
    assert filtered.rewards == (0.0, 1.0)


def test_invalid_weight_is_rejected():
    collections = [_collection(0, {"a": "passed"}), _collection(1, {"a": "failed"})]

    with pytest.raises(ValueError, match="finite non-negative"):
        identity_aware_pass_ratios(
            collections,
            [1.0, 0.0],
            [True, True],
            exact_weights={"trial-0:a": -1.0},
        )
