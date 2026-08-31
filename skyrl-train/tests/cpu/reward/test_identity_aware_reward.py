import pytest

from skyrl_train.trajectory_runners.harbor.identity_aware_reward import (
    IdentityAwareFallback,
    identity_aware_pass_ratios,
)


def test_uniform_tests_are_ignored_before_pass_ratio_is_computed(verifier_test_collection_factory):
    collections = [
        verifier_test_collection_factory(0, {"always-pass": "passed", "mixed": "passed", "always-fail": "failed"}),
        verifier_test_collection_factory(1, {"always-pass": "passed", "mixed": "failed", "always-fail": "failed"}),
    ]

    result = identity_aware_pass_ratios(collections, [2 / 3, 1 / 3], [True, True])

    assert result.rewards == (1.0, 0.0)
    assert result.informative_test_count == 1
    assert result.fallback_reason is None


def test_group_with_no_non_uniform_tests_has_zero_reward_signal(verifier_test_collection_factory):
    collections = [
        verifier_test_collection_factory(0, {"pass": "passed", "fail": "failed"}),
        verifier_test_collection_factory(1, {"pass": "passed", "fail": "failed"}),
    ]

    result = identity_aware_pass_ratios(collections, [0.5, 0.5], [True, True])

    assert result.rewards == (0.0, 0.0)
    assert result.informative_test_count == 0


def test_missing_identity_falls_back_to_aggregate_reward():
    result = identity_aware_pass_ratios([None, None], [0.75, 0.25], [True, True])

    assert result.rewards == (0.75, 0.25)
    assert result.fallback_reason is IdentityAwareFallback.INCOMPLETE_TESTS


def test_incomplete_identity_falls_back_to_aggregate_reward(verifier_test_collection_factory):
    collections = [
        verifier_test_collection_factory(0, {"a": "passed"}, complete=False),
        verifier_test_collection_factory(1, {"a": "failed"}),
    ]

    result = identity_aware_pass_ratios(collections, [0.75, 0.25], [True, True])

    assert result.rewards == (0.75, 0.25)
    assert result.fallback_reason is IdentityAwareFallback.INCOMPLETE_TESTS


def test_mismatched_identity_falls_back_to_aggregate_reward(verifier_test_collection_factory):
    collections = [
        verifier_test_collection_factory(0, {"a": "passed"}),
        verifier_test_collection_factory(1, {"b": "failed"}),
    ]

    result = identity_aware_pass_ratios(collections, [0.75, 0.25], [True, True])

    assert result.rewards == (0.75, 0.25)
    assert result.fallback_reason is IdentityAwareFallback.MISMATCHED_TESTS


def test_ineligible_trials_do_not_determine_whether_a_test_is_uniform(verifier_test_collection_factory):
    collections = [
        verifier_test_collection_factory(0, {"a": "passed"}),
        verifier_test_collection_factory(1, {"a": "passed"}),
        None,
    ]

    result = identity_aware_pass_ratios(collections, [1.0, 1.0, 0.0], [True, True, False])

    assert result.rewards == (0.0, 0.0, 0.0)
    assert result.fallback_reason is None


def test_exact_record_weights_and_post_hoc_filter_are_composable(verifier_test_collection_factory):
    collections = [
        verifier_test_collection_factory(0, {"a": "passed", "b": "failed"}),
        verifier_test_collection_factory(1, {"a": "failed", "b": "passed"}),
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
        weight_filter=lambda record: 1.0 if record["test_id"] == "b" else 0.0,
    )

    assert weighted.rewards == (0.75, 0.5)
    assert filtered.rewards == (0.0, 1.0)


def test_invalid_weight_is_rejected(verifier_test_collection_factory):
    collections = [
        verifier_test_collection_factory(0, {"a": "passed"}),
        verifier_test_collection_factory(1, {"a": "failed"}),
    ]

    with pytest.raises(ValueError, match="finite non-negative"):
        identity_aware_pass_ratios(
            collections,
            [1.0, 0.0],
            [True, True],
            exact_weights={"trial-0:a": -1.0},
        )
