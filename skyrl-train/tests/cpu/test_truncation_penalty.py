"""Truncation penalty keyed on per-turn truncation (regression #166).

A single generation that hit ``max_generate_length`` marks the trial as truncated; the
trajectory-level ``stop_reason`` almost never reaches the cap on real workloads.
"""

import pytest

from skyrl_train.trajectory_runners.harbor.truncation_penalty import apply_truncation_penalty, detect_turn_truncation


@pytest.mark.parametrize(
    ("turn_lengths", "max_generate_length", "truncated"),
    [
        ([1000, 4096, 500], 4096, True),
        ([4096], 4096, True),
        ([1000, 2000, 500], 4096, False),
        ([], 4096, False),
        (None, 4096, False),
        ([4096], 0, False),
    ],
)
def test_detect_turn_truncation(turn_lengths, max_generate_length, truncated):
    assert detect_turn_truncation(turn_lengths, max_generate_length) is truncated


@pytest.mark.parametrize(
    ("reward", "original_reward", "turn_truncated", "truncation_penalty", "expected"),
    [
        pytest.param(0.0, 0.0, True, 0.5, (-0.5, True), id="truncated-failure"),
        # Shaped partial credit is kept, so truncated trials still rank by partial credit.
        pytest.param(0.3, 0.0, True, 0.5, (pytest.approx(-0.2), True), id="truncated-failure-with-shaping"),
        pytest.param(0.0, 0.0, True, 0.0, (0.0, False), id="penalty-disabled"),
        pytest.param(0.8, 0.8, False, 0.5, (0.8, False), id="not-truncated"),
        pytest.param(1.0, 1.0, True, 0.5, (1.0, False), id="truncated-success"),
    ],
)
def test_apply_truncation_penalty_only_penalizes_truncated_failures(
    reward, original_reward, turn_truncated, truncation_penalty, expected
):
    assert (
        apply_truncation_penalty(
            reward=reward,
            original_reward=original_reward,
            turn_truncated=turn_truncated,
            truncation_penalty=truncation_penalty,
        )
        == expected
    )
