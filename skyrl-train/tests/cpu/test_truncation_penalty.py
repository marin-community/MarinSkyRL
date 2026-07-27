"""CPU tests for the truncation penalty.

The penalty keys on PER-TURN truncation: a single generation that hit
``max_generate_length``, not the trajectory-level ``stop_reason`` which is
almost never reached on real workloads.
"""

import os
import sys

import pytest

_EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "..", "examples")
if _EXAMPLES not in sys.path:
    sys.path.insert(0, _EXAMPLES)

from terminal_bench.truncation_penalty import apply_truncation_penalty, detect_turn_truncation  # noqa: E402

try:
    from terminal_bench.harbor_config import REWARD_SHAPING_SCHEMA  # noqa: E402
except ImportError:
    REWARD_SHAPING_SCHEMA = None


class TestDetectTurnTruncation:
    def test_detects_turn_at_cap(self):
        assert detect_turn_truncation([1000, 4096, 500], 4096) is True

    def test_no_truncation_when_all_below(self):
        assert detect_turn_truncation([1000, 2000, 500], 4096) is False

    def test_no_truncation_when_empty(self):
        assert detect_turn_truncation([], 4096) is False

    def test_no_truncation_when_none(self):
        assert detect_turn_truncation(None, 4096) is False

    def test_no_truncation_when_max_is_zero(self):
        assert detect_turn_truncation([4096], 0) is False

    def test_single_turn_at_cap(self):
        assert detect_turn_truncation([4096], 4096) is True


class TestApplyTruncationPenalty:
    def test_penalizes_truncated_zero_reward(self):
        reward, penalized = apply_truncation_penalty(
            reward=0.0, original_reward=0.0, turn_truncated=True, truncation_penalty=0.5
        )
        assert penalized is True
        assert reward == -0.5

    def test_noop_when_penalty_is_zero(self):
        reward, penalized = apply_truncation_penalty(
            reward=0.0, original_reward=0.0, turn_truncated=True, truncation_penalty=0.0
        )
        assert penalized is False
        assert reward == 0.0

    def test_noop_when_not_turn_truncated(self):
        reward, penalized = apply_truncation_penalty(
            reward=0.8, original_reward=0.8, turn_truncated=False, truncation_penalty=0.5
        )
        assert penalized is False
        assert reward == 0.8

    def test_does_not_touch_successful_truncated_trial(self):
        reward, penalized = apply_truncation_penalty(
            reward=1.0, original_reward=1.0, turn_truncated=True, truncation_penalty=0.5
        )
        assert penalized is False
        assert reward == 1.0

    def test_penalizes_even_when_shaped_reward_is_nonzero(self):
        reward, penalized = apply_truncation_penalty(
            reward=0.3, original_reward=0.0, turn_truncated=True, truncation_penalty=0.5
        )
        assert penalized is True
        assert reward == pytest.approx(-0.2)

    def test_preserves_partial_credit_ordering(self):
        better, _ = apply_truncation_penalty(
            reward=0.8, original_reward=0.0, turn_truncated=True, truncation_penalty=0.25
        )
        worse, _ = apply_truncation_penalty(
            reward=0.1, original_reward=0.0, turn_truncated=True, truncation_penalty=0.25
        )
        assert better > worse

    def test_truncated_ranks_below_untruncated_at_equal_shaped_reward(self):
        truncated, _ = apply_truncation_penalty(
            reward=0.4, original_reward=0.0, turn_truncated=True, truncation_penalty=0.25
        )
        finished, _ = apply_truncation_penalty(
            reward=0.4, original_reward=0.0, turn_truncated=False, truncation_penalty=0.25
        )
        assert truncated < finished


@pytest.mark.skipif(REWARD_SHAPING_SCHEMA is None, reason="harbor deps unavailable")
class TestTruncationPenaltyConfig:
    def test_default_is_zero(self):
        field = REWARD_SHAPING_SCHEMA.fields["truncation_penalty"]
        assert field.default == 0.0

    def test_field_exists(self):
        assert "truncation_penalty" in REWARD_SHAPING_SCHEMA.fields
