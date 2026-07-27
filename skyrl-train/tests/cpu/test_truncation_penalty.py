"""CPU tests for the truncation penalty.

Verifies that a cap-truncated trial scoring zero is penalized below the zero
floor (so RLOO assigns it a negative advantage, distinguishing it from an
honest wrong answer), and that the default-off config is byte-identical.

The penalty logic lives in ``apply_truncation_penalty`` (a standalone function)
and the config knob ``truncation_penalty`` defaults to 0.0 (no-op).
"""

import os
import sys

import pytest

_EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "..", "examples")
if _EXAMPLES not in sys.path:
    sys.path.insert(0, _EXAMPLES)

from terminal_bench.truncation_penalty import apply_truncation_penalty  # noqa: E402

try:
    from terminal_bench.harbor_config import REWARD_SHAPING_SCHEMA  # noqa: E402
except ImportError:
    REWARD_SHAPING_SCHEMA = None


class TestApplyTruncationPenalty:
    def test_penalizes_truncated_zero_reward(self):
        reward, penalized = apply_truncation_penalty(
            reward=0.0, original_reward=0.0, stop_reason="length", truncation_penalty=0.5
        )
        assert penalized is True
        assert reward == -0.5

    def test_noop_when_penalty_is_zero(self):
        """Default-off: byte-identical when truncation_penalty=0.0."""
        reward, penalized = apply_truncation_penalty(
            reward=0.0, original_reward=0.0, stop_reason="length", truncation_penalty=0.0
        )
        assert penalized is False
        assert reward == 0.0

    def test_noop_when_stop_reason_complete(self):
        reward, penalized = apply_truncation_penalty(
            reward=0.8, original_reward=0.8, stop_reason="complete", truncation_penalty=0.5
        )
        assert penalized is False
        assert reward == 0.8

    def test_noop_when_stop_reason_error(self):
        reward, penalized = apply_truncation_penalty(
            reward=0.0, original_reward=0.0, stop_reason="error", truncation_penalty=0.5
        )
        assert penalized is False
        assert reward == 0.0

    def test_does_not_touch_successful_truncated_trial(self):
        """A trial that truncated but still passed its verifier is left untouched."""
        reward, penalized = apply_truncation_penalty(
            reward=1.0, original_reward=1.0, stop_reason="length", truncation_penalty=0.5
        )
        assert penalized is False
        assert reward == 1.0

    def test_penalizes_even_when_shaped_reward_is_nonzero(self):
        """The penalty fires on original_reward==0 (verifier failure), even if
        the shaper produced a nonzero reward from partial credit."""
        reward, penalized = apply_truncation_penalty(
            reward=0.3, original_reward=0.0, stop_reason="length", truncation_penalty=0.5
        )
        assert penalized is True
        assert reward == -0.5


@pytest.mark.skipif(REWARD_SHAPING_SCHEMA is None, reason="harbor deps unavailable")
class TestTruncationPenaltyConfig:
    def test_default_is_zero(self):
        """truncation_penalty must default to 0.0 (byte-identical when off)."""
        field = REWARD_SHAPING_SCHEMA.fields["truncation_penalty"]
        assert field.default == 0.0

    def test_field_exists(self):
        assert "truncation_penalty" in REWARD_SHAPING_SCHEMA.fields
