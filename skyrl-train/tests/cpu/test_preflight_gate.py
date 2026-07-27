"""CPU tests for the pre-flight reward gate.

The gate checks mean per-sample reward against a band [min, max] and refuses
to start when the reward distribution leaves the range where RLOO has usable
within-group variance.

The gate logic lives in ``skyrl_train.utils.preflight_gate.check_preflight_gate``
and is a pure function — no trainer, no Ray, no GPU.
"""

from unittest.mock import MagicMock

import pytest

from skyrl_train.utils.preflight_gate import check_preflight_gate


class TestSparseRegime:
    """All-zero or near-zero rewards -> gate FAILS (sparse)."""

    def test_all_zero_rewards_fails(self):
        result = check_preflight_gate([0.0] * 256)
        assert not result.passed
        assert result.reason == "sparse"
        assert result.mean_reward == 0.0
        assert result.non_zero_fraction == 0.0

    def test_near_zero_rewards_fails(self):
        # 10% non-zero with tiny rewards
        rewards = [0.1] * 26 + [0.0] * 230
        result = check_preflight_gate(rewards)
        assert not result.passed
        assert result.reason == "sparse"
        assert result.mean_reward < 0.25

    def test_sparse_bound_reported(self):
        result = check_preflight_gate([0.0] * 256, min_reward=0.25)
        assert result.bound is not None
        assert result.bound[1] == 0.25  # the threshold


class TestSaturatedRegime:
    """All-one rewards -> gate FAILS (saturated)."""

    def test_all_one_rewards_fails(self):
        result = check_preflight_gate([1.0] * 256)
        assert not result.passed
        assert result.reason == "saturated"
        assert result.mean_reward == 1.0
        assert result.non_zero_fraction == 1.0

    def test_near_one_rewards_fails(self):
        rewards = [1.0] * 220 + [0.0] * 36  # mean ~0.86
        result = check_preflight_gate(rewards, max_reward=0.75)
        assert not result.passed
        assert result.reason == "saturated"

    def test_saturated_bound_reported(self):
        result = check_preflight_gate([1.0] * 256, max_reward=0.75)
        assert result.bound is not None
        assert result.bound[1] == 0.75


class TestPassingBand:
    """Rewards near 0.5 -> gate PASSES."""

    def test_rewards_near_half_pass(self):
        rewards = [1.0] * 128 + [0.0] * 128  # mean = 0.5
        result = check_preflight_gate(rewards)
        assert result.passed
        assert abs(result.mean_reward - 0.5) < 0.01
        assert abs(result.non_zero_fraction - 0.5) < 0.01

    def test_healthy_arm_passes(self):
        """The arm that measured 0.643 should pass."""
        rewards = [1.0] * 165 + [0.0] * 92  # mean ~0.642
        result = check_preflight_gate(rewards)
        assert result.passed

    def test_high_but_healthy_passes(self):
        """An arm at pass@8=1.0 with per-sample reward 0.55 should pass."""
        rewards = [1.0] * 141 + [0.0] * 115  # mean ~0.55
        result = check_preflight_gate(rewards)
        assert result.passed

    def test_boundary_inclusive(self):
        """Exactly min_reward should pass (>=)."""
        rewards = [0.25] * 256
        result = check_preflight_gate(rewards)
        assert result.passed


def test_gate_disabled_always_passes():
    """When enabled=false the code path is untouched. Verified by the callback
    not being appended in create_default_callbacks. Here we verify the gate
    function itself is vacuously True on empty input."""
    result = check_preflight_gate([])
    assert result.passed
    assert result.num_samples == 0


def test_custom_band():
    """A custom band [0.1, 0.9] should pass what the default fails."""
    rewards = [1.0] * 220 + [0.0] * 36  # mean ~0.86, fails default max 0.75
    result_default = check_preflight_gate(rewards)
    assert not result_default.passed

    result_custom = check_preflight_gate(rewards, min_reward=0.1, max_reward=0.9)
    assert result_custom.passed


class TestCallbackIntegration:
    """Verify the callback reads config and fires correctly."""

    def test_callback_disabled_by_default(self):
        from skyrl_train.callbacks.builtin import PreflightGateCallback

        cb = PreflightGateCallback()
        assert cb.enabled is False
        assert cb._checked is False

    def test_callback_aborts_on_sparse(self):
        from skyrl_train.callbacks.builtin import PreflightGateCallback, PreflightGateError

        cb = PreflightGateCallback(enabled=True, min_reward=0.25, max_reward=0.75, on_failure="abort")
        trainer = MagicMock()
        trainer._current_step_rewards = [0.0] * 256

        from skyrl_train.callbacks.base import TrainerState, TrainerControl

        state = TrainerState(global_step=1, epoch=0, total_steps=80, num_steps_per_epoch=80)
        control = TrainerControl()

        with pytest.raises(PreflightGateError, match="sparse"):
            cb.on_step_end(state, control, trainer=trainer)

    def test_callback_warns_on_sparse(self):
        from skyrl_train.callbacks.builtin import PreflightGateCallback

        cb = PreflightGateCallback(enabled=True, min_reward=0.25, max_reward=0.75, on_failure="warn")
        trainer = MagicMock()
        trainer._current_step_rewards = [0.0] * 256

        from skyrl_train.callbacks.base import TrainerState, TrainerControl

        state = TrainerState(global_step=1, epoch=0, total_steps=80, num_steps_per_epoch=80)
        control = TrainerControl()

        # Should NOT raise
        result = cb.on_step_end(state, control, trainer=trainer)
        assert cb._checked is True

    def test_callback_passes_on_healthy(self):
        from skyrl_train.callbacks.builtin import PreflightGateCallback

        cb = PreflightGateCallback(enabled=True, min_reward=0.25, max_reward=0.75)
        trainer = MagicMock()
        trainer._current_step_rewards = [1.0] * 128 + [0.0] * 128

        from skyrl_train.callbacks.base import TrainerState, TrainerControl

        state = TrainerState(global_step=1, epoch=0, total_steps=80, num_steps_per_epoch=80)
        control = TrainerControl()

        result = cb.on_step_end(state, control, trainer=trainer)
        assert cb._checked is True

    def test_callback_checks_only_once(self):
        from skyrl_train.callbacks.builtin import PreflightGateCallback

        cb = PreflightGateCallback(enabled=True)
        trainer = MagicMock()
        trainer._current_step_rewards = [0.5] * 256

        from skyrl_train.callbacks.base import TrainerState, TrainerControl

        state = TrainerState(global_step=1, epoch=0, total_steps=80, num_steps_per_epoch=80)
        control = TrainerControl()

        cb.on_step_end(state, control, trainer=trainer)
        assert cb._checked is True
        # Second call should be a no-op even if rewards change
        trainer._current_step_rewards = [0.0] * 256
        cb.on_step_end(state, control, trainer=trainer)  # should not raise
