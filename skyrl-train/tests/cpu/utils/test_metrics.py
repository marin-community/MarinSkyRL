import pytest

from skyrl_train.utils.metrics import policy_training_metrics


def test_policy_training_metrics_reduces_each_key_over_optimizer_windows() -> None:
    metrics = {
        "policy_loss": [1.0, 3.0],
        "policy_lr": [0.0, 4e-6],
        "response_length": [128, 256],
        "log_ratio_abs_max": [19.0, 0.0],
        "log_ratio_diagnostics_failed": [0.0, 1.0],
        "n_tokens_dp_gt_1pct": [3.0, 7.0],
    }

    result = policy_training_metrics(metrics, policy_update_steps=2.0)

    assert result == {
        "policy_loss": 2.0,
        "policy_lr": pytest.approx(4e-6),
        "log_ratio_abs_max": 19.0,
        "log_ratio_diagnostics_failed": 1.0,
        "n_tokens_dp_gt_1pct": 10.0,
        "policy_update_steps": 2.0,
    }
