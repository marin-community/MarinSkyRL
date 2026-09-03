import pytest

from skyrl_train.utils.metrics import policy_training_metrics


def test_policy_training_metrics_preserves_latest_learning_rate() -> None:
    metrics = {
        "policy_loss": [1.0, 3.0],
        "policy_lr": [0.0, 4e-6],
        "response_length": [128, 256],
    }

    result = policy_training_metrics(metrics, policy_update_steps=1.0)

    assert result == {
        "policy_loss": 2.0,
        "policy_lr": pytest.approx(4e-6),
        "policy_update_steps": 1.0,
    }
