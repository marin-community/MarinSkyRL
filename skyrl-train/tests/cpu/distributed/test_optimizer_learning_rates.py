import pytest
import torch

from skyrl_train.distributed.optimizer_learning_rates import validate_optimizer_learning_rates


def _optimizer_with_learning_rates(*learning_rates: float) -> torch.optim.Optimizer:
    groups = [
        {"params": [torch.nn.Parameter(torch.zeros(1))], "lr": learning_rate}
        for learning_rate in learning_rates
    ]
    return torch.optim.SGD(groups, lr=learning_rates[0])


def test_optimizer_learning_rates_reject_hidden_group_default():
    optimizer = _optimizer_with_learning_rates(1e-5, 6e-4)

    with pytest.raises(ValueError, match="undeclared learning rates"):
        validate_optimizer_learning_rates(
            optimizer,
            master_learning_rate=1e-5,
            optimizer_kwargs={},
        )


def test_optimizer_learning_rates_reject_duplicate_master_setting():
    optimizer = _optimizer_with_learning_rates(6e-4)

    with pytest.raises(ValueError, match="optimizer_config.lr"):
        validate_optimizer_learning_rates(
            optimizer,
            master_learning_rate=1e-5,
            optimizer_kwargs={"lr": 6e-4},
        )
