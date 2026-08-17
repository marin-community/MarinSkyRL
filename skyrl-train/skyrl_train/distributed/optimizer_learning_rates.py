from collections.abc import Mapping

from torch.optim import Optimizer


def validate_optimizer_learning_rates(
    optimizer: Optimizer,
    *,
    master_learning_rate: float,
    optimizer_kwargs: Mapping[str, object] | None,
) -> None:
    """Require every optimizer group rate to come from explicit run configuration."""
    overrides = {} if optimizer_kwargs is None else dict(optimizer_kwargs)
    if "lr" in overrides:
        raise ValueError("Set the master learning rate with optimizer_config.lr, not optimizer_kwargs.lr")

    configured_rates = {float(master_learning_rate)}
    configured_rates.update(float(value) for name, value in overrides.items() if name.endswith("_lr"))
    undeclared_groups = [
        (index, float(group["lr"]))
        for index, group in enumerate(optimizer.param_groups)
        if float(group["lr"]) not in configured_rates
    ]
    if undeclared_groups:
        raise ValueError(
            f"Optimizer parameter groups use undeclared learning rates {undeclared_groups}; "
            f"configured rates are {sorted(configured_rates)}"
        )
