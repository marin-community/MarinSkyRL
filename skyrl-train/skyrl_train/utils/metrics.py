"""Scalar metric aggregation."""

from collections.abc import Mapping, Sequence


def mean_metrics(metrics: Mapping[str, Sequence[float]]) -> dict[str, float]:
    """Average each scalar metric sequence, rejecting empty or nonnumeric values."""
    reduced = {}
    for name, values in metrics.items():
        if not values:
            raise ValueError(f"No values for metric {name}")
        if not all(isinstance(value, (int, float)) for value in values):
            raise TypeError(f"Metric {name} contains a non-numeric value")
        reduced[name] = sum(values) / len(values)
    return reduced


def policy_training_metrics(
    metrics: Mapping[str, Sequence[float]],
    policy_update_steps: float,
) -> dict[str, float]:
    """Average policy metrics, omit response length, and add update count."""
    status = mean_metrics({name: values for name, values in metrics.items() if name != "response_length"})
    status["policy_update_steps"] = policy_update_steps
    return status


def policy_progress_metrics(status: Mapping[str, float]) -> dict[str, float]:
    """Select the compact policy progress fields displayed during training."""
    progress = {
        "pg": status["policy_loss"],
        "glen": status["response_length"],
        "policy_lr": status["policy_lr"],
        "ent": status["policy_entropy"],
    }
    if "raw_grad_norm" in status:
        progress["grad_norm"] = status["raw_grad_norm"]
    return progress
