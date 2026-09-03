"""Scalar metric aggregation."""

from collections.abc import Mapping, Sequence

from skyrl_train.utils.importance_ratio_diagnostics import STATUS_REDUCTION_OPS


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
    """Combine a step's mini-batch metrics, omit response length, and add update count.

    This is the SECOND reduction axis. Reducing correctly across ranks and then averaging the result
    back down over mini-batches is the same category error one level lower: at n mini-batches per
    step a max becomes mean-of-n-maxima, and a "did any rank fail" flag becomes 1/n. The op map is
    shared with Strategy.all_reduce_status so the two axes cannot drift apart.
    """
    scalars = {name: values for name, values in metrics.items() if name != "response_length"}
    status = mean_metrics({name: v for name, v in scalars.items() if name not in STATUS_REDUCTION_OPS})
    for name, values in scalars.items():
        op = STATUS_REDUCTION_OPS.get(name)
        if op is None:
            continue
        if not values:
            raise ValueError(f"No values for metric {name}")
        status[name] = max(values) if op == "max" else sum(values)
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
