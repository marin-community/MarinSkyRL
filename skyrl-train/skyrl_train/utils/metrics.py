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

    The learning rate is the third case: neither a mean nor a specially-reduced key, but the LATEST
    value, because a schedule that moved during the step is described by where it ended rather than
    by the average of where it was.
    """
    scalars = {name: values for name, values in metrics.items() if name != "response_length"}
    status = mean_metrics({name: v for name, v in scalars.items() if name not in STATUS_REDUCTION_OPS})
    # Explicit dispatch. The earlier form was `max(values) if op == "max" else sum(values)`, which
    # silently SUMMED any op it did not know -- so adding a new op to the map would have published a
    # sum of flags and looked like a number. An unknown op must fail loudly instead.
    reducers = {"max": max, "min": min, "sum": sum}
    for name, values in scalars.items():
        op = STATUS_REDUCTION_OPS.get(name)
        if op is None:
            continue
        if not values:
            raise ValueError(f"No values for metric {name}")
        # mean_metrics rejects non-numeric values; this path must too, or a 0-d tensor is published
        # where a float is expected on exactly the keys designated gate-grade.
        if not all(isinstance(value, (int, float)) for value in values):
            raise TypeError(f"Metric {name} contains a non-numeric value")
        if op not in reducers:
            raise ValueError(f"Metric {name} has unknown reduction op {op!r}")
        status[name] = reducers[op](values)
    if learning_rates := metrics.get("policy_lr"):
        status["policy_lr"] = learning_rates[-1]
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
