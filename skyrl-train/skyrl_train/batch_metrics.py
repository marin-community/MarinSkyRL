from collections.abc import Sequence
from typing import Literal

import torch
from torch import Tensor

from skyrl_train.metric_reduction import MetricReduction
from skyrl_train.telemetry import ConsumedWork
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.utils.advantage_estimators import GRPO_FLAT_REWARD_STD_TOLERANCE
from skyrl_train.utils.importance_ratio_diagnostics import exact_ratio_statistics


class LocalReduction:
    """Preserve local tensor arithmetic for driver diagnostics."""

    def mean(self, values: Tensor) -> float:
        return values.mean().item()

    def std(self, values: Tensor) -> float:
        return values.std().item()

    def sum(self, values: Tensor) -> Tensor:
        return values.sum()

    def combine(self, values: Tensor, operation: Literal["sum", "max"]) -> Tensor:
        return values

    def ratios(self, delta: Tensor, *, eps_clip_low: float, eps_clip_high: float) -> dict[str, float]:
        return exact_ratio_statistics(delta, eps_clip_low=eps_clip_low, eps_clip_high=eps_clip_high)


def consumed_work(batch: TrainingInputBatch) -> ConsumedWork:
    """Count finalized training rows and tokens excluding DP padding."""
    rows = batch.batch_size - batch.metadata.get("pad_size", 0)
    return ConsumedWork(
        sequences=rows,
        response_tokens=int(batch["response_mask"][:rows].sum().item()),
        loss_tokens=int(batch["loss_mask"][:rows].sum().item()),
    )


def zero_std_group_fraction(uids: Sequence[str], rewards: Tensor) -> float:
    """Count complete groups whose sample reward deviation is below the GRPO threshold."""
    groups: dict[str, list[Tensor]] = {}
    for uid, reward in zip(uids, rewards, strict=True):
        groups.setdefault(uid, []).append(reward)
    if not groups:
        return 0.0
    flat = sum(
        len(group) > 1 and torch.std(torch.stack(group)).item() <= GRPO_FLAT_REWARD_STD_TOLERANCE
        for group in groups.values()
    )
    return flat / len(groups)


def advantage_metrics(batch: TrainingInputBatch, *, reduction: MetricReduction, step_wise: bool) -> dict[str, float]:
    """Compute reward and advantage diagnostics before loop credit is finalized."""
    rows = batch.batch_size - batch.metadata.get("pad_size", 0)
    rewards = batch["rewards"].sum(-1)[:rows]
    if step_wise:
        rewards = rewards[batch["is_last_step"][:rows]]
    advantages = torch.masked_select(batch["advantages"][:rows], batch["response_mask"][:rows].bool())
    average_reward = reduction.mean(rewards)
    average_advantage = reduction.mean(advantages)
    average_absolute = reduction.mean(advantages.abs())
    batch.metadata.setdefault("metrics", {}).update(
        avg_final_rewards=average_reward,
        avg_response_length=batch.metadata["avg_response_length"],
        avg_advantages=average_advantage,
        avg_advantages_abs=average_absolute,
    )
    return {
        "loss/avg_final_rewards": average_reward,
        "loss/avg_raw_advantages": average_advantage,
        "loss/avg_raw_advantages_abs": average_absolute,
    }


def probability_difference_metrics(batch: TrainingInputBatch, *, reduction: MetricReduction) -> dict[str, float]:
    """Summarize rollout-to-learner probability ratios on loss-bearing tokens."""
    valid = batch["loss_mask"] > 0
    differences = (batch["rollout_logprobs"][valid] - batch["action_log_probs"][valid]).exp().abs()
    return {
        "policy/rollout_train_prob_diff_mean": reduction.mean(differences),
        "policy/rollout_train_prob_diff_std": reduction.std(differences),
    }


def consumed_stop_metrics(stop_reasons: Sequence[str | None] | None, sequence_count: int) -> dict[str, float]:
    """Count length stops on admitted sequences; the fraction needs every stop reason."""
    reasons = [None] * sequence_count if stop_reasons is None else stop_reasons
    known = sum(reason is not None and reason != "" for reason in reasons)
    length_stops = sum(reason == "length" for reason in reasons)
    metrics = {
        "sequences": float(sequence_count),
        "length_stop_count": float(length_stops),
        "known_stop_count": float(known),
        "unknown_stop_count": float(sequence_count - known),
    }
    if sequence_count:
        metrics["stop_reason_coverage"] = known / sequence_count
        if known == sequence_count:
            metrics["length_stop_fraction"] = length_stops / sequence_count
    return {f"consumed/{name}": value for name, value in metrics.items()}
