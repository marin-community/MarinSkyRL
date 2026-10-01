"""Loop-boundary preferences and exact final-token preference loss.

The loss follows Liquid4All/antidoom reference_ftpo_trainer.py at
bd6a126476e18554b0cacaea3fd9f258fdde1f97 (Apache-2.0). It uses the native
suffix detector and per-batch frequency balancing; token IDs never round-trip
through decoded text.
"""

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from skyrl_train.config.ftpo import FTPOConfig
from skyrl_train.trajectory_runners.trajectory_reward_shaping import LoopCreditConfig, tail_loop


@dataclass(frozen=True)
class FTPOTargets:
    """Existing candidate IDs plus FTPO selection and frozen boundary logits."""

    candidate_ids: torch.Tensor
    chosen_mask: torch.Tensor
    reference_logits: torch.Tensor

    def to(self, device: torch.device) -> "FTPOTargets":
        return FTPOTargets(self.candidate_ids.to(device), self.chosen_mask.to(device), self.reference_logits.to(device))

    def pin_memory(self) -> "FTPOTargets":
        return FTPOTargets(
            self.candidate_ids.pin_memory(), self.chosen_mask.pin_memory(), self.reference_logits.pin_memory()
        )


@dataclass(frozen=True)
class FTPOCounts:
    examples: float
    targets: float
    other: float


@dataclass(frozen=True)
class FTPOInputs:
    logits: torch.Tensor
    targets: FTPOTargets
    rejected_ids: torch.Tensor
    counts: FTPOCounts


def boundary_values(values: torch.Tensor, chosen_mask: torch.Tensor) -> torch.Tensor:
    """Select the single FTPO position in each response; inactive rows select zero."""
    active = chosen_mask.any(-1)
    if torch.any(active.sum(-1) > 1):
        raise ValueError("FTPO supports at most one boundary per response")
    positions = active.long().argmax(-1)
    return values[torch.arange(values.shape[0], device=values.device), positions]


def compact_boundary_logits(
    logits: torch.Tensor,
    attention_mask: torch.Tensor,
    chosen_mask: torch.Tensor,
) -> torch.Tensor:
    """Gather prediction logits after Megatron removes masked sequence positions."""
    response_length = chosen_mask.shape[1]
    # Count real tokens through the predecessor of each response token. This
    # handles unequal prompts and response padding without scattering [B,S,V].
    compact_positions = attention_mask.long().cumsum(-1)[:, -response_length - 1 : -1] - 1
    positions = boundary_values(compact_positions, chosen_mask)
    positions = torch.where(chosen_mask.any((-1, -2)), positions, 0)
    return logits[torch.arange(logits.shape[0], device=logits.device), positions]


def select_ftpo_candidates(
    response_ids: Sequence[Sequence[int]],
    loss_masks: Sequence[Sequence[int]],
    candidate_ids: torch.Tensor,
    candidate_logprobs: torch.Tensor,
    *,
    decode: Callable[[list[int]], str],
    loop: LoopCreditConfig,
    config: FTPOConfig,
    seed: int,
    eligible: Sequence[bool],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select alternatives at each first repeat and balance within this rollout batch.

    Rejected frequency controls example weights. Chosen frequency caps prune
    alternatives using a seeded permutation. No cross-batch state is needed.
    """
    chosen = torch.zeros_like(candidate_ids, dtype=torch.bool)
    weights = torch.zeros(candidate_ids.shape[:2], dtype=torch.float32)
    surfaces: dict[int, str] = {}
    rejected_by_row = {}
    for row, (response, mask, allowed) in enumerate(zip(response_ids, loss_masks, eligible, strict=True)):
        hit = tail_loop(response, mask, loop) if allowed else None
        if hit is None:
            continue
        position = hit.repeat_start
        rejected = response[position]
        ids = candidate_ids[row, position].tolist()
        scores = candidate_logprobs[row, position]
        if any(token < 0 for token in ids) or len(set(ids)) != len(ids) or not torch.isfinite(scores).all():
            raise ValueError("FTPO boundary candidates require unique token IDs and finite log probabilities")
        rejected_surface = decode([rejected]).lower()
        retained = []
        for column, token in enumerate(ids):
            if token not in surfaces:
                surfaces[token] = decode([token])
            surface = surfaces[token]
            if token == rejected or surface.lower() == rejected_surface:
                continue
            if len(surface.strip()) < config.min_decoded_chars or (
                config.require_alnum and not any(c.isalnum() for c in surface)
            ):
                continue
            retained.append(column)
        if not retained:
            continue
        best = scores[retained].max()
        retained = [column for column in retained if (scores[column] - best).exp() >= config.min_p]
        # Keep the most likely surviving alternatives, independent of transport order.
        retained.sort(key=lambda column: (-float(scores[column]), ids[column]))
        retained = retained[: config.max_chosen_tokens]
        chosen[row, position, retained] = True
        rejected_by_row[row] = rejected

    occurrences: dict[int, list[tuple[int, int, int]]] = {}
    for row, position, column in chosen.nonzero().tolist():
        occurrences.setdefault(int(candidate_ids[row, position, column]), []).append((row, position, column))
    generator = torch.Generator().manual_seed(seed)
    if occurrences and config.chosen_balance_strength:
        median = float(torch.tensor([len(items) for items in occurrences.values()], dtype=torch.float32).quantile(0.5))
        for items in occurrences.values():
            keep = max(1, round(len(items) * min(1.0, (median / len(items)) ** config.chosen_balance_strength)))
            for index in torch.randperm(len(items), generator=generator)[keep:].tolist():
                chosen[items[index]] = False
    rejected_counts = Counter(token for row, token in rejected_by_row.items() if chosen[row].any())
    if rejected_counts:
        median = float(torch.tensor(list(rejected_counts.values()), dtype=torch.float32).quantile(0.5))
        for row, token in rejected_by_row.items():
            active = chosen[row].any(-1)
            if active.any():
                weights[row, active] = min(1.0, (median / rejected_counts[token]) ** config.rejected_balance_strength)
    return chosen, weights


@torch.no_grad()
def ftpo_counts(
    targets: Sequence[FTPOTargets],
    weights: Sequence[torch.Tensor],
    all_reduce_sum: Callable[[torch.Tensor], torch.Tensor],
) -> FTPOCounts:
    """Count weighted examples and vocabulary entries across the optimizer step."""
    counts = torch.zeros(3, dtype=torch.float64, device=weights[0].device)
    for target, weight in zip(targets, weights, strict=True):
        active_weights = weight * target.chosen_mask.any(-1)
        target_count = target.chosen_mask.sum(-1) + 1  # chosen IDs and the rejected ID
        counts[0] += active_weights.sum()
        counts[1] += (active_weights * target_count).sum()
        counts[2] += (active_weights * (target.reference_logits.shape[-1] - target_count)).sum()
    return FTPOCounts(*all_reduce_sum(counts).tolist())


def ftpo_loss(inputs: FTPOInputs, config: FTPOConfig) -> tuple[torch.Tensor, dict[str, float]]:
    """Return response-shaped losses for the existing global token-mean reducer.

    MSE denominators count vocabulary entries over the entire optimizer step,
    matching the reference even with different chosen counts or microbatching.
    """
    targets = inputs.targets
    active = targets.chosen_mask.any(-1)
    active_rows = active.any(-1)
    chosen = boundary_values(targets.chosen_mask, targets.chosen_mask)
    ids = boundary_values(targets.candidate_ids, targets.chosen_mask).masked_fill(~chosen, 0)
    logits = torch.where(active_rows[:, None], inputs.logits.float(), 0)
    reference = torch.where(active_rows[:, None], targets.reference_logits.float(), 0).detach()
    if reference.shape != logits.shape:
        raise ValueError("FTPO policy and reference vocabulary shapes must match")
    rejected = inputs.rejected_ids
    rejected_logits = logits.gather(-1, rejected[:, None])
    delta = logits.gather(-1, ids) - rejected_logits
    gap = config.margin - delta
    preferences = (F.softplus(gap) * (gap / config.margin).clamp(0, 1) * chosen).sum(-1) / chosen.sum(-1).clamp(min=1)
    target_mask = torch.zeros_like(logits, dtype=torch.bool)
    rows, columns = chosen.nonzero(as_tuple=True)
    target_mask[rows, ids[rows, columns]] = True
    target_mask.scatter_(1, rejected[:, None], True)
    difference = logits - reference
    other = difference.square().masked_fill(target_mask, 0).sum(-1)
    target = (difference.abs() - config.tau_mse_target).clamp(min=0).square().masked_fill(~target_mask, 0).sum(-1)
    counts = inputs.counts
    losses = preferences + counts.examples * (
        config.lambda_mse * other / (counts.other or 1.0) + config.lambda_mse_target * target / (counts.targets or 1.0)
    )
    values = torch.where(active, losses[:, None], 0)
    wins = ((delta > 0) & chosen).sum(-1) / chosen.sum(-1).clamp(min=1)
    # The worker aggregates these counts before computing the chosen-win rate.
    metrics = {
        "ftpo/active_examples": float(active_rows.sum()),
        "ftpo/chosen_win_sum": float(wins[active_rows].detach().sum()),
    }
    return values, metrics
