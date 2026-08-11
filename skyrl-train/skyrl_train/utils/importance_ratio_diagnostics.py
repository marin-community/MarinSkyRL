"""Importance-ratio and token probability-change diagnostics for policy training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from loguru import logger

from skyrl_train.utils.policy_math import LOG_PROB_DELTA_CLIP, masked_mean, safe_exp_delta


TIS_DIAG_KEYS = ("tis/imp_ratio_mean", "tis/imp_ratio_capped_fraction", "tis/log_ratio_abs_mean")
LOG_RATIO_BASE_METRIC_KEYS = (
    "log_ratio_abs_mean",
    "log_ratio_abs_max",
    "n_tokens_dp_gt_1pct",
    "n_tokens_dp_gt_10pct",
    "n_tokens_dp_gt_50pct",
    "log_ratio_abs_p99",
    "log_ratio_diagnostics_failed",
)


def _log_ratio_position_metric_keys(n_position_buckets: int) -> tuple[str, ...]:
    return tuple(f"log_ratio_abs_pos{i * (100 // n_position_buckets):02d}" for i in range(n_position_buckets))


@dataclass
class LogRatioAccumulator:
    """Tensor statistics accumulated across policy micro-batches."""

    abs_sum: torch.Tensor
    n_valid: torch.Tensor
    abs_max: torch.Tensor
    n_gt_1pct: torch.Tensor
    n_gt_10pct: torch.Tensor
    n_gt_50pct: torch.Tensor
    topk_abs: torch.Tensor
    bucket_sums: torch.Tensor
    bucket_counts: torch.Tensor


class LogRatioMonitor:
    """Accumulate a fixed-key log-ratio metric contract across microbatches."""

    def __init__(self, device: torch.device):
        self._accumulator = _empty_log_ratio_accumulator(device)
        self._failed = False

    def add(self, log_probs: torch.Tensor, old_log_probs: torch.Tensor, loss_mask: torch.Tensor) -> None:
        if self._failed:
            return
        try:
            partial = compute_log_ratio_partial(log_probs, old_log_probs, loss_mask)
            merge_log_ratio_partial(self._accumulator, partial)
        except Exception as error:
            logger.warning(f"Log-ratio diagnostics skipped after accumulation failed: {error!r}")
            self._failed = True

    def metrics(self) -> dict[str, float]:
        if self._failed:
            return _failed_log_ratio_metrics()
        try:
            return finalize_log_ratio_metrics(self._accumulator)
        except Exception as error:
            logger.warning(f"Log-ratio diagnostics marked failed after finalization failed: {error!r}")
            return _failed_log_ratio_metrics()


def _failed_log_ratio_metrics() -> dict[str, float]:
    metrics = _log_ratio_diag_zero_metrics()
    metrics["log_ratio_diagnostics_failed"] = 1.0
    return metrics


def compute_tis_diagnostics(
    old_action_log_probs: torch.Tensor,
    rollout_action_logprobs: Optional[torch.Tensor],
    loss_mask: torch.Tensor,
    cap: float,
) -> dict:
    """TIS importance-ratio diagnostics for a policy training micro-batch.

    Mask-weighted means of the importance ratio exp(old_lp - rollout_lp) over
    response tokens. At an on-policy step the ratio should be ~1.0; a large
    deviation or a heavy capped fraction at `cap` (tis_imp_ratio_cap) signals
    that the rollout logprobs are misaligned to the training tokens.

    Always returns the full TIS_DIAG_KEYS set — including the fallback branch
    when rollout logprobs are absent — so every rank contributes identical keys
    to the per-key all_reduce(status) (mismatched keysets deadlock NCCL).
    Callers gate on use_tis; this function does not read config.
    """
    if rollout_action_logprobs is None:
        # Preserve identical rank keysets when the generator omits rollout logprobs.
        values = (1.0, 0.0, 0.0)
        return dict(zip(TIS_DIAG_KEYS, values, strict=True))
    with torch.no_grad():
        cap = float(cap)
        delta = (old_action_log_probs - rollout_action_logprobs).float()
        imp = safe_exp_delta(delta)
        m = loss_mask.float()
        values = (
            masked_mean(imp, m).item(),  # imp_ratio_mean
            masked_mean((imp > cap).float(), m).item(),  # imp_ratio_capped_fraction
            masked_mean(delta.abs(), m).item(),  # log_ratio_abs_mean
        )
        return dict(zip(TIS_DIAG_KEYS, values, strict=True))


def _log_ratio_diag_zero_metrics(n_position_buckets: int = 10) -> dict:
    """The full key set the diagnostic emits, with all values zero.

    Used as a fallback so every rank contributes identical keys to
    `strategy.all_reduce(status)` even if a rank's input is empty/all-padded
    or the helper raises. Mismatched keysets across ranks would deadlock the
    per-key NCCL all-reduce.
    """
    keys = LOG_RATIO_BASE_METRIC_KEYS + _log_ratio_position_metric_keys(n_position_buckets)
    return dict.fromkeys(keys, 0.0)


def _empty_log_ratio_accumulator(device, n_position_buckets: int = 10) -> LogRatioAccumulator:
    """Return a zeroed cross-micro-batch accumulator for log-ratio diagnostics.

    Each micro-batch contributes partial statistics. The last micro-batch finalizes
    them while preserving identical metric keysets across ranks.
    """
    return LogRatioAccumulator(
        abs_sum=torch.zeros((), device=device, dtype=torch.float32),
        n_valid=torch.zeros((), device=device, dtype=torch.float32),
        abs_max=torch.zeros((), device=device, dtype=torch.float32),
        n_gt_1pct=torch.zeros((), device=device, dtype=torch.float32),
        n_gt_10pct=torch.zeros((), device=device, dtype=torch.float32),
        n_gt_50pct=torch.zeros((), device=device, dtype=torch.float32),
        topk_abs=torch.zeros((0,), device=device, dtype=torch.float32),
        bucket_sums=torch.zeros(n_position_buckets, device=device, dtype=torch.float32),
        bucket_counts=torch.zeros(n_position_buckets, device=device, dtype=torch.float32),
    )


def compute_log_ratio_partial(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    loss_mask: torch.Tensor,
    n_position_buckets: int = 10,
) -> LogRatioAccumulator:
    """Compute mergeable log-ratio statistics for one policy micro-batch.

    `topk_abs` stores this micro-batch's top-1%-of-valid values. At finalize,
    the concatenated tensor across micro-batches is the global "top of top"
    sample; its min approximates p99.
    """
    device = log_probs.device

    if log_probs.numel() == 0:
        return _empty_log_ratio_accumulator(device, n_position_buckets)

    abs_log_ratio = (log_probs - old_log_probs).detach().abs().clamp(max=LOG_PROB_DELTA_CLIP).float()
    mask_f = loss_mask.float()
    masked = abs_log_ratio * mask_f

    # Bound top-k by valid tokens so padding cannot contaminate short responses.
    n_valid_int = int(mask_f.sum().item())
    if n_valid_int == 0:
        return _empty_log_ratio_accumulator(device, n_position_buckets)

    # Topk over masked-aware flat. Sentinel ensures topk never picks a padded
    # position when k_local <= n_valid (always true by construction).
    sentinel = torch.tensor(-1.0e9, device=device, dtype=abs_log_ratio.dtype)
    flat = torch.where(mask_f.bool(), abs_log_ratio, sentinel).flatten()
    k_local = max(1, n_valid_int // 100)
    topk_abs = torch.topk(flat, k=min(k_local, flat.numel()), largest=True).values.float()

    # Per-position bucket sums via single scatter_add (no Python loop allocs).
    B, T = log_probs.shape
    seq_lens = mask_f.sum(dim=-1, keepdim=True).clamp(min=1)
    positions = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(0).expand(B, T)
    buckets = (positions / seq_lens * n_position_buckets).clamp(0, n_position_buckets - 1).long()
    bsums = torch.zeros(n_position_buckets, device=device, dtype=torch.float32)
    bcounts = torch.zeros(n_position_buckets, device=device, dtype=torch.float32)
    bsums.scatter_add_(0, buckets.flatten(), masked.flatten())
    bcounts.scatter_add_(0, buckets.flatten(), mask_f.flatten())

    return LogRatioAccumulator(
        abs_sum=masked.sum().float(),
        n_valid=torch.as_tensor(float(n_valid_int), device=device, dtype=torch.float32),
        abs_max=masked.max().float(),
        n_gt_1pct=((abs_log_ratio > 0.01) * mask_f).sum().float(),
        n_gt_10pct=((abs_log_ratio > 0.10) * mask_f).sum().float(),
        n_gt_50pct=((abs_log_ratio > 0.50) * mask_f).sum().float(),
        topk_abs=topk_abs,
        bucket_sums=bsums,
        bucket_counts=bcounts,
    )


def merge_log_ratio_partial(acc: LogRatioAccumulator, partial: LogRatioAccumulator) -> None:
    """In-place merge: additive for sums/counts, max for abs_max, concat for
    topk_abs. Mutates `acc`.
    """
    acc.abs_sum = acc.abs_sum + partial.abs_sum
    acc.n_valid = acc.n_valid + partial.n_valid
    acc.abs_max = torch.maximum(acc.abs_max, partial.abs_max)
    acc.n_gt_1pct = acc.n_gt_1pct + partial.n_gt_1pct
    acc.n_gt_10pct = acc.n_gt_10pct + partial.n_gt_10pct
    acc.n_gt_50pct = acc.n_gt_50pct + partial.n_gt_50pct
    acc.topk_abs = torch.cat([acc.topk_abs, partial.topk_abs])
    acc.bucket_sums = acc.bucket_sums + partial.bucket_sums
    acc.bucket_counts = acc.bucket_counts + partial.bucket_counts


def finalize_log_ratio_metrics(acc: LogRatioAccumulator, n_position_buckets: int = 10) -> dict:
    """Reduce the accumulator to the public scalar metric dictionary.

    One sync (final stack→CPU transfer). Returns the full keyset always
    (zeros where input was empty), so downstream per-key `all_reduce(status)`
    stays keyset-compatible across ranks.
    """
    device = acc.abs_sum.device

    n_valid_safe = acc.n_valid.clamp(min=1.0)
    abs_mean = acc.abs_sum / n_valid_safe
    abs_max = acc.abs_max

    # The minimum concatenated per-batch top-1% value approximates global p99.
    # Uneven micro-batch sizes introduce acceptable monitoring-grade bias.
    if acc.topk_abs.numel() > 0:
        abs_p99 = acc.topk_abs.min()
    else:
        abs_p99 = torch.zeros((), device=device, dtype=torch.float32)

    bucket_means = acc.bucket_sums / acc.bucket_counts.clamp(min=1.0)

    base_vals = (
        torch.stack(
            [
                abs_mean.float(),
                abs_max.float(),
                acc.n_gt_1pct.float(),
                acc.n_gt_10pct.float(),
                acc.n_gt_50pct.float(),
                abs_p99.float(),
                torch.zeros((), device=device, dtype=torch.float32),
            ]
        )
        .cpu()
        .tolist()
    )
    metrics = dict(zip(LOG_RATIO_BASE_METRIC_KEYS, base_vals, strict=True))

    bucket_vals = bucket_means.cpu().tolist()
    metrics.update(dict(zip(_log_ratio_position_metric_keys(n_position_buckets), bucket_vals, strict=True)))

    return metrics
