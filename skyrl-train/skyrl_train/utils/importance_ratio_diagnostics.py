"""Importance-ratio and token probability-change diagnostics for policy training."""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
from typing import Optional

import torch
from loguru import logger

from skyrl_train.utils.policy_math import LOG_PROB_DELTA_CLIP, masked_mean, safe_exp_delta


LOG_RATIO_TAIL_CAPACITY = 8192

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


def ratio_statistics(delta: torch.Tensor, *, eps_clip_low: float = 0.2, eps_clip_high: float = 0.2) -> dict[str, float]:
    """Finite-token likelihood-ratio diagnostics; exponentials alone are clipped.

    These are empirical token diagnostics, not unbiased policy KL estimates.
    In particular, async consume/generate ratios include age drift and engine
    mismatch. Empty populations expose coverage without inventing statistics.
    """
    delta = delta.detach().double().reshape(-1)
    selected = delta.numel()
    delta = delta[torch.isfinite(delta)]
    result = {"selected_tokens": float(selected), "finite_tokens": float(delta.numel())}
    if selected:
        result["finite_fraction"] = delta.numel() / selected
    if not delta.numel():
        return result
    absolute = delta.abs()
    quantiles = torch.quantile(absolute, absolute.new_tensor([0.5, 0.95, 0.99, 0.999]))
    weights = (delta - delta.max()).exp()
    result.update(
        log_ratio_mean=delta.mean().item(),
        abs_log_ratio_mean=absolute.mean().item(),
        abs_log_ratio_p50=quantiles[0].item(),
        abs_log_ratio_p95=quantiles[1].item(),
        abs_log_ratio_p99=quantiles[2].item(),
        abs_log_ratio_p999=quantiles[3].item(),
        abs_log_ratio_max=absolute.max().item(),
        lower_clip_pressure=(delta < (math.log1p(-eps_clip_low) if eps_clip_low < 1 else -math.inf))
        .double()
        .mean()
        .item(),
        upper_clip_pressure=(delta > math.log1p(eps_clip_high)).double().mean().item(),
        frac_outside_0_5_2=(absolute > math.log(2)).double().mean().item(),
        frac_below_1e_5=(delta < math.log(1e-5)).double().mean().item(),
        ess_fraction=(weights.sum().square() / (delta.numel() * weights.square().sum())).item(),
        kl_k1=(-delta).mean().item(),
        kl_k3=(delta.clamp(-LOG_PROB_DELTA_CLIP, LOG_PROB_DELTA_CLIP).exp() - delta - 1).mean().item(),
        chi2=(2 * delta.clamp(-LOG_PROB_DELTA_CLIP, LOG_PROB_DELTA_CLIP)).exp().mean().item() - 1,
        mean_squared_log_ratio=delta.square().mean().item(),
    )
    result["frac_outside_0.5_2"] = result.pop("frac_outside_0_5_2")
    result["frac_below_1e-5"] = result.pop("frac_below_1e_5")
    return result


def mismatch_ratio_metrics(
    learner_logprobs: torch.Tensor,
    rollout_logprobs: torch.Tensor | None,
    loss_mask: torch.Tensor,
    rollout_age: torch.Tensor | None,
    *,
    position_window: int = 256,
    eps_clip_low: float = 0.2,
    eps_clip_high: float = 0.2,
) -> dict[str, float]:
    """Condition consume-time trainer/vLLM ratios on age and token position.

    Only age0 isolates engine mismatch. Other ages and pooled values measure
    its product with policy drift. Position buckets overlap on short responses.
    """
    if type(position_window) is not int or position_window <= 0:
        raise ValueError("position_window must be a positive integer")
    mask = loss_mask.detach().cpu() > 0
    ages = torch.zeros(mask.shape[0], dtype=torch.int32) if rollout_age is None else rollout_age.detach().cpu()
    if ages.shape != (mask.shape[0],) or (ages < 0).any() or not torch.equal(ages, ages.int()):
        raise ValueError("rollout_age must contain one nonnegative integer per response row")
    buckets = {"pooled": torch.ones_like(ages, dtype=torch.bool)}
    buckets.update({f"age{age}": ages == age for age in range(4)})
    buckets.update({"age4-7": (ages >= 4) & (ages <= 7), "age8+": ages >= 8})
    delta = None
    if rollout_logprobs is not None:
        # CPU float64 and masking before subtraction avoid padded NaNs.
        delta = torch.zeros_like(mask, dtype=torch.float64)
        delta[mask] = learner_logprobs.detach().cpu().double()[mask] - rollout_logprobs.detach().cpu().double()[mask]
    positions = torch.arange(mask.shape[1]).unsqueeze(0)
    lengths = mask.sum(-1, keepdim=True)
    position_masks = {
        f"first{position_window}": positions < position_window,
        f"last{position_window}": positions >= lengths - position_window,
    }
    position_masks["middle"] = ~position_masks[f"first{position_window}"] & ~position_masks[f"last{position_window}"]
    result = {}
    for name, rows in buckets.items():
        selected = mask & rows.unsqueeze(1)
        prefix = f"policy/mismatch/{name}/"
        if delta is None:
            metrics = {"selected_tokens": float(selected.sum()), "finite_tokens": 0.0, "missing_behavior": 1.0}
        else:
            metrics = {
                **ratio_statistics(delta[selected], eps_clip_low=eps_clip_low, eps_clip_high=eps_clip_high),
                "missing_behavior": 0.0,
            }
        result.update({prefix + key: value for key, value in metrics.items()})
        for position, position_mask in position_masks.items():
            selected_position = selected & position_mask
            stats = (
                ratio_statistics(delta[selected_position])
                if delta is not None
                else {
                    "selected_tokens": float(selected_position.sum()),
                    "finite_tokens": 0.0,
                }
            )
            for key in ("selected_tokens", "finite_tokens", "abs_log_ratio_mean", "frac_outside_0.5_2"):
                if key in stats:
                    result[f"{prefix}pos_{position}/{key}"] = stats[key]
    return result


def behavior_drift_metrics(
    learner_logprobs: torch.Tensor,
    rollout_logprobs: torch.Tensor | None,
    loss_mask: torch.Tensor,
    *,
    eps_clip_low: float,
    eps_clip_high: float,
) -> dict[str, float]:
    """Measure pre-update learner minus reported behavior logprobs on the full batch.

    Mask before arithmetic; pool selected tokens rather than averaging rank or
    sequence means. Raw generator logprobs need not describe processed sampling,
    so these are model-likelihood diagnostics, not KL or trajectory IS estimates.
    ESS describes token-weight concentration, not independent usable samples.
    Missing/empty populations omit undefined statistics and retain coverage counts.
    """
    selected = loss_mask > 0
    count = int(selected.sum().item())
    metrics = {
        "selected_tokens": float(count),
        "finite_tokens": 0.0,
        "missing_behavior": float(rollout_logprobs is None),
    }
    if rollout_logprobs is not None and count:
        learner = learner_logprobs.detach()[selected].double()
        behavior = rollout_logprobs.detach()[selected].double()
        finite = torch.isfinite(learner) & torch.isfinite(behavior)
        delta = learner[finite] - behavior[finite]
        metrics["finite_tokens"] = float(delta.numel())
        metrics["finite_fraction"] = delta.numel() / count
        if delta.numel():
            absolute = delta.abs()
            quantiles = torch.quantile(absolute, absolute.new_tensor([0.5, 0.95, 0.99]))
            # Shift, never clamp: a severe outlier must still reduce concentration.
            weights = (delta - delta.max()).exp()
            lower = math.log1p(-eps_clip_low) if eps_clip_low < 1 else -math.inf
            metrics.update(
                log_ratio_mean=delta.mean().item(),
                mean_squared_log_ratio=delta.square().mean().item(),
                abs_log_ratio_mean=absolute.mean().item(),
                abs_log_ratio_p50=quantiles[0].item(),
                abs_log_ratio_p95=quantiles[1].item(),
                abs_log_ratio_p99=quantiles[2].item(),
                abs_log_ratio_max=absolute.max().item(),
                log_mean_ratio=(delta.logsumexp(0) - math.log(delta.numel())).item(),
                lower_clip_pressure=(delta < lower).double().mean().item(),
                upper_clip_pressure=(delta > math.log1p(eps_clip_high)).double().mean().item(),
                token_weight_ess_fraction=(weights.sum().square() / (delta.numel() * weights.square().sum())).item(),
            )
    elif count:
        metrics["finite_fraction"] = 0.0
    return {f"policy/behavior_drift/{name}": value for name, value in metrics.items()}


def _stale_metric_aliases(metrics: dict[str, float]) -> dict[str, float]:
    """Name the training-forward/consume-forward ratio while keeping old panels."""
    names = {
        "mean": "log_ratio_mean",
        "mean_squared": "mean_squared_log_ratio",
        "abs_mean": "abs_log_ratio_mean",
        "abs_max": "abs_log_ratio_max",
        "abs_p99": "abs_log_ratio_p99",
        "p99_approximate": "p99_approximate",
        "abs_p999": "abs_log_ratio_p999",
        "frac_outside_0_5_2": "frac_outside_0.5_2",
        "frac_below_1e_5": "frac_below_1e-5",
        "ess_fraction": "ess_fraction",
        "kl_k1": "kl_k1",
        "kl_k3": "kl_k3",
        "chi2": "chi2",
        "statistics_valid": "statistics_valid",
        "selected_tokens": "selected_tokens",
        "p999_valid": "p999_valid",
    }
    aliases = {f"stale/{name}": metrics[f"log_ratio_{old}"] for old, name in names.items()}
    aliases.update(
        {
            "stale/" + key.removeprefix("log_ratio_").replace("frac_outside_0_5_2", "frac_outside_0.5_2"): value
            for key, value in metrics.items()
            if key.startswith("log_ratio_pos_")
        }
    )
    return aliases


def _ratio_extra_keys(window: int = 256) -> tuple[str, ...]:
    keys = (
        "log_ratio_mean",
        "log_ratio_mean_squared",
        "log_ratio_abs_p999",
        "log_ratio_frac_outside_0_5_2",
        "log_ratio_frac_below_1e_5",
        "log_ratio_ess_fraction",
        "log_ratio_kl_k1",
        "log_ratio_kl_k3",
        "log_ratio_chi2",
        "log_ratio_statistics_valid",
        "log_ratio_selected_tokens",
        "log_ratio_p999_valid",
    )
    return keys + tuple(
        f"log_ratio_pos_{position}/{key}"
        for position in (f"first{window}", f"last{window}", "middle")
        for key in ("selected_tokens", "abs_log_ratio_mean", "frac_outside_0_5_2")
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
    moments: torch.Tensor  # delta, delta², clipped exp(delta), clipped exp(2delta), outside2x, below1e-5
    maximum_delta: torch.Tensor
    shifted_weight_sum: torch.Tensor
    shifted_weight_square_sum: torch.Tensor
    top_per_mille: torch.Tensor
    position_sums: torch.Tensor
    position_counts: torch.Tensor
    position_outside: torch.Tensor


class LogRatioMonitor:
    """Accumulate a fixed-key log-ratio metric contract across microbatches."""

    def __init__(self, device: torch.device, *, position_window: int = 256):
        if type(position_window) is not int or position_window <= 0:
            raise ValueError("position_window must be a positive integer")
        self.position_window = position_window
        self._accumulator = _empty_log_ratio_accumulator(device)
        self._failed = False

    def add(self, log_probs: torch.Tensor, old_log_probs: torch.Tensor, loss_mask: torch.Tensor) -> None:
        if self._failed:
            return
        try:
            partial = compute_log_ratio_partial(
                log_probs, old_log_probs, loss_mask, position_window=self.position_window
            )
            merge_log_ratio_partial(self._accumulator, partial)
        except Exception as error:
            logger.warning(f"Log-ratio diagnostics skipped after accumulation failed: {error!r}")
            self._failed = True

    def metrics(self, *, gather_fn=None, owns_tokens: bool = True) -> dict[str, float]:
        accumulator = self._accumulator
        failed = self._failed
        if gather_fn is not None:
            # Every rank participates even after local failure. Reducing fixed
            # sufficient statistics before finalization keeps ESS, token means,
            # tail quantiles and validity meaningful for uneven rank populations.
            header, tail = pack_log_ratio_accumulator(accumulator, failed=failed, owns_tokens=owns_tokens)
            headers, tails = gather_fn(header), gather_fn(tail)
            accumulator, failed = pool_log_ratio_accumulators(headers, tails)
        if failed:
            return _failed_log_ratio_metrics(self.position_window)
        try:
            return finalize_log_ratio_metrics(accumulator, position_window=self.position_window)
        except Exception as error:
            logger.warning(f"Log-ratio diagnostics marked failed after finalization failed: {error!r}")
            return _failed_log_ratio_metrics(self.position_window)


def gather_ratio_tensor(tensor: torch.Tensor, *, group=None) -> list[torch.Tensor]:
    """Gather a fixed-size statistics tensor over the supplied ownership group."""
    if not torch.distributed.is_initialized():
        return [tensor]
    values = [torch.empty_like(tensor) for _ in range(torch.distributed.get_world_size(group))]
    torch.distributed.all_gather(values, tensor, group=group)
    return values


def pack_log_ratio_accumulator(accumulator, *, failed: bool = False, owns_tokens: bool = True):
    """Encode bounded sufficient statistics, excluding replicated token owners."""
    if not owns_tokens:
        accumulator = _empty_log_ratio_accumulator(accumulator.abs_sum.device)
    scalar_fields = [field.name for field in fields(accumulator) if field.name not in {"topk_abs", "top_per_mille"}]
    approximate_p99 = (
        accumulator.topk_abs.min() if accumulator.topk_abs.numel() else accumulator.abs_sum.new_tensor(math.inf)
    )
    header = torch.cat(
        [accumulator.abs_sum.new_tensor([float(failed)], dtype=torch.float64), approximate_p99.double().reshape(1)]
        + [getattr(accumulator, name).double().reshape(-1) for name in scalar_fields]
    )
    tail = accumulator.abs_sum.new_full((LOG_RATIO_TAIL_CAPACITY,), -math.inf, dtype=torch.float32)
    tail[: accumulator.top_per_mille.numel()] = accumulator.top_per_mille
    return header, tail


def pool_log_ratio_accumulators(headers, tails):
    """Merge rank payloads before computing any nonlinear scalar statistic."""
    if len(headers) != len(tails) or not headers:
        raise ValueError("Ratio reduction needs one header and tail per rank")
    pooled = _empty_log_ratio_accumulator(headers[0].device)
    scalar_fields = [field.name for field in fields(pooled) if field.name not in {"topk_abs", "top_per_mille"}]
    failed = False
    for header, tail in zip(headers, tails, strict=True):
        failed = failed or bool(header[0].item())
        partial = _empty_log_ratio_accumulator(header.device)
        offset = 2
        for name in scalar_fields:
            target = getattr(partial, name)
            target.copy_(header[offset : offset + target.numel()].reshape(target.shape))
            offset += target.numel()
        if offset != header.numel() or tail.numel() != LOG_RATIO_TAIL_CAPACITY:
            raise ValueError("Ratio reduction payload shape changed")
        partial.topk_abs = header[1:2].float() if torch.isfinite(header[1]) else partial.topk_abs
        partial.top_per_mille = tail[torch.isfinite(tail)]
        merge_log_ratio_partial(pooled, partial)
    return pooled, failed


def _failed_log_ratio_metrics(position_window: int = 256) -> dict[str, float]:
    metrics = _log_ratio_diag_zero_metrics(position_window=position_window)
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


def _log_ratio_diag_zero_metrics(n_position_buckets: int = 10, *, position_window: int = 256) -> dict:
    """The full key set the diagnostic emits, with all values zero.

    Used as a fallback so every rank contributes identical keys to
    `strategy.all_reduce(status)` even if a rank's input is empty/all-padded
    or the helper raises. Mismatched keysets across ranks would deadlock the
    per-key NCCL all-reduce.
    """
    keys = (
        LOG_RATIO_BASE_METRIC_KEYS
        + _log_ratio_position_metric_keys(n_position_buckets)
        + _ratio_extra_keys(position_window)
    )
    metrics = dict.fromkeys(keys, 0.0)
    metrics["log_ratio_p99_approximate"] = 1.0
    return {**metrics, **_stale_metric_aliases(metrics)}


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
        moments=torch.zeros(6, device=device, dtype=torch.float64),
        maximum_delta=torch.tensor(-math.inf, device=device, dtype=torch.float64),
        shifted_weight_sum=torch.zeros((), device=device, dtype=torch.float64),
        shifted_weight_square_sum=torch.zeros((), device=device, dtype=torch.float64),
        top_per_mille=torch.zeros((0,), device=device, dtype=torch.float32),
        position_sums=torch.zeros(3, device=device, dtype=torch.float64),
        position_counts=torch.zeros(3, device=device, dtype=torch.float64),
        position_outside=torch.zeros(3, device=device, dtype=torch.float64),
    )


def compute_log_ratio_partial(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    loss_mask: torch.Tensor,
    n_position_buckets: int = 10,
    *,
    position_window: int = 256,
) -> LogRatioAccumulator:
    """Compute mergeable log-ratio statistics for one policy micro-batch.

    `topk_abs` stores this micro-batch's top-1%-of-valid values. At finalize,
    the concatenated tensor across micro-batches is the global "top of top"
    sample; its min approximates p99.
    """
    device = log_probs.device

    if log_probs.numel() == 0:
        return _empty_log_ratio_accumulator(device, n_position_buckets)

    selected = loss_mask > 0
    delta = torch.zeros_like(log_probs, dtype=torch.float64)
    delta[selected] = log_probs.detach().double()[selected] - old_log_probs.detach().double()[selected]
    if not torch.isfinite(delta[selected]).all():
        raise ValueError("Nonfinite selected log-ratio token")
    abs_log_ratio = delta.abs().float()
    mask_f = selected.float()
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

    values = delta[selected]
    maximum = values.max()
    weights = (values - maximum).exp()
    positions = torch.arange(T, device=device).unsqueeze(0)
    first = (positions < position_window).expand(B, T)
    last = positions >= mask_f.sum(-1, keepdim=True) - position_window
    absolute_masks = torch.stack([first, last, ~first & ~last]) & selected.unsqueeze(0)
    outside = delta.abs() > math.log(2)
    moments = torch.stack(
        [
            values.sum(),
            values.square().sum(),
            values.clamp(-LOG_PROB_DELTA_CLIP, LOG_PROB_DELTA_CLIP).exp().sum(),
            (2 * values.clamp(-LOG_PROB_DELTA_CLIP, LOG_PROB_DELTA_CLIP)).exp().sum(),
            outside[selected].sum().double(),
            (values < math.log(1e-5)).sum().double(),
        ]
    )
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
        moments=moments,
        maximum_delta=maximum,
        shifted_weight_sum=weights.sum(),
        shifted_weight_square_sum=weights.square().sum(),
        top_per_mille=torch.topk(values.abs(), k=min(LOG_RATIO_TAIL_CAPACITY, n_valid_int)).values.float(),
        position_sums=(absolute_masks * delta.abs()).sum((1, 2)),
        position_counts=absolute_masks.sum((1, 2)).double(),
        position_outside=(absolute_masks & outside).sum((1, 2)).double(),
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
    maximum = torch.maximum(acc.maximum_delta, partial.maximum_delta)
    safe_maximum = torch.where(torch.isfinite(maximum), maximum, maximum.new_zeros(()))
    old_scale, new_scale = (acc.maximum_delta - safe_maximum).exp(), (partial.maximum_delta - safe_maximum).exp()
    acc.shifted_weight_sum = acc.shifted_weight_sum * old_scale + partial.shifted_weight_sum * new_scale
    acc.shifted_weight_square_sum = (
        acc.shifted_weight_square_sum * old_scale.square() + partial.shifted_weight_square_sum * new_scale.square()
    )
    acc.maximum_delta = maximum
    acc.moments = acc.moments + partial.moments
    tail = torch.cat([acc.top_per_mille, partial.top_per_mille])
    acc.top_per_mille = torch.topk(tail, k=min(LOG_RATIO_TAIL_CAPACITY, tail.numel())).values
    acc.position_sums = acc.position_sums + partial.position_sums
    acc.position_counts = acc.position_counts + partial.position_counts
    acc.position_outside = acc.position_outside + partial.position_outside


def finalize_log_ratio_metrics(
    acc: LogRatioAccumulator, n_position_buckets: int = 10, *, position_window: int = 256
) -> dict:
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
    metrics["log_ratio_p99_approximate"] = 1.0

    bucket_vals = bucket_means.cpu().tolist()
    metrics.update(dict(zip(_log_ratio_position_metric_keys(n_position_buckets), bucket_vals, strict=True)))

    count = acc.n_valid.double().clamp(min=1)
    means = acc.moments / count
    ess = acc.shifted_weight_sum.square() / (count * acc.shifted_weight_square_sum).clamp(min=1e-300)
    # Keep a bounded global top tail, not local top-0.1% minima: a microbatch
    # containing every outlier must retain enough values for the pooled quantile.
    rank_from_top = (int(acc.n_valid.item()) - 1) * 0.001
    upper = math.ceil(rank_from_top)
    p999_valid = 0 <= upper < acc.top_per_mille.numel()
    p999 = acc.n_valid.new_zeros(())
    if p999_valid:
        lower = math.floor(rank_from_top)
        fraction = rank_from_top - lower
        p999 = acc.top_per_mille[lower] * (1 - fraction) + acc.top_per_mille[upper] * fraction
    valid = (acc.n_valid > 0).double()
    extra = torch.stack(
        [
            means[0],
            means[1],
            p999,
            means[4],
            means[5],
            ess,
            -means[0],
            (means[2] - means[0] - 1) * valid,
            (means[3] - 1) * valid,
            valid,
            acc.n_valid,
            valid.new_tensor(float(p999_valid)),
        ]
    )
    position_counts = acc.position_counts.clamp(min=1)
    positions = torch.stack(
        [acc.position_counts, acc.position_sums / position_counts, acc.position_outside / position_counts], dim=1
    ).reshape(-1)
    values = torch.cat([extra, positions]).cpu().tolist()
    metrics.update(dict(zip(_ratio_extra_keys(position_window), values, strict=True)))
    metrics.update(_stale_metric_aliases(metrics))
    return metrics
