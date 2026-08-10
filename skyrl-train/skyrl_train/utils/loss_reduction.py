"""Policy loss masks, reductions, and global denominators.

Adapted from VERL's ``trainer/ppo/core_algos.py`` (ByteDance and Hugging Face),
licensed under Apache 2.0.
"""

from __future__ import annotations

from typing import Literal, Optional

import torch

from skyrl_train.utils.policy_math import masked_mean, right_pad_to_match


SPAN_THINK_TAG: int = 1  # mirrors span_tagger.SPAN_THINK (kept local to avoid a torch-free import cycle)


def build_think_weighted_loss_mask(
    loss_mask: Optional[torch.Tensor],
    response_span_tags: Optional[torch.Tensor],
    think_token_weight: float,
) -> Optional[torch.Tensor]:
    """Return a per-token loss-weight mask that down-weights THINK tokens.

    Args:
        loss_mask: the 0/1 (or already-weighted) loss mask, shape (B, A).
        response_span_tags: per-token span tags (SPAN_THINK==1), shape (B, A) or
            None. Tagged 1:1 with the response tokens (== loss_mask layout).
        think_token_weight: weight applied to THINK tokens (1.0 == no-op).

    Returns:
        - `loss_mask` UNCHANGED (same object) when ``think_token_weight == 1.0``
          or ``response_span_tags is None`` or ``loss_mask is None`` — the
          byte-identical default/flag-off path.
        - Otherwise a NEW float tensor equal to ``loss_mask`` everywhere except
          THINK positions, which are multiplied by ``think_token_weight``
          (down-weighted but, for weight > 0, still counted in the support).
    """
    if loss_mask is None or response_span_tags is None or think_token_weight == 1.0:
        return loss_mask

    # Per-token multiplier: think_token_weight on THINK tokens, 1.0 elsewhere.
    # Align tags to the loss_mask response width defensively (right-padded).
    tags = right_pad_to_match(response_span_tags, loss_mask)
    is_think = tags == SPAN_THINK_TAG
    weight = torch.where(
        is_think,
        torch.as_tensor(think_token_weight, dtype=torch.float32, device=loss_mask.device),
        torch.ones((), dtype=torch.float32, device=loss_mask.device),
    )
    return loss_mask.to(torch.float32) * weight


def reduce_loss(
    loss: torch.Tensor,
    loss_mask: Optional[torch.Tensor],
    loss_reduction: Literal["token_mean", "sequence_mean", "seq_mean_token_sum_norm", "seq_mean_token_sum_norm_global"],
    max_seq_len: Optional[int] = None,
    global_denom: Optional[float] = None,
) -> torch.Tensor:
    if loss_reduction == "token_mean":
        # sum over *all* valid tokens, divide by total valid-token count
        loss = masked_mean(loss, loss_mask)
    elif loss_reduction == "sequence_mean":
        # per-sequence token-mean (dim=-1), then batch-mean
        loss = masked_mean(loss, loss_mask, dim=-1).mean()
    elif loss_reduction == "seq_mean_token_sum_norm":
        # per-sequence token-sum, normalized by the max sequence length, then batch mean
        # this is the Dr. GRPO loss reduction to avoid length bias by normalizing by a constant
        assert max_seq_len is not None, "max_seq_len must be provided for seq_mean_token_sum_norm loss reduction"
        # NOTE: max_seq_len is computed as cfg.generator.max_input_length + cfg.generator.sampling_params.max_generate_length by default
        if loss_mask is not None:
            seq_losses = torch.sum(loss * loss_mask, dim=-1) / max_seq_len
        else:
            # If no mask, assume all tokens are valid
            seq_losses = torch.sum(loss, dim=-1) / max_seq_len
        loss = torch.mean(seq_losses)
    elif loss_reduction == "seq_mean_token_sum_norm_global":
        # Sum each micro-batch numerator against the driver-computed global denominator.
        # This term is already normalized, so callers must not divide by accumulation steps.
        assert global_denom is not None, "global_denom must be provided for seq_mean_token_sum_norm_global"
        if loss_mask is not None:
            loss = torch.sum(loss * loss_mask) / global_denom
        else:
            # If no mask, assume all tokens are valid
            loss = torch.sum(loss) / global_denom
    else:
        raise ValueError(f"Invalid loss reduction type: {loss_reduction}")
    return loss


def count_nonzero_advantage_seqs(advantages: torch.Tensor) -> float:
    """Number of sequences (rows) carrying at least one non-zero-advantage token.

    Zero-advantage sequences (excluded / k<2 / zero-variance RLOO groups) contribute
    no gradient, so they must not inflate the global loss denominator Z.

    BIT-IDENTICAL under any disjoint row partition: each row's ``abs().sum(dim=-1) > 0``
    is an EXACT all-zero test (a sum of non-negative magnitudes is 0 iff every element
    is exactly 0 -- no float cancellation possible), independent of device, dtype, and
    how the rows are chunked across data-parallel ranks. Hence summing this count over
    disjoint per-rank shards equals computing it once over the full concatenated batch.
    """
    return float((advantages.abs().sum(dim=-1) > 0).sum().item())


def compute_global_loss_denom(advantages: torch.Tensor, max_seq_len: int, ranks_per_dp_group: int) -> float:
    """Compute the collective-free global denominator for sequence-normalized loss.

    Under ``MeshDispatch`` the full batch is split into ``dp_size`` disjoint row-chunks;
    every rank in a data-parallel group receives the same chunk, so the full-group sum
    equals ``ranks_per_dp_group * (nonzero-adv count over the full batch)`` -- because
    the per-chunk counts summed over the dp groups equal the full-batch count
    (:func:`count_nonzero_advantage_seqs` is partition-invariant).

    ``ranks_per_dp_group = world_size // dp_size``. The ``max(., 1.0)`` clamp matches the
    reduction contract so an all-zero-advantage batch still yields a valid denominator.
    """
    global_num_seqs = ranks_per_dp_group * count_nonzero_advantage_seqs(advantages)
    return max(global_num_seqs, 1.0) * max_seq_len
