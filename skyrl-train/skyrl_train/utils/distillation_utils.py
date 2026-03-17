"""
Distillation utility functions.

Provides KL divergence computation from sparse (top-K) teacher logprobs,
and Best-of-N selection logic.
"""

from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F


def best_of_n_select(
    rewards: List[float],
    n_samples_per_prompt: int,
) -> List[int]:
    """Select the best sample per prompt group based on reward.

    Args:
        rewards: Flat list of per-sample rewards (response-level, not token-level).
            Length must be divisible by n_samples_per_prompt.
        n_samples_per_prompt: Number of samples generated per prompt.

    Returns:
        List of indices (into the flat rewards list) of the best sample per group.
    """
    assert len(rewards) % n_samples_per_prompt == 0, (
        f"Number of rewards ({len(rewards)}) must be divisible by n_samples_per_prompt ({n_samples_per_prompt})"
    )
    num_prompts = len(rewards) // n_samples_per_prompt
    best_indices = []
    for i in range(num_prompts):
        start = i * n_samples_per_prompt
        end = start + n_samples_per_prompt
        group_rewards = rewards[start:end]
        best_local = max(range(len(group_rewards)), key=lambda j: group_rewards[j])
        best_indices.append(start + best_local)
    return best_indices


def reconstruct_sparse_log_probs(
    top_k_logprobs: torch.Tensor,
    top_k_indices: torch.Tensor,
    vocab_size: int,
) -> torch.Tensor:
    """Reconstruct a full log-probability distribution from sparse top-K values.

    Non-top-K entries are set to -inf (zero probability after softmax).

    Args:
        top_k_logprobs: [B, S, K] top-K log-probability values.
        top_k_indices: [B, S, K] corresponding vocabulary indices (long tensor).
        vocab_size: Size of the vocabulary.

    Returns:
        [B, S, V] log-probability tensor with -inf for non-top-K entries.
    """
    B, S, K = top_k_logprobs.shape
    full = torch.full(
        (B, S, vocab_size),
        float("-inf"),
        dtype=top_k_logprobs.dtype,
        device=top_k_logprobs.device,
    )
    full.scatter_(2, top_k_indices, top_k_logprobs)
    return full


def _normalize_sparse_log_probs(
    top_k_logprobs: torch.Tensor,
) -> torch.Tensor:
    """Re-normalize top-K log-probs so they sum to 1 in probability space.

    This is needed because the top-K log-probs from the teacher don't sum to 1
    (the tail mass is missing). We re-normalize over just the top-K entries.

    Args:
        top_k_logprobs: [B, S, K] top-K log-probability values.

    Returns:
        [B, S, K] re-normalized log-probability values.
    """
    return F.log_softmax(top_k_logprobs, dim=-1)


def forward_kl_from_sparse(
    teacher_top_k_logprobs: torch.Tensor,
    teacher_top_k_indices: torch.Tensor,
    student_log_probs_full: torch.Tensor,
    loss_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute forward KL divergence: KL(teacher || student).

    Mode-covering loss. Uses top-K teacher distribution (re-normalized)
    and gathers the corresponding student log-probs.

    Args:
        teacher_top_k_logprobs: [B, S, K] teacher's top-K log-probs.
        teacher_top_k_indices: [B, S, K] corresponding vocab indices.
        student_log_probs_full: [B, S, V] student's full log-prob distribution.
        loss_mask: Optional [B, S] mask (1 = include, 0 = exclude).

    Returns:
        [B, S] per-token forward KL values.
    """
    # Re-normalize teacher over top-K
    teacher_log_probs = _normalize_sparse_log_probs(teacher_top_k_logprobs)
    teacher_probs = teacher_log_probs.exp()

    # Gather student log-probs at teacher's top-K positions
    student_log_probs_at_k = student_log_probs_full.gather(2, teacher_top_k_indices)

    # KL(teacher || student) = sum_k p_teacher(k) * [log p_teacher(k) - log p_student(k)]
    kl = (teacher_probs * (teacher_log_probs - student_log_probs_at_k)).sum(dim=-1)

    if loss_mask is not None:
        kl = kl * loss_mask

    return kl


def reverse_kl_from_sparse(
    teacher_top_k_logprobs: torch.Tensor,
    teacher_top_k_indices: torch.Tensor,
    student_log_probs_full: torch.Tensor,
    loss_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute reverse KL divergence: KL(student || teacher).

    Mode-seeking loss. Approximated using top-K teacher entries only.

    Note: True reverse KL requires summing over the student's full distribution,
    but tokens outside the teacher's top-K have -inf teacher log-prob, making
    their contribution infinite. We approximate by only summing over the teacher's
    top-K support, which is valid when the student's mass outside top-K is small.

    Args:
        teacher_top_k_logprobs: [B, S, K] teacher's top-K log-probs.
        teacher_top_k_indices: [B, S, K] corresponding vocab indices.
        student_log_probs_full: [B, S, V] student's full log-prob distribution.
        loss_mask: Optional [B, S] mask.

    Returns:
        [B, S] per-token reverse KL values.
    """
    teacher_log_probs = _normalize_sparse_log_probs(teacher_top_k_logprobs)

    # Gather student log-probs and probs at teacher's top-K positions
    student_log_probs_at_k = student_log_probs_full.gather(2, teacher_top_k_indices)
    student_probs_at_k = student_log_probs_at_k.exp()

    # KL(student || teacher) ≈ sum_k p_student(k) * [log p_student(k) - log p_teacher(k)]
    kl = (student_probs_at_k * (student_log_probs_at_k - teacher_log_probs)).sum(dim=-1)

    if loss_mask is not None:
        kl = kl * loss_mask

    return kl


def jsd_from_sparse(
    teacher_top_k_logprobs: torch.Tensor,
    teacher_top_k_indices: torch.Tensor,
    student_log_probs_full: torch.Tensor,
    loss_mask: Optional[torch.Tensor] = None,
    alpha: float = 0.5,
) -> torch.Tensor:
    """Compute Jensen-Shannon divergence between teacher and student.

    JSD(teacher, student) = alpha * KL(teacher || M) + (1-alpha) * KL(student || M)
    where M = alpha * teacher + (1-alpha) * student.

    Approximated using the teacher's top-K support.

    Args:
        teacher_top_k_logprobs: [B, S, K] teacher's top-K log-probs.
        teacher_top_k_indices: [B, S, K] corresponding vocab indices.
        student_log_probs_full: [B, S, V] student's full log-prob distribution.
        loss_mask: Optional [B, S] mask.
        alpha: Mixture weight for teacher (default 0.5 for symmetric JSD).

    Returns:
        [B, S] per-token JSD values.
    """
    teacher_log_probs = _normalize_sparse_log_probs(teacher_top_k_logprobs)
    teacher_probs = teacher_log_probs.exp()

    student_log_probs_at_k = student_log_probs_full.gather(2, teacher_top_k_indices)
    student_probs_at_k = student_log_probs_at_k.exp()

    # Mixture distribution M at top-K positions
    m_probs = alpha * teacher_probs + (1 - alpha) * student_probs_at_k
    m_log_probs = m_probs.log()

    # JSD = alpha * KL(teacher || M) + (1-alpha) * KL(student || M)
    kl_teacher_m = (teacher_probs * (teacher_log_probs - m_log_probs)).sum(dim=-1)
    kl_student_m = (student_probs_at_k * (student_log_probs_at_k - m_log_probs)).sum(dim=-1)
    jsd = alpha * kl_teacher_m + (1 - alpha) * kl_student_m

    if loss_mask is not None:
        jsd = jsd * loss_mask

    return jsd


def kl_from_sparse_teacher_logprobs(
    teacher_top_k_logprobs: torch.Tensor,
    teacher_top_k_indices: torch.Tensor,
    student_log_probs_full: torch.Tensor,
    kl_type: str = "forward",
    loss_mask: Optional[torch.Tensor] = None,
    jsd_alpha: float = 0.5,
) -> torch.Tensor:
    """Unified interface for KL-family divergences from sparse teacher logprobs.

    Args:
        teacher_top_k_logprobs: [B, S, K] teacher's top-K log-probs.
        teacher_top_k_indices: [B, S, K] corresponding vocab indices.
        student_log_probs_full: [B, S, V] student's full log-prob distribution.
        kl_type: One of "forward", "reverse", "jsd".
        loss_mask: Optional [B, S] mask.
        jsd_alpha: Mixture weight for JSD (only used when kl_type="jsd").

    Returns:
        [B, S] per-token divergence values.
    """
    if kl_type == "forward":
        return forward_kl_from_sparse(
            teacher_top_k_logprobs, teacher_top_k_indices, student_log_probs_full, loss_mask
        )
    elif kl_type == "reverse":
        return reverse_kl_from_sparse(
            teacher_top_k_logprobs, teacher_top_k_indices, student_log_probs_full, loss_mask
        )
    elif kl_type == "jsd":
        return jsd_from_sparse(
            teacher_top_k_logprobs, teacher_top_k_indices, student_log_probs_full, loss_mask, jsd_alpha
        )
    else:
        raise ValueError(f"Unknown kl_type: {kl_type}. Must be one of 'forward', 'reverse', 'jsd'.")
