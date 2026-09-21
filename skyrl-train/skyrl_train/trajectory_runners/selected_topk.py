"""Align generated behavior-policy candidates with admitted training tokens."""

from collections.abc import Sequence
from typing import NamedTuple

import torch

from skyrl_train.distillation import INVALID_TOPK_INDEX
from skyrl_train.trajectory_runners.types import TrajectoryBatch


class AlignedStudentTopK(NamedTuple):
    indices: tuple[tuple[int, ...], ...]
    topk_logprobs: tuple[tuple[float, ...], ...]


def align_student_topk(
    response_ids: Sequence[int],
    loss_mask: Sequence[int],
    generated_ids: Sequence[int],
    candidate_ids: Sequence[Sequence[int]] | None,
    candidate_scores: Sequence[Sequence[float]] | None,
) -> AlignedStudentTopK | None:
    """Return aligned rows, or no evidence if candidates cannot be trusted."""
    if (candidate_ids is None) != (candidate_scores is None):
        raise ValueError("student top-K IDs and behavior scores must be provided together")
    if candidate_ids is None:
        return None
    if len(response_ids) != len(loss_mask):
        raise ValueError("student top-K alignment requires response IDs and loss mask to have equal length")
    if len(generated_ids) != len(candidate_ids) or len(candidate_scores) != len(candidate_ids):
        raise ValueError("student top-K candidate rows must align with generated IDs")
    if not candidate_ids:
        return None
    if list(generated_ids) != [token_id for token_id, keep in zip(response_ids, loss_mask, strict=True) if keep]:
        return None
    width = len(candidate_ids[0])
    if width == 0:
        raise ValueError("student top-K candidate width must be positive")
    if any(
        len(ids) != width or len(scores) != width for ids, scores in zip(candidate_ids, candidate_scores, strict=True)
    ):
        raise ValueError("student top-K candidate widths must agree across tokens")
    aligned_ids = []
    aligned_scores = []
    generated_index = 0
    for keep in loss_mask:
        if keep:
            aligned_ids.append(tuple(candidate_ids[generated_index]))
            aligned_scores.append(tuple(candidate_scores[generated_index]))
            generated_index += 1
        else:
            aligned_ids.append((INVALID_TOPK_INDEX,) * width)
            aligned_scores.append((0.0,) * width)
    return AlignedStudentTopK(tuple(aligned_ids), tuple(aligned_scores))


def collate_behavior_topk(
    trajectory_batch: TrajectoryBatch,
    response_token_ids: list[list[int]],
    loss_mask: torch.Tensor,
    top_k: int,
    *,
    sampled_logprobs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate and right-pad exact behavior top-k evidence for learner tokens."""
    index_rows = trajectory_batch.get("student_topk_indices")
    behavior_rows = trajectory_batch.get("behavior_topk_logprobs")
    loss_masks = trajectory_batch.get("loss_masks")
    if index_rows is None or behavior_rows is None:
        raise ValueError("behavior top-k IDs and logprobs are required for every score-centered batch")
    if (
        top_k < 1
        or loss_mask.ndim != 2
        or loss_masks is None
        or len(index_rows) != len(response_token_ids)
        or len(behavior_rows) != len(response_token_ids)
        or len(loss_masks) != len(response_token_ids)
    ):
        raise ValueError("behavior top-k rows must align with response rows and have positive width")
    if (
        loss_mask.shape[0] != len(response_token_ids)
        or sampled_logprobs is not None
        and sampled_logprobs.shape != loss_mask.shape
    ):
        raise ValueError("behavior top-k mask and sampled logprobs must align with response rows")

    indices = torch.full((*loss_mask.shape, top_k), INVALID_TOPK_INDEX, dtype=torch.long)
    behavior = torch.full((*loss_mask.shape, top_k), torch.nan, dtype=torch.float32)
    for row, (ids, scores, response, mask) in enumerate(
        zip(index_rows, behavior_rows, response_token_ids, loss_masks, strict=True)
    ):
        if len(ids) != len(response) or len(scores) != len(response) or len(response) > loss_mask.shape[1]:
            raise ValueError("behavior top-k evidence must align with exact response token IDs")
        if len(mask) != len(response) or any(value not in (0, 1) for value in mask):
            raise ValueError("behavior top-k loss masks must align with response tokens and contain 0 or 1")
        if not torch.equal(loss_mask[row, : len(response)].to(torch.bool), torch.tensor(mask, dtype=torch.bool)):
            raise ValueError("behavior top-k loss masks changed between rollout and learner")
        if any(len(token_ids) != top_k or len(token_scores) != top_k for token_ids, token_scores in zip(ids, scores)):
            raise ValueError("behavior top-k width must match the configured capture width")
        indices[row, : len(response)] = torch.tensor(ids, dtype=torch.long)
        behavior[row, : len(response)] = torch.tensor(scores, dtype=torch.float32)

    selected = loss_mask.to(torch.bool)
    selected_ids = indices[selected]
    selected_scores = behavior[selected]
    if selected_ids.numel():
        if torch.any(selected_ids < 0):
            raise ValueError("trainable behavior top-k token IDs must be nonnegative")
        ordered = selected_ids.sort(dim=-1).values
        if torch.any(ordered[:, 1:] == ordered[:, :-1]):
            raise ValueError("trainable behavior top-k token IDs must be unique")
        if not torch.isfinite(selected_scores).all() or torch.any(selected_scores > 0):
            raise ValueError("trainable behavior top-k logprobs must be finite and nonpositive")
        if torch.any(selected_scores.exp().sum(dim=-1) > 1 + 1e-4):
            raise ValueError("behavior top-k probabilities must be normalized over the full vocabulary")
        if sampled_logprobs is not None:
            padded_response = torch.zeros(loss_mask.shape, dtype=torch.long)
            for row, response in enumerate(response_token_ids):
                padded_response[row, : len(response)] = torch.tensor(response, dtype=torch.long)
            matched = selected_ids == padded_response[selected].unsqueeze(-1)
            if torch.any(matched):
                head_sample_logprobs = selected_scores[matched]
                expected_logprobs = sampled_logprobs[selected].unsqueeze(-1).expand_as(matched)[matched]
                if not torch.allclose(head_sample_logprobs, expected_logprobs, rtol=0, atol=1e-4):
                    raise ValueError("behavior top-k and sampled-token logprobs disagree for the same token IDs")

    indices.masked_fill_(~selected.unsqueeze(-1), INVALID_TOPK_INDEX)
    behavior.masked_fill_(~selected.unsqueeze(-1), torch.nan)
    return indices, behavior
