"""Align generated behavior-policy candidates with admitted training tokens."""

from collections.abc import Sequence
from typing import NamedTuple

from skyrl_train.distillation import INVALID_TOPK_INDEX


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
