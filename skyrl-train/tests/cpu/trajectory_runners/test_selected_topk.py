"""Selected-candidate alignment keeps token identity distinct from malformed evidence."""

import pytest
import torch

from skyrl_train.trajectory_runners.selected_topk import align_student_topk, collate_behavior_topk


def test_changed_trainable_token_identity_discards_selected_evidence():
    assert align_student_topk([4, 5], [1, 0], [3], [[3, 6]], [[-0.1, -2.0]]) is None


@pytest.mark.parametrize(
    ("candidate_ids", "candidate_scores", "message"),
    [
        ([[3, 6]], [], "align with generated IDs"),
        ([[3]], [[-0.1, -2.0]], "widths must agree"),
        ([[]], [[]], "width must be positive"),
    ],
)
def test_malformed_selected_evidence_fails_closed(candidate_ids, candidate_scores, message):
    with pytest.raises(ValueError, match=message):
        align_student_topk([3], [1], [3], candidate_ids, candidate_scores)


def test_behavior_topk_collation_ignores_masked_failure_but_checks_sampled_probability():
    batch = {
        "loss_masks": [[0, 0], [1]],
        "student_topk_indices": [[[-1, -1], [-1, -1]], [[3, 4]]],
        "behavior_topk_logprobs": [[[0.0, 0.0], [0.0, 0.0]], [[-0.5, -1.5]]],
    }
    mask = torch.tensor([[0, 0], [1, 0]])
    sampled = torch.tensor([[0.0, 0.0], [-0.5, 0.0]])
    indices, scores = collate_behavior_topk(batch, [[2, 3], [3]], mask, 2, sampled_logprobs=sampled)
    assert indices[1, 0].tolist() == [3, 4]
    assert indices[0].tolist() == [[-1, -1], [-1, -1]]
    assert torch.isnan(scores[0]).all()

    with pytest.raises(ValueError, match="disagree"):
        collate_behavior_topk(batch, [[2, 3], [3]], mask, 2, sampled_logprobs=sampled - 0.1)
