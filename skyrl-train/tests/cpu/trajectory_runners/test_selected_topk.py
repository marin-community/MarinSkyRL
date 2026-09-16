"""Selected-candidate alignment keeps token identity distinct from malformed evidence."""

import pytest

from skyrl_train.trajectory_runners.selected_topk import align_student_topk


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
