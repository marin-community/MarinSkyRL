"""Selected-candidate alignment keeps token identity distinct from malformed evidence."""

import numpy as np
import pytest

from skyrl_train.trajectory_runners.selected_topk import align_student_topk


def test_changed_trainable_token_identity_discards_selected_evidence():
    assert align_student_topk([4, 5], [1, 0], [3], [[3, 6]], [[-0.1, -2.0]]) is None


def test_selected_candidates_stay_in_aligned_arrays():
    aligned = align_student_topk([3, 99, 4], [1, 0, 1], [3, 4], [[3, 6], [4, 7]], [[-0.1, -2.0], [-0.2, -1.9]])

    assert aligned.indices.dtype == np.int32
    assert aligned.topk_logprobs.dtype == np.float32
    np.testing.assert_array_equal(aligned.indices, [[3, 6], [-1, -1], [4, 7]])
    np.testing.assert_allclose(aligned.topk_logprobs, [[-0.1, -2.0], [0.0, 0.0], [-0.2, -1.9]])


def test_selected_candidates_keep_large_token_ids_exact():
    token_id = 2**31
    aligned = align_student_topk([token_id], [1], [token_id], [[token_id]], [[-0.1]])

    assert aligned.indices.dtype == np.int64
    np.testing.assert_array_equal(aligned.indices, [[token_id]])


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
