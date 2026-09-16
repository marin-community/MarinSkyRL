"""Behavior-policy response top-K capture contracts."""

from types import SimpleNamespace

import pytest

from skyrl_train.inference_engines.response_topk import select_chat_response_topk, select_response_topk


def test_selected_token_outside_topk_is_not_forced_into_student_candidates():
    # vLLM includes the sampled token in addition to the requested top K.
    ids, scores = select_response_topk(
        {9: SimpleNamespace(logprob=-7.0), 3: SimpleNamespace(logprob=-0.2), 2: SimpleNamespace(logprob=-0.1)},
        2,
    )
    assert ids == [2, 3]
    assert scores == [-0.1, -0.2]


def test_tied_candidates_use_stable_token_id_order():
    assert select_response_topk({8: -0.5, 4: -0.5}, 2) == ([4, 8], [-0.5, -0.5])


@pytest.mark.parametrize("scores", [{3: float("nan")}, {3: 0.1}, {-1: -0.2}, {"3": -0.2}])
def test_invalid_scores_fail_closed(scores):
    with pytest.raises(ValueError):
        select_response_topk(scores, 1)


def test_missing_candidates_fail_closed():
    with pytest.raises(ValueError, match="fewer response candidates"):
        select_response_topk({2: -0.1}, 2)


def test_chat_candidates_use_exact_token_ids_and_discard_extra_sampled_token():
    ids, scores = select_chat_response_topk(
        [
            {"token": "token_id:9", "logprob": -7.0},
            {"token": "token_id:3", "logprob": -0.2},
            {"token": "token_id:2", "logprob": -0.1},
        ],
        2,
    )
    assert ids == [2, 3]
    assert scores == [-0.1, -0.2]


def test_chat_candidates_reject_decoded_strings():
    with pytest.raises(ValueError, match="exact token IDs"):
        select_chat_response_topk([{"token": "hello", "logprob": -0.1}], 1)
