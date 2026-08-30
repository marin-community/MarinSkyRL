import json

import pytest

from skyrl_gym.envs.reasoning_gym.scoring import score_response

CHAIN_SUM_GROUND_TRUTH = json.dumps(
    {
        "task": "chain_sum",
        "entry": {
            "question": "4 + 2 =",
            "answer": "6",
            "metadata": {"source_dataset": "chain_sum"},
        },
    }
)


@pytest.mark.parametrize(
    "response, expected",
    [
        ("Adding 4 and 2 gives 6.\nAnswer: 6", 1.0),
        ("6", 1.0),
        ("Answer: 7", 0.0),
        # Multiple markers: only the text after the last one is scored.
        ("Answer: 7 was wrong, let me redo it.\nAnswer: 6", 1.0),
    ],
)
def test_score_response_extracts_final_answer(response, expected):
    assert score_response(response, CHAIN_SUM_GROUND_TRUTH) == expected
