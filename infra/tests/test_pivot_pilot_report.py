"""Paired task bootstrap accepts all combinations of binary verifier grades."""

import pytest

from infra.rl_data.pivot_pilot_report import paired_difference, VERIFIERS


@pytest.mark.parametrize("scores", [(1, 1, 1), (1, 0, 0), (0, 0, 0)])
def test_constant_paired_difference_has_exact_interval(scores):
    reference = [
        {
            "phase": "eval",
            "status": "verified",
            "step": 20,
            "source_id": str(index),
            "task_id": str(index % 154),
            **dict.fromkeys(VERIFIERS, 0),
        }
        for index in range(256)
    ]
    candidate = [{**row, **dict(zip(VERIFIERS, scores))} for row in reference]
    result = paired_difference(candidate, reference, seed=42, samples=100)
    for verifier, score in zip(VERIFIERS, scores):
        assert result["matrix"][verifier]["accuracy_difference"] == score
        assert result["matrix"][verifier]["ci95"] == [score, score]
