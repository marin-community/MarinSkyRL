"""Check exact token provenance and the additive mismatch decomposition."""

import pytest
import torch

from skyrl_train.utils.mismatch_decomposition import mismatch_decomposition_record


def _probe(**changes):
    inputs = {
        "response_ids": [[3, 4, 5], [6, 7]],
        "version_rows": [
            [
                {"start": 0, "token_count": 2, "policy_version": 0},
                {"start": 2, "token_count": 1, "policy_version": 1},
            ],
            [
                {"start": 0, "token_count": 1, "policy_version": 0},
                {"start": 1, "token_count": 1, "policy_version": 1},
            ],
        ],
        "sequences": torch.tensor([[0, 9, 3, 4, 5], [8, 9, 6, 7, 0]]),
        "attention_mask": torch.tensor([[0, 1, 1, 1, 1], [1, 1, 1, 1, 0]]),
        "response_mask": torch.tensor([[1, 1, 1], [1, 1, 0]]),
        "loss_mask": torch.tensor([[1, 1, 1], [1, 1, 0]]),
        "behavior_logprobs": torch.tensor([[-1.0, -2.0, -3.0], [-1.0, -2.0, 0.0]], dtype=torch.float64),
        "reference_logprobs": torch.tensor([[-0.9, -1.8, -2.9], [-0.9, -1.9, 0.0]], dtype=torch.float64),
        "current_logprobs": torch.tensor([[-0.95, -1.9, -2.8], [-0.95, -1.8, 0.0]], dtype=torch.float64),
        "consuming_step": 3,
        "reference_version": 0,
        "tis_cap": 1.05,
        "sample_rows": 2,
    }
    return mismatch_decomposition_record(**(inputs | changes))


def test_mismatch_decomposition_selects_exact_version_spans_and_reconstructs_gap():
    record = _probe()
    assert record["selected_tokens"] == 5
    assert record["schema_version"] == 2
    assert record["reference_tokens"] == 3
    assert record["fresh_version_tokens"] == 0
    assert record["matched_tokens"] == 3
    assert record["other_version_tokens"] == 2
    assert record["mixed_version_responses"] == 2
    assert record["summaries"]["age_2"]["tokens"] == 3
    assert record["summaries"]["all"]["engine"]["log_ratio_abs_mean"] == pytest.approx(0.4 / 3)
    assert record["summaries"]["all"]["stale"]["log_ratio_abs_mean"] == pytest.approx(0.2 / 3)
    assert record["summaries"]["all"]["combined"]["log_ratio_abs_mean"] == pytest.approx(0.2 / 3)
    assert record["summaries"]["all"]["reconstruction_abs_max"] < 1e-12
    assert record["summaries"]["all"]["opposite_sign_fraction"] == 1.0
    assert record["summaries"]["all"]["canceled_absolute_mean"] == pytest.approx(2 * (0.05 + 0.1 + 0.05) / 3)
    assert {position for row in record["samples"] for position in row["positions"]} == {0, 1}
    for row in record["samples"]:
        for a, b, c in zip(row["A_inference"], row["B_generating_trainer"], row["C_consumer_trainer"], strict=True):
            assert (b - a) + (c - b) == pytest.approx(c - a)


def test_mismatch_decomposition_same_weights_puts_stale_term_at_zero():
    reference = torch.tensor([[-0.9, -1.8, -2.9], [-0.9, -1.9, 0.0]], dtype=torch.float64)
    record = _probe(current_logprobs=reference)
    assert record["summaries"]["all"]["stale"]["log_ratio_abs_max"] == 0.0
    assert record["summaries"]["all"]["engine"]["log_ratio_abs_mean"] > 0


def test_mismatch_decomposition_scores_fresh_span_with_current_generating_weights():
    record = _probe(
        version_rows=[
            [
                {"start": 0, "token_count": 2, "policy_version": 0},
                {"start": 2, "token_count": 1, "policy_version": 2},
            ],
            [
                {"start": 0, "token_count": 1, "policy_version": 0},
                {"start": 1, "token_count": 1, "policy_version": 2},
            ],
        ]
    )
    assert record["reference_tokens"] == 3
    assert record["fresh_version_tokens"] == 2
    assert record["matched_tokens"] == record["selected_tokens"] == 5
    assert record["other_version_tokens"] == 0
    assert record["mixed_version_responses"] == 2
    assert record["summaries"]["age_0"]["tokens"] == 2
    assert record["summaries"]["age_0"]["stale"]["log_ratio_abs_max"] == 0
    assert record["summaries"]["all"]["reconstruction_abs_max"] == 0
    for row in record["samples"]:
        fresh_position = 2 if row["row"] == 0 else 1
        sample_index = row["positions"].index(fresh_position)
        assert row["B_generating_trainer"][sample_index] == row["C_consumer_trainer"][sample_index]
        assert row["B_source"][sample_index] == "fresh_consuming_policy"


def test_mismatch_decomposition_rejects_token_id_shift():
    with pytest.raises(ValueError, match="sequence IDs"):
        _probe(sequences=torch.tensor([[0, 9, 4, 3, 5], [8, 9, 6, 7, 0]]))
