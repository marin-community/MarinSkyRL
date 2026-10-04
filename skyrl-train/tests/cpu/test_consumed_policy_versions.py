import pytest

from skyrl_train.rollouts.policy_versions import consumed_policy_versions


def test_consumed_versions_preserve_mixed_weights_and_masked_assembly_tokens():
    batch = {
        "response_ids": [[10, 11, 12, 13], [20, 21]],
        "loss_masks": [[1, 1, 0, 1], [1, 0]],
        "token_policy_versions": [[2, 3, -1, 3], [1, 1]],
    }
    record, metrics = consumed_policy_versions(batch, ["a", "b"], 4)
    assert record["loss_token_publication_gap_counts"] == {"1": 2, "2": 1, "3": 1}
    assert metrics["async/token_publication_gap_mean"] == 1.75
    assert metrics["async/mixed_policy_response_fraction"] == 0.5
    assert record["rows"][0]["policy_version_spans"] == [
        {"start": 0, "end": 1, "version": 2},
        {"start": 1, "end": 2, "version": 3},
        {"start": 2, "end": 3, "version": -1},
        {"start": 3, "end": 4, "version": 3},
    ]
    # A fully masked token may lack generation provenance; a consumed token may not.
    batch["loss_masks"][0][2] = 1
    with pytest.raises(ValueError, match="unknown or future"):
        consumed_policy_versions(batch, ["a", "b"], 4)
    batch["token_policy_versions"][0][2] = 5
    with pytest.raises(ValueError, match="unknown or future"):
        consumed_policy_versions(batch, ["a", "b"], 4)
    batch["loss_masks"] = [[0, 0, 0, 0], [0, 0]]
    record, metrics = consumed_policy_versions(batch, ["a", "b"], 4)
    assert record["loss_token_publication_gap_counts"] == {}
    assert metrics["async/measured_policy_loss_tokens"] == 0.0
    assert "async/token_publication_gap_mean" not in metrics
