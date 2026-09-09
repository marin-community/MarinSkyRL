import math

import pytest

from skyrl_train.weight_sync.publication_version import PublicationVersionHistory


def test_first_token_clock_not_output_delivery_selects_publication():
    history = PublicationVersionHistory()
    history.record_resume(100.0, 2)
    history.record_resume(200.0, 3)
    assert history.at_first_token(150.0) == 2
    assert history.at_first_token(200.0) == 3
    assert history.at_first_token(250.0) == 3
    assert history.at_first_token(99.0) is None


@pytest.mark.parametrize("timestamp", [None, 0.0, -1.0, math.nan, math.inf])
def test_missing_or_invalid_first_token_keeps_submission_stamp(timestamp):
    history = PublicationVersionHistory([(100.0, 2)])
    assert history.at_first_token(timestamp) is None


@pytest.mark.parametrize("boundary,version", [(99.0, 3), (101.0, 1), (math.nan, 3), (101.0, True)])
def test_invalid_boundary_or_version_cannot_relabel_outputs(boundary, version):
    history = PublicationVersionHistory([(100.0, 2)])
    with pytest.raises(ValueError):
        history.record_resume(boundary, version)


@pytest.mark.parametrize(
    "rows,versions,expected",
    [
        ([[1], [2]], [4, 2], 2),
        ([[1], []], [2, 9], 2),
        ([[], [2]], [9, 3], 3),
        ([[1], [2]], [None, 3], None),
        ([[], []], [9, 10], None),
    ],
)
def test_only_emitted_tokens_contribute_to_earliest_version(rows, versions, expected):
    from skyrl_train.weight_sync.publication_version import earliest_sampled_policy_version

    assert earliest_sampled_policy_version(rows, versions) == expected


@pytest.mark.parametrize("rows,versions", [([[1]], []), ([[1]], [True]), ([[1]], [-1])])
def test_malformed_first_token_evidence_fails(rows, versions):
    from skyrl_train.weight_sync.publication_version import earliest_sampled_policy_version

    with pytest.raises(ValueError):
        earliest_sampled_policy_version(rows, versions)
