import pytest

from skyrl_train.policy_version import PolicyVersionHistory, consumed_token_age_metrics


def test_first_token_maps_to_the_version_live_when_it_was_sampled():
    history = PolicyVersionHistory()
    history.record_resume(10.0, 0)
    history.record_resume(20.0, 1)

    assert history.at_first_token(10.0) == 0
    assert history.at_first_token(19.9) == 0
    assert history.at_first_token(20.0) == 1
    assert history.at_first_token(35.0) == 1


def test_first_token_before_any_boundary_has_no_version():
    history = PolicyVersionHistory()
    assert history.at_first_token(5.0) is None
    history.record_resume(10.0, 0)
    assert history.at_first_token(9.0) is None
    # vLLM reports zero when no token was ever sampled.
    assert history.at_first_token(0.0) is None
    assert history.at_first_token(None) is None


@pytest.mark.parametrize("boundary, version", [(10.0, 0), (5.0, 2), (12.0, 0)])
def test_boundaries_and_versions_must_not_move_backwards(boundary, version):
    history = PolicyVersionHistory()
    history.record_resume(10.0, 1)
    with pytest.raises(ValueError):
        history.record_resume(boundary, version)


def test_consumed_token_age_uses_each_trainable_version_span():
    metrics = consumed_token_age_metrics(
        response_ids=[[11, 12, 13, 14, 15], [21, 22]],
        loss_masks=[[1, 1, 0, 1, 1], [0, 1]],
        version_rows=[
            [
                {"start": 0, "token_count": 3, "policy_version": 6},
                {"start": 3, "token_count": 2, "policy_version": 9},
            ],
            [{"start": 1, "token_count": 1, "policy_version": 10}],
        ],
        consuming_step=11,
    )
    assert metrics["async/consumed_token_age_mean"] == 2.0
    assert metrics["async/consumed_token_age_p90"] == 4.0
    assert metrics["async/consumed_token_age_max"] == 4.0
    assert metrics["async/consumed_token_age_stale_fraction"] == 0.8
    assert metrics["async/consumed_token_age_at_least_four_fraction"] == 0.4
    assert metrics["async/consumed_token_age_tokens"] == 5.0


def test_consumed_token_age_rejects_unversioned_trainable_token():
    with pytest.raises(ValueError, match="known installed policy version"):
        consumed_token_age_metrics(
            response_ids=[[11, 12]],
            loss_masks=[[1, 1]],
            version_rows=[
                [
                    {"start": 0, "token_count": 1, "policy_version": 0},
                    {"start": 1, "token_count": 1, "policy_version": None},
                ]
            ],
            consuming_step=1,
        )
