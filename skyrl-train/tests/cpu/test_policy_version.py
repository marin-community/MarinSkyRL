import pytest

from skyrl_train.policy_version import PolicyVersionHistory


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


def test_a_first_token_inside_the_requests_lifetime_shares_the_actors_clock():
    history = PolicyVersionHistory()
    history.record_resume(10.0, 0)

    history.check_same_clock(12.0, submitted_at=11.0, returned_at=13.0)
    # vLLM reports zero when no token was sampled.
    history.check_same_clock(0.0, submitted_at=11.0, returned_at=13.0)


@pytest.mark.parametrize("first_token_ts", [5.0, 90_000.0])
def test_a_first_token_outside_the_requests_lifetime_is_another_hosts_clock(first_token_ts):
    history = PolicyVersionHistory()
    history.record_resume(10.0, 0)

    with pytest.raises(RuntimeError, match="EngineCore on the engine actor's host"):
        history.check_same_clock(first_token_ts, submitted_at=11.0, returned_at=13.0)


def test_the_clock_is_not_checked_until_a_version_is_installed():
    PolicyVersionHistory().check_same_clock(90_000.0, submitted_at=11.0, returned_at=13.0)
