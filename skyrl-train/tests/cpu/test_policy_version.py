import pytest

from skyrl_train.policy_version import PolicyVersionHistory


@pytest.mark.parametrize(
    ("first_token_ts", "lifetime", "expected"),
    [
        (15.0, (1.0, 100.0), 0),
        (20.0, (1.0, 100.0), 1),
        (35.0, (1.0, 100.0), 1),
        (5.0, (1.0, 100.0), None),
        # vLLM reports zero when no token was sampled.
        (0.0, (1.0, 100.0), None),
        (None, (1.0, 100.0), None),
        # An EngineCore on another host stamps times from a different monotonic clock.
        (5.0, (11.0, 13.0), RuntimeError),
        (90_000.0, (11.0, 13.0), RuntimeError),
    ],
)
def test_a_first_token_maps_to_the_version_live_when_it_was_sampled(first_token_ts, lifetime, expected):
    history = PolicyVersionHistory()
    history.record_resume(10.0, 0)
    history.record_resume(20.0, 1)
    submitted_at, returned_at = lifetime

    if expected is RuntimeError:
        with pytest.raises(RuntimeError, match="EngineCore on its actor's host"):
            history.version_at(first_token_ts, submitted_at=submitted_at, returned_at=returned_at)
    else:
        assert history.version_at(first_token_ts, submitted_at=submitted_at, returned_at=returned_at) == expected
