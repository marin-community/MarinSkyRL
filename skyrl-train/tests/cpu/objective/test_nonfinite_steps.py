import pytest

from skyrl_train.distributed.step_policy import NonfiniteStepPolicy, nonfinite_step_policy


@pytest.mark.parametrize(
    ("consecutive_skipped", "limit", "expected"),
    [
        (0, None, NonfiniteStepPolicy.FAIL),
        (0, 3, NonfiniteStepPolicy.SKIP),
        (2, 3, NonfiniteStepPolicy.SKIP),
        (3, 3, NonfiniteStepPolicy.FAIL),
    ],
)
def test_nonfinite_steps_skip_the_allowed_streak_then_fail(consecutive_skipped, limit, expected):
    assert nonfinite_step_policy(consecutive_skipped, limit) is expected
