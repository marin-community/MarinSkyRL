import pytest

from skyrl_train.distributed.megatron.nonfinite_steps import NonfiniteStepAction, nonfinite_step_action


@pytest.mark.parametrize(("norm", "found_inf"), [(float("nan"), False), (float("inf"), False), (2.0, True)])
def test_nonfinite_steps_skip_the_allowed_streak_then_fail(norm, found_inf):
    assert nonfinite_step_action(norm, found_inf, 0, None) is NonfiniteStepAction.FAIL
    for streak in range(3):
        assert nonfinite_step_action(norm, found_inf, streak, 3) is NonfiniteStepAction.SKIP
    assert nonfinite_step_action(norm, found_inf, 3, 3) is NonfiniteStepAction.FAIL
    assert nonfinite_step_action(2.0, False, 3, 3) is NonfiniteStepAction.APPLY
