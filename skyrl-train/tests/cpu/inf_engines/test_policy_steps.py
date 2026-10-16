from skyrl_train.inference_engines.base import UNSTAMPED_POLICY_STEP
from skyrl_train.inference_engines.policy_steps import PolicyStepStamps


def test_policy_step_stamps_follow_the_step_installed_before_each_engine_step():
    stamps = PolicyStepStamps()
    stamps.observe("a", 1, 10.0)
    stamps.install(1, 11.0)
    stamps.observe("a", 2, 11.5)
    stamps.install(2, 12.0)
    # Sampled before the second install and delivered after it: the step timestamp decides, not the arrival.
    stamps.observe("a", 1, 11.9)
    stamps.observe("a", 1, 12.5)

    assert stamps.finish("a", 5) == [UNSTAMPED_POLICY_STEP, 1, 1, 1, 2]
