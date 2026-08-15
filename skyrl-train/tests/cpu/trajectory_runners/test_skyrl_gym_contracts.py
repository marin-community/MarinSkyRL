from skyrl_gym.verification import VerificationResult, VerificationStatus
from skyrl_train.trajectory_runners.skyrl_gym_contracts import reward_from_env_step, verification_from_env_step


def test_legacy_environment_reward_adapts_to_verified_outcome():
    step = {"reward": -1.0, "metadata": {"answer": "wrong"}}

    verification = verification_from_env_step(step)
    reward = reward_from_env_step(step, verification)

    assert verification.status is VerificationStatus.VERIFIED
    assert verification.score == -1.0
    assert reward.unshaped_reward == -1.0
    assert reward.optimization_reward == -1.0


def test_native_verification_stays_separate_from_shaped_reward():
    native = VerificationResult.verified(1.0, passed=True, diagnostics={"answer": "42"})
    step = {"reward": 0.25, "verification": native, "metadata": {"length_shaped": True}}

    verification = verification_from_env_step(step)
    reward = reward_from_env_step(step, verification)

    assert verification is native
    assert reward.unshaped_reward == 1.0
    assert reward.optimization_reward == 0.25


def test_missing_legacy_reward_is_not_a_zero_verdict():
    verification = verification_from_env_step({"reward": None, "metadata": {}})

    assert verification.status is VerificationStatus.UNAVAILABLE
    assert verification.score is None
