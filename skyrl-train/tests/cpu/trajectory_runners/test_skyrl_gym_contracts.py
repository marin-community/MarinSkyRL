from skyrl_gym.verification import RewardResult, VerificationResult, VerificationStatus
from skyrl_train.trajectory_runners.skyrl_gym_contracts import (
    fold_verification_results,
    reward_from_env_step,
    verification_from_env_step,
)


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
    native_reward = RewardResult(unshaped_reward=1.0, optimization_reward=0.25, components={"length": -0.75})
    step = {
        "reward": 0.25,
        "verification": native,
        "reward_result": native_reward,
        "metadata": {"length_shaped": True},
    }

    verification = verification_from_env_step(step)
    reward = reward_from_env_step(step, verification)

    assert verification is native
    assert reward is native_reward


def test_missing_legacy_reward_is_not_a_zero_verdict():
    verification = verification_from_env_step({"reward": None, "metadata": {}})

    assert verification.status is VerificationStatus.UNAVAILABLE
    assert verification.score is None


def test_multiturn_verification_averages_scored_turns_and_requires_all_to_pass():
    results = [
        VerificationResult.verified(5.0, passed=True, score_range=(1.0, 5.0)),
        VerificationResult.unavailable("tool turn has no verdict"),
        VerificationResult.verified(0.0, passed=False),
    ]

    verification, outcome = fold_verification_results(results)

    assert outcome == 0.5
    assert verification.score == 0.5
    assert verification.passed is False
    assert verification.diagnostics["num_scored_steps"] == 2
