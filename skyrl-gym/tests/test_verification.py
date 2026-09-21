import math

import pytest

from skyrl_gym.verification import (
    RewardResult,
    RolloutEvidence,
    TrainingDisposition,
    VerificationResult,
    VerificationStatus,
)


def test_verified_zero_is_distinct_from_unavailable():
    verified = VerificationResult.verified(0.0, passed=False)
    unavailable = VerificationResult.unavailable("verifier did not return a result")

    assert verified.status is VerificationStatus.VERIFIED
    assert verified.score == 0.0
    assert unavailable.status is VerificationStatus.UNAVAILABLE
    assert unavailable.score is None


@pytest.mark.parametrize("score", [math.nan, math.inf, -math.inf])
def test_verification_rejects_non_finite_scores(score):
    with pytest.raises(ValueError, match="score must be finite"):
        VerificationResult.verified(score)


def test_unavailable_verification_cannot_carry_a_score():
    with pytest.raises(ValueError, match="cannot carry a verifier verdict"):
        VerificationResult(status=VerificationStatus.UNAVAILABLE, score=0.0, reason="missing")


def test_reward_channels_validate_against_rollout_evidence():
    evidence = RolloutEvidence(response_token_ids=(11, 12, 13))
    reward = RewardResult(unshaped_reward=1.0, optimization_reward=0.5, token_credit=(0.0, 0.5))

    with pytest.raises(ValueError, match="token_credit must align"):
        reward.validate_for(evidence)


def test_rollout_evidence_rejects_misaligned_behavior_logprobs():
    with pytest.raises(ValueError, match="behavior_logprobs must align"):
        RolloutEvidence(response_token_ids=(11, 12), behavior_logprobs=(-0.1,))


def test_rollout_evidence_accepts_response_or_full_prefix_routes():
    row = ((1, 2),)
    response = (11, 12)
    prompt = (8, 9, 10)

    # These two accepted lengths are the response-only and full-prefix wire contracts.
    RolloutEvidence(prompt_token_ids=prompt, response_token_ids=response, routed_experts=(row,) * 2)
    RolloutEvidence(prompt_token_ids=prompt, response_token_ids=response, routed_experts=(row,) * 4)
    with pytest.raises(ValueError, match="full-prefix prediction positions"):
        RolloutEvidence(prompt_token_ids=prompt, response_token_ids=response, routed_experts=(row,) * 3)


def test_rollout_evidence_requires_aligned_student_topk_pairs():
    with pytest.raises(ValueError, match="provided together"):
        RolloutEvidence(response_token_ids=(11,), student_topk_indices=((2, 3),))
    with pytest.raises(ValueError, match="align with response_token_ids"):
        RolloutEvidence(
            response_token_ids=(11, 12),
            student_topk_indices=((2, 3),),
            behavior_topk_logprobs=((-0.1, -0.2),),
        )
    with pytest.raises(ValueError, match="matching widths"):
        RolloutEvidence(
            response_token_ids=(11,),
            student_topk_indices=((2, 3),),
            behavior_topk_logprobs=((-0.1,),),
        )


def test_mask_disposition_excludes_loss_and_baseline():
    infrastructure_failure = TrainingDisposition.mask(
        "infrastructure failure",
        exception_type="OrchestratorFailure",
    )

    assert not infrastructure_failure.baseline_eligible
    assert not infrastructure_failure.loss_eligible
    assert infrastructure_failure.exception_type == "OrchestratorFailure"
