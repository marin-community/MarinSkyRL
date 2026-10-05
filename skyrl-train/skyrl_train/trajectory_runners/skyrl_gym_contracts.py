"""Adapters between SkyRL-Gym environments and shared verifier contracts."""

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from skyrl_gym.verification import (
    VERIFIER_RUNTIME_ERROR,
    RewardResult,
    RolloutEvidence,
    TrainingDisposition,
    VerificationResult,
    VerificationStatus,
    normalized_verifier_score,
)
from skyrl_train.trajectory_runners.types import AgentLoopOutput


def with_validated_reward(
    output: AgentLoopOutput,
    *,
    unshaped_reward: float | None,
    optimization_reward: float,
    token_rewards: tuple[float, ...] | None,
) -> AgentLoopOutput:
    """Return a rollout with valid reward channels while preserving existing masked or skipped verdicts."""
    if not output.disposition.loss_eligible or output.verification.status is VerificationStatus.SKIPPED:
        return replace(
            output,
            reward=RewardResult(
                unshaped_reward=unshaped_reward,
                optimization_reward=0.0,
                token_rewards=None if token_rewards is None else tuple(0.0 for _ in output.evidence.response_token_ids),
            ),
        )
    try:
        if not output.evidence.response_token_ids or not any(output.loss_mask):
            raise ValueError("reward placement requires a response action token")
        reward = RewardResult(
            unshaped_reward=unshaped_reward,
            optimization_reward=optimization_reward,
            token_rewards=token_rewards,
        )
        reward.validate_for(output.evidence)
    except ValueError as error:
        return replace(
            output,
            verification=VerificationResult.error(
                "invalid reward channels",
                diagnostics={"error_type": type(error).__name__, "error_message": str(error)},
            ),
            reward=RewardResult(
                unshaped_reward=None,
                optimization_reward=0.0,
                token_rewards=None if token_rewards is None else tuple(0.0 for _ in output.evidence.response_token_ids),
            ),
            disposition=TrainingDisposition.mask("invalid reward channels", exception_type=VERIFIER_RUNTIME_ERROR),
            env_metrics={**output.env_metrics, "verifier_error": 1.0},
        )
    return replace(output, reward=reward)


def verification_from_env_step(step_output: BaseTextEnvStepOutput) -> VerificationResult:
    """Preserve a native verifier result or adapt a legacy scalar reward."""
    verification = step_output.get("verification")
    if verification is not None:
        if not isinstance(verification, VerificationResult):
            raise TypeError("environment verification must be a VerificationResult")
        return verification

    reward = step_output.get("reward")
    if reward is None:
        return VerificationResult.unavailable("environment step produced no verifier verdict")
    return VerificationResult.verified(reward, diagnostics=step_output.get("metadata", {}))


def reward_from_env_step(
    step_output: BaseTextEnvStepOutput,
    verification: VerificationResult,
    *,
    optimization_reward: float | None = None,
) -> RewardResult:
    """Keep verifier outcome separate from the environment's optimization reward."""
    native_reward = step_output.get("reward_result")
    if native_reward is not None:
        if not isinstance(native_reward, RewardResult):
            raise TypeError("environment reward_result must be a RewardResult")
        return native_reward
    if optimization_reward is None:
        optimization_reward = step_output["reward"]
    if optimization_reward is None:
        raise ValueError("a trainer reward is required after environment verification")
    return RewardResult(
        unshaped_reward=verification.score,
        optimization_reward=optimization_reward,
    )


def environment_metrics_from_step(
    step_output: BaseTextEnvStepOutput,
    episode_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Combine terminal environment metrics with the current step diagnostics."""
    return {**episode_metrics, **step_output["metadata"]}


def publish_rollout_evidence(
    env: BaseTextEnv,
    *,
    response: str,
    stop_reason: str,
    response_token_ids: Sequence[int],
    prompt_token_ids: Sequence[int] | None = None,
    behavior_logprobs: Sequence[float] | None = None,
    messages: Sequence[dict[str, Any]] = (),
    metadata: dict[str, Any] | None = None,
) -> RolloutEvidence:
    """Build and publish one model turn's evidence to a gym environment."""
    evidence = RolloutEvidence(
        messages=tuple(messages),
        response=response,
        stop_reason=stop_reason,
        generated_token_count=len(response_token_ids),
        prompt_token_ids=() if prompt_token_ids is None else tuple(prompt_token_ids),
        response_token_ids=tuple(response_token_ids),
        behavior_logprobs=None if behavior_logprobs is None else np.asarray(behavior_logprobs, dtype=np.float32),
        metadata={} if metadata is None else metadata,
    )
    env.set_rollout_evidence(evidence)
    return evidence


def fold_verification_results(
    results: Sequence[VerificationResult],
) -> tuple[VerificationResult, float | None]:
    """Fold per-turn verdicts into the terminal trajectory verdict and outcome."""
    if not results:
        return VerificationResult.unavailable("environment produced no verification result"), None
    if len(results) == 1:
        return results[0], results[0].score
    if results[-1].score is None:
        return results[-1], None

    scored = [result for result in results if result.score is not None]
    outcome = sum(normalized_verifier_score(result) for result in scored) / len(scored)
    passed = all(
        result.passed if result.passed is not None else normalized_verifier_score(result) >= 1.0 for result in scored
    )
    return (
        VerificationResult.verified(
            outcome,
            passed=passed,
            diagnostics={"steps": tuple(result.diagnostics for result in results), "num_scored_steps": len(scored)},
        ),
        outcome,
    )
