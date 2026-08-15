"""Adapters between SkyRL-Gym environments and shared verifier contracts."""

from collections.abc import Sequence
from typing import Any

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from skyrl_gym.verification import RewardResult, RolloutEvidence, VerificationResult


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
        behavior_logprobs=None if behavior_logprobs is None else tuple(behavior_logprobs),
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

    outcome = sum(result.score for result in results if result.score is not None)
    return (
        VerificationResult.verified(
            outcome,
            diagnostics={"steps": tuple(result.diagnostics for result in results)},
        ),
        outcome,
    )
