"""Batch grading shared by rollout engines and training projections."""

import asyncio
import json
from typing import Any

import requests
from skyrl_gym.envs.nemotron_ultra.genrm import grade_genrm_group, response_object
from skyrl_gym.envs.nemotron_ultra.judge import OpenAIJudge
from skyrl_gym.verification import RewardResult, TrainingDisposition, VerificationResult, VerificationStatus

from skyrl_train.trajectory_runners.types import AgentLoopOutput, TrajectoryRequestBatch

GENRM_AGENTS = frozenset({"genrm_simple_agent", "genrm_simple_agent_reasoning_off"})


async def apply_genrm_cohort_rewards(
    outputs: list[AgentLoopOutput],
    input_batch: TrajectoryRequestBatch,
    config: dict[str, Any],
    judge: OpenAIJudge | None,
) -> None:
    """Replace provisional scores with comparisons within each prompt group."""
    env_extras = input_batch.get("env_extras") or []

    def ultra_at(index: int) -> dict[str, Any] | None:
        extra_info = env_extras[index].get("extra_info") if index < len(env_extras) else None
        ultra = extra_info.get("nemotron_ultra") if isinstance(extra_info, dict) else None
        return ultra if isinstance(ultra, dict) else None

    genrm_indices = [index for index in range(len(outputs)) if (ultra_at(index) or {}).get("agent") in GENRM_AGENTS]
    if not genrm_indices:
        return
    batch_metadata = input_batch.get("batch_metadata")
    if batch_metadata is not None and batch_metadata.training_phase == "eval":
        for index in genrm_indices:
            outputs[index].env_metrics["genrm/cohort_skipped_eval"] = 1.0
            if outputs[index].verification.status is not VerificationStatus.VERIFIED:
                continue
            outputs[index].verification = VerificationResult.unavailable("GenRM evaluation needs a comparison cohort")
            outputs[index].reward = RewardResult(unshaped_reward=None, optimization_reward=0.0)
            outputs[index].disposition = TrainingDisposition.mask("GenRM evaluation has no comparison cohort")
        return
    if judge is None:
        raise RuntimeError("Nemotron Ultra GenRM rows require environment.skyrl_gym.nemotron_ultra.genrm.judge")
    trajectory_ids = input_batch.get("trajectory_ids")
    if trajectory_ids is None:
        raise ValueError("GenRM cohort rewards require trajectory IDs")

    groups: dict[str, list[int]] = {}
    for index in genrm_indices:
        groups.setdefault(trajectory_ids[index].instance_id, []).append(index)
    expected_size = int(config.get("num_rollouts_per_prompt", 16))
    for indices in groups.values():
        if len(indices) != expected_size:
            raise ValueError(f"GenRM cohort requires {expected_size} rollouts for a prompt, received {len(indices)}")
        indices = [
            index
            for index in indices
            if outputs[index].disposition.loss_eligible
            and outputs[index].verification.status is VerificationStatus.VERIFIED
        ]
        if len(indices) < 2:
            for index in indices:
                outputs[index].verification = VerificationResult.unavailable("Insufficient valid GenRM peers")
                outputs[index].reward = RewardResult(unshaped_reward=None, optimization_reward=0.0)
                outputs[index].disposition = TrainingDisposition.mask("Insufficient valid GenRM peers")
            continue
        histories = [input_batch["prompts"][index] for index in indices]
        if any(history != histories[0] for history in histories):
            raise ValueError("GenRM cohort rows must share the same conversation")
        records = [json.loads((ultra_at(index) or {})["record_json"]) for index in indices]
        if not all(isinstance(record, dict) for record in records):
            raise TypeError("GenRM record_json must decode to an object")
        principles = {record.get("principle") for record in records}
        if len(principles) != 1 or None in principles:
            raise ValueError("GenRM cohort rows must agree on a non-empty principle")
        response_objects = []
        for index in indices:
            messages = outputs[index].evidence.messages
            assistant_message = next(
                (dict(message) for message in reversed(messages) if message.get("role") == "assistant"),
                {},
            )
            assistant_message["content"] = outputs[index].evidence.response or ""
            response_objects.append(response_object(assistant_message))
        try:
            rewards, metrics = await asyncio.to_thread(
                grade_genrm_group,
                conversation_history=input_batch["prompts"][indices[0]],
                response_objects=response_objects,
                principle=next(iter(principles)),
                judge=judge,
                config=config,
            )
        except (RuntimeError, ValueError, requests.RequestException) as error:
            for index in indices:
                outputs[index].verification = VerificationResult.error(
                    "GenRM comparisons failed",
                    diagnostics={"error_type": type(error).__name__, "error_message": str(error)},
                )
                outputs[index].reward = RewardResult(unshaped_reward=None, optimization_reward=0.0)
                outputs[index].disposition = TrainingDisposition.mask("GenRM comparisons failed")
                outputs[index].env_metrics["genrm/comparison_failure"] = 1.0
            continue
        for index, reward in zip(indices, rewards, strict=True):
            old_token_rewards = outputs[index].reward.token_rewards
            token_rewards = None
            if old_token_rewards is not None:
                token_rewards_list = [0.0] * len(old_token_rewards)
                credited = [position for position, value in enumerate(outputs[index].loss_mask) if value]
                if credited:
                    token_rewards_list[credited[-1]] = reward
                token_rewards = tuple(token_rewards_list)
            outputs[index].reward = RewardResult(
                unshaped_reward=reward,
                optimization_reward=reward,
                token_rewards=token_rewards,
            )
            outputs[index].verification = VerificationResult.verified(
                reward,
                diagnostics={"agent": (ultra_at(index) or {})["agent"], "genrm_metrics": metrics},
                score_min=1.0,
                score_max=5.0,
            )
            outputs[index].env_metrics.update({f"genrm/{name}": value for name, value in metrics.items()})
