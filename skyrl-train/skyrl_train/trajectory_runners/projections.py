"""Projection of harness interaction records into trainer samples."""

import copy
from typing import Generic, Protocol, Sequence, TypeVar

from omegaconf import DictConfig

from skyrl_train.metric_names import TOKEN_PROVENANCE_RECONSTRUCTED_FRACTION_METRIC
from skyrl_gym.verification import RewardResult, TrainingDisposition
from skyrl_train.trajectory_runners.types import (
    AgentLoopOutput,
    TokenProvenance,
    TrajectoryBatch,
    TrajectoryRequestBatch,
)
from skyrl_train.trajectory_runners.trajectory_processing import (
    apply_overlong_filtering,
    get_rollout_metrics,
    minimum_captured_global_step,
)


InteractionT = TypeVar("InteractionT")


class TrainableInteraction(Protocol):
    loss_mask: list[int]
    disposition: TrainingDisposition


class RewardedInteraction(Protocol):
    reward: RewardResult


class TrajectoryProjection(Protocol, Generic[InteractionT]):
    """Convert structured interaction results into a trainer batch."""

    def project(self, outputs: InteractionT, request: TrajectoryRequestBatch) -> TrajectoryBatch: ...


class IdentityTrajectoryProjection:
    """Return a batch that a collector has already normalized."""

    def project(self, outputs: TrajectoryBatch, request: TrajectoryRequestBatch) -> TrajectoryBatch:
        return outputs


class WholeTrajectoryProjection:
    """Emit one trainer sample for each completed environment trajectory."""

    def __init__(self, runner_cfg: DictConfig, tokenizer):
        self._cfg = runner_cfg
        self._tokenizer = tokenizer

    def project(
        self,
        outputs: Sequence[AgentLoopOutput],
        request: TrajectoryRequestBatch,
    ) -> TrajectoryBatch:
        responses = [list(output.evidence.response_token_ids) for output in outputs]
        rewards = [output.reward.to_trainer_reward() for output in outputs]
        loss_masks = _loss_masks(outputs, responses, self._cfg, self._tokenizer)
        candidate_logprobs = [
            None if output.evidence.behavior_logprobs is None else list(output.evidence.behavior_logprobs)
            for output in outputs
        ]
        get_logprobs = _logprobs_requested(request, self._cfg)
        rollout_logprobs = (
            candidate_logprobs if get_logprobs and all(x is not None for x in candidate_logprobs) else None
        )

        rollout_metrics = get_rollout_metrics(
            responses,
            rewards,
            [output.env_metrics for output in outputs],
            request["env_classes"],
            successes=_verification_successes(outputs),
        )
        rollout_metrics.update(_token_provenance_metrics(outputs))
        batch = TrajectoryBatch(
            prompt_token_ids=[list(output.evidence.prompt_token_ids) for output in outputs],
            response_ids=responses,
            rewards=rewards,
            loss_masks=loss_masks,
            stop_reasons=[output.evidence.stop_reason for output in outputs],
            rollout_metrics=rollout_metrics,
            rollout_logprobs=rollout_logprobs,
            exclude_from_baseline=[not output.disposition.baseline_eligible for output in outputs],
            actual_global_step=minimum_captured_global_step(outputs),
        )
        _attach_reward_channels(batch, outputs, responses)
        return batch


class StepWiseTrajectoryProjection:
    """Emit one trainer sample for every environment transition."""

    def __init__(self, runner_cfg: DictConfig, tokenizer):
        self._cfg = runner_cfg
        self._tokenizer = tokenizer

    def project(
        self,
        outputs: Sequence[Sequence[AgentLoopOutput]],
        request: TrajectoryRequestBatch,
    ) -> TrajectoryBatch:
        trajectory_ids = request.get("trajectory_ids")
        if trajectory_ids is None:
            raise ValueError("step-wise projection requires trajectory_ids")

        steps = [step for trajectory in outputs for step in trajectory]
        responses = [list(step.evidence.response_token_ids) for step in steps]
        rewards = [step.reward.to_trainer_reward() for step in steps]
        loss_masks = _loss_masks(steps, responses, self._cfg, self._tokenizer)

        projected_ids = []
        is_last_step = []
        for trajectory_id, trajectory in zip(trajectory_ids, outputs):
            for step_index in range(len(trajectory)):
                projected_id = copy.deepcopy(trajectory_id)
                projected_id.step = step_index
                projected_ids.append(projected_id)
                is_last_step.append(step_index == len(trajectory) - 1)

        get_logprobs = _logprobs_requested(request, self._cfg)
        rollout_logprobs = (
            [
                None if step.evidence.behavior_logprobs is None else list(step.evidence.behavior_logprobs)
                for step in steps
            ]
            if get_logprobs
            else None
        )

        rollout_metrics = get_rollout_metrics(responses, rewards, successes=_verification_successes(steps))
        rollout_metrics.update(_token_provenance_metrics(steps))
        batch = TrajectoryBatch(
            prompt_token_ids=[list(step.evidence.prompt_token_ids) for step in steps],
            response_ids=responses,
            rewards=rewards,
            loss_masks=loss_masks,
            stop_reasons=[step.evidence.stop_reason for step in steps],
            rollout_metrics=rollout_metrics,
            rollout_logprobs=rollout_logprobs,
            trajectory_ids=projected_ids,
            is_last_step=is_last_step,
            exclude_from_baseline=[not step.disposition.baseline_eligible for step in steps],
            actual_global_step=minimum_captured_global_step(steps),
        )
        _attach_reward_channels(batch, steps, responses)
        return batch


def _logprobs_requested(request: TrajectoryRequestBatch, runner_cfg: DictConfig) -> bool:
    sampling_params = request.get("sampling_params")
    if sampling_params is not None:
        return sampling_params.get("logprobs") is not None
    return runner_cfg.sampling_params.logprobs is not None


def _loss_masks(outputs, responses, runner_cfg: DictConfig, tokenizer):
    loss_masks = [project_loss_mask(output, response) for output, response in zip(outputs, responses)]
    if runner_cfg.apply_overlong_filtering:
        return apply_overlong_filtering(loss_masks, responses, tokenizer.eos_token_id)
    return loss_masks


def project_loss_mask(output: TrainableInteraction, response: Sequence[int]) -> list[int]:
    """Zero the aligned trainer mask when the disposition rejects training."""
    return output.loss_mask if output.disposition.loss_eligible else [0] * len(response)


def _attach_reward_channels(
    batch: TrajectoryBatch,
    outputs: Sequence[RewardedInteraction],
    responses: Sequence[Sequence[int]],
) -> None:
    unshaped_rewards = [output.reward.unshaped_reward for output in outputs]
    attach_unshaped_rewards(batch, unshaped_rewards)

    token_credit = [output.reward.token_credit for output in outputs]
    if any(credit is not None for credit in token_credit):
        batch["token_level_shaping"] = [
            list(credit) if credit is not None else [0.0] * len(response)
            for credit, response in zip(token_credit, responses)
        ]


def attach_unshaped_rewards(batch: TrajectoryBatch, rewards: Sequence[float | None]) -> None:
    """Project a complete raw-reward channel onto the trainer transport."""
    if all(reward is not None for reward in rewards):
        batch["unshaped_rewards"] = [float(reward) for reward in rewards if reward is not None]


def _token_provenance_metrics(outputs: Sequence[AgentLoopOutput]) -> dict[str, float]:
    reconstructed = sum(output.token_provenance == TokenProvenance.RECONSTRUCTED for output in outputs)
    return {TOKEN_PROVENANCE_RECONSTRUCTED_FRACTION_METRIC: reconstructed / len(outputs) if outputs else 0.0}


def _verification_successes(outputs: Sequence[AgentLoopOutput]) -> list[bool]:
    return [
        output.verification.passed
        if output.verification.passed is not None
        else output.verification.score is not None and output.verification.score > 0.0
        for output in outputs
    ]
