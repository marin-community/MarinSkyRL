"""Projection of harness interaction records into trainer samples."""

import copy
from typing import Generic, Protocol, Sequence, TypeVar

from omegaconf import DictConfig

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
        responses = [output.response_ids for output in outputs]
        rewards = [output.reward for output in outputs]
        loss_masks = _loss_masks(outputs, responses, self._cfg, self._tokenizer)
        candidate_logprobs = [output.rollout_logprobs for output in outputs]
        get_logprobs = _logprobs_requested(request, self._cfg)
        rollout_logprobs = (
            candidate_logprobs if get_logprobs and all(x is not None for x in candidate_logprobs) else None
        )

        rollout_metrics = get_rollout_metrics(
            responses,
            rewards,
            [output.env_metrics for output in outputs],
            request["env_classes"],
        )
        rollout_metrics.update(_token_provenance_metrics(outputs))
        return TrajectoryBatch(
            prompt_token_ids=[output.prompt_ids for output in outputs],
            response_ids=responses,
            rewards=rewards,
            loss_masks=loss_masks,
            stop_reasons=[output.stop_reason for output in outputs],
            rollout_metrics=rollout_metrics,
            rollout_logprobs=rollout_logprobs,
            actual_global_step=minimum_captured_global_step(outputs),
        )


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
        responses = [step.response_ids for step in steps]
        rewards = [step.reward for step in steps]
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
        rollout_logprobs = [step.rollout_logprobs for step in steps] if get_logprobs else None

        rollout_metrics = get_rollout_metrics(responses, rewards)
        rollout_metrics.update(_token_provenance_metrics(steps))
        return TrajectoryBatch(
            prompt_token_ids=[step.prompt_ids for step in steps],
            response_ids=responses,
            rewards=rewards,
            loss_masks=loss_masks,
            stop_reasons=[step.stop_reason for step in steps],
            rollout_metrics=rollout_metrics,
            rollout_logprobs=rollout_logprobs,
            trajectory_ids=projected_ids,
            is_last_step=is_last_step,
            actual_global_step=minimum_captured_global_step(steps),
        )


def _logprobs_requested(request: TrajectoryRequestBatch, runner_cfg: DictConfig) -> bool:
    sampling_params = request.get("sampling_params")
    if sampling_params is not None:
        return sampling_params.get("logprobs") is not None
    return runner_cfg.sampling_params.logprobs is not None


def _loss_masks(outputs, responses, runner_cfg: DictConfig, tokenizer):
    loss_masks = [output.loss_mask for output in outputs]
    if runner_cfg.apply_overlong_filtering:
        return apply_overlong_filtering(loss_masks, responses, tokenizer.eos_token_id)
    return loss_masks


def _token_provenance_metrics(outputs: Sequence[AgentLoopOutput]) -> dict[str, float]:
    reconstructed = sum(output.token_provenance == TokenProvenance.RECONSTRUCTED for output in outputs)
    return {"generate/token_provenance/reconstructed_fraction": reconstructed / len(outputs) if outputs else 0.0}
