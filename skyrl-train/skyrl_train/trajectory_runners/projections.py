"""Projection of harness interaction records into trainer samples."""

import copy
from typing import Protocol, Sequence

from omegaconf import DictConfig

from skyrl_train.trajectory_runners.types import AgentLoopOutput, TrajectoryBatch, TrajectoryRequestBatch
from skyrl_train.trajectory_runners.utils import (
    apply_overlong_filtering,
    get_rollout_metrics,
    minimum_captured_global_step,
)


class TrajectoryProjection(Protocol):
    """Convert structured interaction results into a trainer batch."""

    step_wise: bool

    def project(
        self,
        outputs: Sequence[AgentLoopOutput] | Sequence[Sequence[AgentLoopOutput]],
        request: TrajectoryRequestBatch,
    ) -> TrajectoryBatch: ...


class WholeTrajectoryProjection:
    """Emit one trainer sample for each completed environment trajectory."""

    step_wise = False

    def __init__(self, runner_cfg: DictConfig, tokenizer):
        self._cfg = runner_cfg
        self._tokenizer = tokenizer

    def project(
        self,
        outputs: Sequence[AgentLoopOutput] | Sequence[Sequence[AgentLoopOutput]],
        request: TrajectoryRequestBatch,
    ) -> TrajectoryBatch:
        trajectories = list(outputs)
        if any(isinstance(output, (list, tuple)) for output in trajectories):
            raise TypeError("WholeTrajectoryProjection requires one interaction result per trajectory")

        typed_outputs: list[AgentLoopOutput] = trajectories  # type: ignore[assignment]
        responses = [output.response_ids for output in typed_outputs]
        rewards = [output.reward for output in typed_outputs]
        loss_masks = [output.loss_mask for output in typed_outputs]
        if self._cfg.apply_overlong_filtering:
            loss_masks = apply_overlong_filtering(loss_masks, responses, self._tokenizer.eos_token_id)

        sampling_params = request.get("sampling_params")
        get_logprobs = (
            sampling_params.get("logprobs") is not None
            if sampling_params is not None
            else self._cfg.sampling_params.logprobs is not None
        )
        candidate_logprobs = [output.rollout_logprobs for output in typed_outputs]
        rollout_logprobs = candidate_logprobs if get_logprobs and all(x is not None for x in candidate_logprobs) else None

        return TrajectoryBatch(
            prompt_token_ids=[output.prompt_ids for output in typed_outputs],
            response_ids=responses,
            rewards=rewards,
            loss_masks=loss_masks,
            stop_reasons=[output.stop_reason for output in typed_outputs],
            rollout_metrics=get_rollout_metrics(
                responses,
                rewards,
                [output.env_metrics for output in typed_outputs],
                request["env_classes"],
            ),
            rollout_logprobs=rollout_logprobs,
            actual_global_step=minimum_captured_global_step(typed_outputs),
        )


class StepWiseTrajectoryProjection:
    """Emit one trainer sample for every environment transition."""

    step_wise = True

    def __init__(self, runner_cfg: DictConfig, tokenizer):
        self._cfg = runner_cfg
        self._tokenizer = tokenizer

    def project(
        self,
        outputs: Sequence[AgentLoopOutput] | Sequence[Sequence[AgentLoopOutput]],
        request: TrajectoryRequestBatch,
    ) -> TrajectoryBatch:
        trajectories = list(outputs)
        if any(not isinstance(output, (list, tuple)) for output in trajectories):
            raise TypeError("StepWiseTrajectoryProjection requires step records grouped by trajectory")
        trajectory_ids = request.get("trajectory_ids")
        if trajectory_ids is None:
            raise ValueError("step-wise projection requires trajectory_ids")

        grouped_outputs: list[Sequence[AgentLoopOutput]] = trajectories  # type: ignore[assignment]
        steps = [step for trajectory in grouped_outputs for step in trajectory]
        responses = [step.response_ids for step in steps]
        rewards = [step.reward for step in steps]
        loss_masks = [step.loss_mask for step in steps]
        if self._cfg.apply_overlong_filtering:
            loss_masks = apply_overlong_filtering(loss_masks, responses, self._tokenizer.eos_token_id)

        projected_ids = []
        is_last_step = []
        for trajectory_id, trajectory in zip(trajectory_ids, grouped_outputs):
            for step_index in range(len(trajectory)):
                projected_id = copy.deepcopy(trajectory_id)
                projected_id.step = step_index
                projected_ids.append(projected_id)
                is_last_step.append(step_index == len(trajectory) - 1)

        sampling_params = request.get("sampling_params")
        get_logprobs = (
            sampling_params.get("logprobs") is not None
            if sampling_params is not None
            else self._cfg.sampling_params.logprobs is not None
        )
        rollout_logprobs = [step.rollout_logprobs for step in steps] if get_logprobs else None

        return TrajectoryBatch(
            prompt_token_ids=[step.prompt_ids for step in steps],
            response_ids=responses,
            rewards=rewards,
            loss_masks=loss_masks,
            stop_reasons=[step.stop_reason for step in steps],
            rollout_metrics=get_rollout_metrics(responses, rewards),
            rollout_logprobs=rollout_logprobs,
            trajectory_ids=projected_ids,
            is_last_step=is_last_step,
            actual_global_step=minimum_captured_global_step(steps),
        )
