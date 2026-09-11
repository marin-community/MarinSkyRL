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
    error_treatment: str | None


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
        attach_terminal_classifications(batch, outputs)
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
        attach_terminal_classifications(batch, steps)
        _attach_reward_channels(batch, steps, responses)
        return batch


def attach_terminal_classifications(batch: TrajectoryBatch, outputs: Sequence[TrainableInteraction]) -> None:
    """Project aligned terminal classifications when a runner supplied them."""
    exception_types = [output.disposition.exception_type for output in outputs]
    error_treatments = [output.error_treatment for output in outputs]
    if any(exception_type is not None for exception_type in exception_types):
        batch["exception_types"] = exception_types
    if any(error_treatment is not None for error_treatment in error_treatments):
        batch["error_treatments"] = error_treatments


def _logprobs_requested(request: TrajectoryRequestBatch, runner_cfg: DictConfig) -> bool:
    sampling_params = request.get("sampling_params")
    if sampling_params is not None:
        return sampling_params.get("logprobs") is not None
    return runner_cfg.sampling_params.logprobs is not None


def _loss_masks(outputs, responses, runner_cfg: DictConfig, tokenizer):
    loss_masks = [project_loss_mask(output, response) for output, response in zip(outputs, responses)]
    if runner_cfg.apply_overlong_filtering:
        loss_masks = apply_overlong_filtering(loss_masks, responses, tokenizer.eos_token_id)
    if bool(runner_cfg.get("mask_length_stops", False)):
        loss_masks = mask_length_stops(loss_masks, outputs)
    if bool(runner_cfg.get("mask_truncated_turns", False)):
        loss_masks = mask_truncated_turns(loss_masks, outputs)
    return loss_masks


def is_length_stopped(output) -> bool:
    """True when the interaction ran into an output cap: any turn hit the per-turn
    max_generate_length (harbor runner `turn_truncated`) or the terminal stop reason is
    "length"."""
    if bool(getattr(output, "turn_truncated", False)):
        return True
    evidence = getattr(output, "evidence", None)
    return getattr(evidence, "stop_reason", None) == "length"


def mask_length_stops(loss_masks: Sequence[Sequence[int]], outputs) -> list[list[int]]:
    """DAPO overlong filtering keyed on the runner's recorded stop signals.

    `apply_overlong_filtering` tests for the tokenizer eos id as the last response token,
    which multi-turn chat templates never place there (Llama-3 turns end in <|eot_id|>,
    the tokenizer eos is <|end_of_text|>), so on those models it silences every sample.
    This variant zeroes the loss mask of samples the runner marked as length-stopped and
    leaves everything else untouched. Rewards are unchanged, so the samples still enter
    the group baseline.
    """
    return [[0] * len(mask) if is_length_stopped(output) else list(mask) for mask, output in zip(loss_masks, outputs)]


def mask_truncated_turns(loss_masks: Sequence[Sequence[int]], outputs) -> list[list[int]]:
    """Zero the loss mask over the sampled tokens of turns that hit the per-turn output cap.

    Narrower than ``mask_length_stops``: the rest of the trajectory keeps its mask and its
    reward, only the capped turn (typically a repetition loop cut at max_generate_length)
    stops receiving gradient.  The token chain is untouched, so the capped turn still
    conditions later turns exactly as it did at sampling time.  Spans come from the
    runner (``truncated_turn_spans``); outputs without spans are returned unchanged.
    """
    masked: list[list[int]] = []
    for mask, output in zip(loss_masks, outputs):
        spans = getattr(output, "truncated_turn_spans", None)
        if not spans:
            masked.append(list(mask))
            continue
        new_mask = list(mask)
        for start, end in spans:
            for i in range(max(0, start), min(end, len(new_mask))):
                new_mask[i] = 0
        masked.append(new_mask)
    return masked


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
