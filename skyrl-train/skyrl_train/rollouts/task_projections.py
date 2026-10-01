"""Convert common rollout records into whole-trajectory or per-step training data."""

from collections.abc import Sequence
from dataclasses import dataclass, replace
import json

import numpy as np
from skyrl_gym.verification import RewardResult, RolloutEvidence, TrainingDisposition, VerificationResult
from taskcompendium.grading import GradeResult, Outcome
from taskcompendium.rollout import RolloutData

from skyrl_train.error_treatment import ErrorTreatment
from skyrl_train.trajectory_runners.projections import (
    StepWiseTrajectoryProjection,
    WholeTrajectoryProjection,
    logprobs_requested,
)
from skyrl_train.trajectory_runners.selected_topk import align_student_topk
from skyrl_train.trajectory_runners.types import AgentLoopOutput, TrajectoryBatch, TrajectoryRequestBatch
from skyrl_train.utils.harbor_errors import (
    DEFAULT_ERROR_HANDLING_CONFIG,
    ErrorHandlingConfig,
    classify_exception_type,
    passthrough_logprob_error_type,
    treatment_excludes_from_baseline,
)


def verification_result(grade: GradeResult) -> VerificationResult:
    """Preserve the task verdict and its availability in the training contract."""
    if grade.status == Outcome.GRADED:
        assert grade.reward is not None
        return VerificationResult.verified(
            grade.reward,
            passed=grade.passed,
            diagnostics=grade.diagnostics,
            score_min=grade.score_min,
            score_max=grade.score_max,
        )
    if grade.status == Outcome.SKIPPED:
        return VerificationResult.skipped(grade.error or "Verification skipped", diagnostics=grade.diagnostics)
    if grade.status == Outcome.UNAVAILABLE:
        return VerificationResult.unavailable(grade.error or "Verification unavailable", diagnostics=grade.diagnostics)
    return VerificationResult.error(grade.error or grade.status.value, diagnostics=grade.diagnostics)


def training_output(
    rollout: RolloutData,
    error_handling: ErrorHandlingConfig = DEFAULT_ERROR_HANDLING_CONFIG,
    *,
    logprobs_required: bool = False,
) -> AgentLoopOutput:
    """Exclude missing grades from loss and baseline calculations."""
    graded = rollout.grade.status == Outcome.GRADED
    skipped = rollout.grade.status == Outcome.SKIPPED
    verification = verification_result(rollout.grade)
    if graded:
        disposition = TrainingDisposition.train()
    elif skipped:
        disposition = TrainingDisposition.train(reason="verification skipped")
    else:
        disposition = TrainingDisposition(
            loss_eligible=False,
            baseline_eligible=False,
            reason=rollout.grade.error or rollout.grade.status.value,
        )
    token_rewards = None
    token_credit = None
    components = {}
    optimization_reward = rollout.grade.reward if graded else 0.0
    if any(step.transition.reward is not None for step in rollout.steps):
        rewards = [0.0] * len(rollout.response_token_ids)
        optimization_reward = 0.0
        for step in rollout.steps:
            if (graded or skipped) and step.transition.reward is not None:
                optimization_reward += step.transition.reward
                if step.transition.token_rewards is None:
                    rewards[step.response_end] += step.transition.reward
                else:
                    start = step.response_end + 1 - len(step.turn.response_token_ids)
                    rewards[start : step.response_end + 1] = step.transition.token_rewards
                for name, value in step.transition.reward_components.items():
                    components[name] = components.get(name, 0.0) + value
        token_rewards = tuple(rewards)
    if any(step.transition.token_credit is not None for step in rollout.steps):
        credit = [0.0] * len(rollout.response_token_ids)
        for step in rollout.steps:
            if (graded or skipped) and step.transition.token_credit is not None:
                start = step.response_end + 1 - len(step.turn.response_token_ids)
                credit[start : step.response_end + 1] = step.transition.token_credit
        token_credit = tuple(credit)
    error_treatment = None
    if rollout.failure is not None:
        failure = rollout.failure
        treatment = (
            classify_exception_type(failure.exception_type, error_handling)
            if error_handling.enable_error_classification
            else ErrorTreatment.MASK
        )
        missing_logprobs = passthrough_logprob_error_type(
            treatment,
            has_rollout_logprobs=rollout.logprobs is not None,
            rollout_logprobs_required=logprobs_required,
        )
        disposition = TrainingDisposition(
            loss_eligible=bool(any(rollout.loss_mask))
            and treatment is not ErrorTreatment.MASK
            and missing_logprobs is None,
            baseline_eligible=not treatment_excludes_from_baseline(treatment, verifier_available=graded)
            and missing_logprobs is None,
            reason="Rollout execution failed",
            exception_type=missing_logprobs or failure.exception_type,
        )
        error_treatment = treatment.value
        verification = replace(
            verification,
            diagnostics={**verification.diagnostics, "exception_type": failure.exception_type, **failure.diagnostics},
        )
        if treatment is not ErrorTreatment.PASSTHROUGH or not graded:
            optimization_reward = 0.0
            token_rewards = None if token_rewards is None else (0.0,) * len(token_rewards)
            token_credit = None if token_credit is None else (0.0,) * len(token_credit)
            components = {}
    candidates = [step.turn.metadata.get("student_topk_indices") for step in rollout.steps]
    scores = [step.turn.metadata.get("behavior_topk_logprobs") for step in rollout.steps]
    selected = None
    if candidates and all(value is not None for value in candidates) and all(value is not None for value in scores):
        generated = []
        admitted_candidates = []
        admitted_scores = []
        for step, step_candidates, step_scores in zip(rollout.steps, candidates, scores, strict=True):
            start = step.response_end + 1 - len(step.turn.response_token_ids)
            for offset, (token, ids, values) in enumerate(
                zip(step.turn.response_token_ids, step_candidates, step_scores, strict=True)
            ):
                if rollout.loss_mask[start + offset]:
                    generated.append(token)
                    admitted_candidates.append(ids)
                    admitted_scores.append(values)
        selected = align_student_topk(
            rollout.response_token_ids,
            rollout.loss_mask,
            generated,
            admitted_candidates,
            admitted_scores,
        )
    routes = [step.turn.metadata.get("routed_experts") for step in rollout.steps]
    tagged_steps = [step.turn.metadata.get("response_span_tags") for step in rollout.steps]
    response_span_tags = None
    if any(tags is not None for tags in tagged_steps):
        response_span_tags = [0] * len(rollout.response_token_ids)
        for step, tags in zip(rollout.steps, tagged_steps, strict=True):
            if tags is not None:
                if len(tags) != len(step.turn.response_token_ids):
                    raise ValueError("Span tags must align with generated tokens")
                response_span_tags[step.response_end + 1 - len(tags) : step.response_end + 1] = tags
    template = next((value for value in routes if value is not None), None)
    routed_experts = None
    if template is not None:
        routed_experts = np.zeros((len(rollout.response_token_ids), *template.shape[1:]), dtype=template.dtype)
        for step, values in zip(rollout.steps, routes, strict=True):
            if values is not None:
                if len(values) != len(step.turn.response_token_ids):
                    raise ValueError("Expert routes must align with generated tokens")
                routed_experts[step.response_end + 1 - len(values) : step.response_end + 1] = values
    return AgentLoopOutput(
        evidence=RolloutEvidence(
            messages=rollout.messages,
            response=rollout.steps[-1].turn.text if rollout.steps else None,
            stop_reason=rollout.stop_reason,
            generated_token_count=sum(rollout.loss_mask),
            prompt_token_ids=rollout.prompt_token_ids,
            response_token_ids=rollout.response_token_ids,
            behavior_logprobs=None if rollout.logprobs is None else np.asarray(rollout.logprobs, dtype=np.float32),
            student_topk_indices=None if selected is None else selected.indices,
            behavior_topk_logprobs=None if selected is None else selected.topk_logprobs,
            routed_experts=routed_experts,
        ),
        verification=verification,
        reward=RewardResult(
            unshaped_reward=rollout.grade.reward,
            optimization_reward=optimization_reward,
            token_rewards=token_rewards,
            token_credit=token_credit,
            components=components,
        ),
        disposition=disposition,
        loss_mask=list(rollout.loss_mask),
        env_metrics=dict(rollout.metrics),
        error_treatment=error_treatment,
        response_span_tags=response_span_tags,
    )


@dataclass(frozen=True)
class WholeTaskProjection:
    projection: WholeTrajectoryProjection
    error_handling: ErrorHandlingConfig = DEFAULT_ERROR_HANDLING_CONFIG
    harbor_error_handling: ErrorHandlingConfig | None = None

    def project(self, rollouts: Sequence[RolloutData], request: TrajectoryRequestBatch) -> TrajectoryBatch:
        required = logprobs_requested(request, self.projection.runner_config)
        policies = task_error_policies(request, self.error_handling, self.harbor_error_handling)
        outputs = [
            training_output(rollout, policy, logprobs_required=required)
            for rollout, policy in zip(rollouts, policies, strict=True)
        ]
        for index, environment in enumerate(request.get("env_classes") or []):
            if environment == "taskcompendium":
                outputs[index] = replace(outputs[index], env_metrics={})
        batch = self.projection.project(outputs, request)
        merge_task_metrics(batch, rollouts, request)
        return batch


@dataclass(frozen=True)
class StepTaskProjection:
    projection: StepWiseTrajectoryProjection
    error_handling: ErrorHandlingConfig = DEFAULT_ERROR_HANDLING_CONFIG
    harbor_error_handling: ErrorHandlingConfig | None = None

    def project(self, rollouts: Sequence[RolloutData], request: TrajectoryRequestBatch) -> TrajectoryBatch:
        required = logprobs_requested(request, self.projection.runner_config)
        policies = task_error_policies(request, self.error_handling, self.harbor_error_handling)
        batch = self.projection.project(
            [
                step_training_outputs(rollout, policy, logprobs_required=required)
                for rollout, policy in zip(rollouts, policies, strict=True)
            ],
            request,
        )
        merge_task_metrics(batch, rollouts, request)
        return batch


def task_error_policies(
    request: TrajectoryRequestBatch, default: ErrorHandlingConfig, harbor: ErrorHandlingConfig | None
) -> list[ErrorHandlingConfig]:
    if harbor is None:
        return [default] * len(request["prompts"])
    return [
        harbor if "harbor" in json.loads(extras["task_spec"])["metadata"] else default
        for extras in request["env_extras"]
    ]


def merge_task_metrics(
    batch: TrajectoryBatch, rollouts: Sequence[RolloutData], request: TrajectoryRequestBatch
) -> None:
    """Sum common-task counters separately from Gym's environment-specific metrics."""
    metrics = batch.get("rollout_metrics") or {}
    environments = request.get("env_classes")
    if environments is None:
        return
    for rollout, environment in zip(rollouts, environments, strict=True):
        if environment != "taskcompendium":
            continue
        for key, value in rollout.metrics.items():
            metrics[key] = metrics.get(key, 0.0) + value
    batch["rollout_metrics"] = metrics


def step_training_outputs(
    rollout: RolloutData,
    error_handling: ErrorHandlingConfig = DEFAULT_ERROR_HANDLING_CONFIG,
    *,
    logprobs_required: bool = False,
) -> list[AgentLoopOutput]:
    """Use exact served prompts and score each transition at its final action token."""
    if not rollout.steps:
        return [training_output(rollout, error_handling, logprobs_required=logprobs_required)]
    outputs = []
    for index, step in enumerate(rollout.steps):
        last = index == len(rollout.steps) - 1
        grade = rollout.grade
        if grade.status in {Outcome.GRADED, Outcome.SKIPPED}:
            grade = step.transition.grade or (
                grade if last else GradeResult(Outcome.SKIPPED, None, "No intermediate verifier")
            )
        reward = step.transition.reward
        if not last and reward is None:
            reward = 0.0
        transition = replace(step.transition, reward=reward, grade=grade)
        turn = step.turn
        output = training_output(
            replace(
                rollout,
                messages=step.messages,
                prompt_token_ids=turn.prompt_token_ids,
                response_token_ids=turn.response_token_ids,
                loss_mask=(1,) * len(turn.response_token_ids),
                logprobs=turn.logprobs,
                grade=grade,
                stop_reason=rollout.stop_reason if last else turn.stop_reason,
                steps=(replace(step, response_end=len(turn.response_token_ids) - 1, transition=transition),),
                metrics=step.transition.metrics,
            ),
            error_handling,
            logprobs_required=logprobs_required,
        )
        if (
            not last
            and not step.transition.done
            and grade.status == Outcome.UNAVAILABLE
            and rollout.grade.status in {Outcome.GRADED, Outcome.SKIPPED}
            and rollout.failure is None
        ):
            output.disposition = TrainingDisposition.train()
        outputs.append(output)
    return outputs
