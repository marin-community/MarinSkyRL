"""Compare native rollout records with the Nemotron GenRM judge."""

from collections.abc import Sequence
from dataclasses import replace

import requests
from rolloutengine.contracts import RolloutData
from skyrl_gym.envs.nemotron_ultra.genrm import grade_genrm_group, response_object
from skyrl_gym.envs.nemotron_ultra.judge import OpenAIJudge
from taskcompendium.grading import GradeResult, Outcome
from taskcompendium.models import TaskSpec
from taskcompendium.submission import conversation_messages


def _with_grade(rollout: RolloutData, grade: GradeResult, metrics: dict[str, float]) -> RolloutData:
    # Group grades replace provisional per-turn rewards and credit.
    has_token_rewards = grade.status == Outcome.GRADED and any(
        step.transition.reward is not None for step in rollout.steps
    )
    last_credited = next((index for index in reversed(range(len(rollout.loss_mask))) if rollout.loss_mask[index]), None)
    steps = tuple(
        replace(
            step,
            transition=replace(
                step.transition,
                reward=(grade.reward if step.response_end == last_credited else 0.0) if has_token_rewards else None,
                token_rewards=None,
                token_credit=None,
                reward_components={},
                grade=grade,
            ),
        )
        for step in rollout.steps
    )
    return replace(rollout, grade=grade, steps=steps, metrics={**rollout.metrics, **metrics})


def grade_genrm_rollouts(
    task: TaskSpec,
    rollouts: Sequence[RolloutData],
    eligible: Sequence[bool],
    phase: str,
) -> list[RolloutData]:
    """Replace provisional grades with comparisons among valid peers."""
    assert task.group_verifier is not None
    parameters = task.group_verifier.parameters
    config = parameters["config"]
    assert isinstance(config, dict)
    result = list(rollouts)
    if phase == "eval":
        for index, rollout in enumerate(rollouts):
            metrics = {"genrm/cohort_skipped_eval": 1.0}
            result[index] = (
                _with_grade(
                    rollout,
                    GradeResult(Outcome.UNAVAILABLE, None, "GenRM evaluation needs a comparison cohort"),
                    metrics,
                )
                if rollout.grade.status == Outcome.GRADED
                else replace(rollout, metrics={**rollout.metrics, **metrics})
            )
        return result
    judge_config = config.get("judge")
    if judge_config is None:
        raise RuntimeError("GenRM tasks require a judge in group_verifier.parameters.config")
    judge = OpenAIJudge(**judge_config)
    expected_size = int(config.get("num_rollouts_per_prompt", 16))
    if len(rollouts) != expected_size:
        raise ValueError(f"GenRM cohort requires {expected_size} rollouts for a prompt, received {len(rollouts)}")
    indices = [
        index
        for index, (rollout, valid) in enumerate(zip(rollouts, eligible, strict=True))
        if valid and rollout.grade.status == Outcome.GRADED
    ]
    if len(indices) < 2:
        for index in indices:
            result[index] = _with_grade(
                rollouts[index], GradeResult(Outcome.UNAVAILABLE, None, "Insufficient valid GenRM peers"), {}
            )
        return result
    responses = []
    for index in indices:
        rollout = rollouts[index]
        message = next(
            (dict(message) for message in reversed(rollout.messages) if message.get("role") == "assistant"), {}
        )
        message["content"] = rollout.steps[-1].turn.text if rollout.steps else ""
        responses.append(response_object(message))
    try:
        rewards, metrics = grade_genrm_group(
            conversation_history=conversation_messages(task.context),
            response_objects=responses,
            principle=parameters["principle"],
            judge=judge,
            config=config,
        )
    except (RuntimeError, ValueError, requests.RequestException) as error:
        grade = GradeResult(
            Outcome.INFRA_ERROR,
            None,
            "GenRM comparisons failed",
            diagnostics={"error_type": type(error).__name__, "error_message": str(error)},
        )
        for index in indices:
            result[index] = _with_grade(rollouts[index], grade, {"genrm/comparison_failure": 1.0})
        return result
    for index, reward in zip(indices, rewards, strict=True):
        result[index] = _with_grade(
            rollouts[index],
            GradeResult(
                Outcome.GRADED,
                reward,
                diagnostics={"agent": parameters["agent"], "genrm_metrics": metrics},
                score_min=1.0,
                score_max=5.0,
            ),
            {f"genrm/{name}": value for name, value in metrics.items()},
        )
    return result
