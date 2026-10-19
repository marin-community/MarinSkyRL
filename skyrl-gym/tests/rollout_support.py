"""Read task episode signals from published Marin rollout records in tests."""

from types import SimpleNamespace

from rolloutengine.models import GradedRollout, TaskRolloutSpec
from skyrl_gym.task_records import StepReward, fold_grades
from taskcompendium.models import NoGrader


def episode_result(result: GradedRollout, spec: TaskRolloutSpec):
    conversation = result.rollout.conversations[-1] if result.rollout.conversations else None
    steps = []
    if conversation is not None:
        for step in conversation.steps:
            data = step.transition.data if isinstance(step.transition.data, StepReward) else StepReward()
            transition = SimpleNamespace(
                **vars(data),
                done=step.transition.done,
                observations=step.transition.observations,
                metrics=step.transition.metrics,
            )
            steps.append(
                SimpleNamespace(turn=step.turn, transition=transition, response_end=step.response_end, messages=step.messages)
            )
    grade = result.grade
    if isinstance(spec.task.grader, NoGrader):
        grades = [step.transition.grade for step in steps if step.transition.grade is not None]
        if steps and steps[-1].transition.task_grade is not None:
            grade = steps[-1].transition.task_grade
        elif grades:
            grade = fold_grades(grades) if spec.harness in {"nemotron_ultra", "openenv"} else grades[-1]
    return SimpleNamespace(
        task_id=spec.task.id,
        grade=grade,
        steps=tuple(steps),
        prompt_token_ids=() if conversation is None else conversation.prompt_token_ids,
        response_token_ids=() if conversation is None else conversation.response_token_ids,
        loss_mask=() if conversation is None else conversation.loss_mask,
        logprobs=None if conversation is None else conversation.logprobs,
        messages=() if conversation is None else conversation.messages,
        stop_reason=result.rollout.stop_reason,
        metrics=result.rollout.metrics,
        failure=result.rollout.failure,
    )
