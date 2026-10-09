"""A multiplication task with correction prompts."""

import re
from typing import Any

from rolloutengine.contracts import ModelTurn, SessionStart, Transition
from shellbox.machine import Machine
from skyrl_gym.source_task import session_parameters
from taskcompendium.grading_result import GradeResult, Outcome
from rolloutengine.spec import LoweredTaskSpec
from rolloutengine.task_session import session_start

from skyrl_gym.answer_tasks import ground_truth
from skyrl_gym.task_records import terminal_grade


class MultiplyTaskSession:
    """Keep the multiplication reference and return feedback until completion."""

    def __init__(self, lowered: LoweredTaskSpec, machine: Machine | None):
        task = lowered.task
        parameters = session_parameters(task)
        extras = parameters.extras
        self.task = task
        self.expected = str(ground_truth(extras)).strip()
        self.max_turns = lowered.session.max_turns
        self.grades: list[GradeResult] = []

    async def prepare(self) -> SessionStart:
        return session_start(self.task)

    async def advance(self, turn: ModelTurn) -> Transition:
        match = re.search(r"\\boxed\{([^}]+)\}", turn.text)
        answer = match.group(1) if match else None
        correct = answer is not None and answer.strip() == self.expected
        done = len(self.grades) + 1 >= self.max_turns or correct
        reward = (1.0 if correct else 0.5 if answer is not None else 0.0) if done else 0.0
        grade = GradeResult(Outcome.GRADED, float(correct), passed=correct)
        self.grades.append(grade)
        feedback = (
            f"Your answer '{answer}' is incorrect. Please try again."
            if answer is not None
            else r"Please provide your answer in the format \boxed{your_answer}."
        )
        return Transition(
            done=done,
            observations=() if done else ({"role": "user", "content": feedback},),
            reward=reward,
            grade=grade,
            metrics={"parsed_answer": answer},
        )

    async def grade(self, messages: tuple[dict[str, Any], ...]) -> GradeResult:
        return terminal_grade(self.grades)

    async def close(self) -> None:
        pass
