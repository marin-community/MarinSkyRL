"""One-turn structure completion from exact model-token prefixes."""

from dataclasses import replace
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, model_validator
from rolloutengine.closure_scoring import ClosureFormat, closure_state, score_closure
from rolloutengine.contracts import ModelTurn, SessionStart, Transition
from rolloutengine.spec import LoweredTaskSpec
from rolloutengine.structure_closure import PrefixStatus
from rolloutengine.task_session import session_start
from shellbox.machine import Machine
from taskcompendium.grading_result import GradeResult, Outcome

from skyrl_gym.source_task import session_parameters

STRUCTURE_CLOSURE_SESSION = "structure_closure"


class ClosurePrompt(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    prompt_token_ids: tuple[Annotated[int, Field(strict=True, ge=0)], ...] = Field(min_length=1)
    format: ClosureFormat
    family: str = Field(min_length=1)
    control: str = Field(min_length=1)
    prefix: str
    expected: str

    @model_validator(mode="after")
    def valid_reference(self) -> "ClosurePrompt":
        if not self.expected.startswith(self.prefix):
            raise ValueError("Closure reference must extend the supplied structure prefix")
        if closure_state(self.prefix, self.format).status != PrefixStatus.EXTENDABLE:
            raise ValueError("Closure task requires an unfinished, extendable prefix")
        if not score_closure(self.expected, self.expected, self.format).success:
            raise ValueError("Closure reference must be complete and valid")
        return self


class ClosureReplaySession:
    """Grade syntax and content separately, without executing the described tools."""

    def __init__(self, lowered: LoweredTaskSpec, _machine: Machine | None):
        self.task = lowered.task
        self.prompt = ClosurePrompt.model_validate(session_parameters(self.task).extras)
        self.result = GradeResult(Outcome.UNAVAILABLE, None, "No model continuation was completed")

    async def prepare(self) -> SessionStart:
        return replace(
            session_start(self.task, shell_tool=None),
            prompt_token_ids=self.prompt.prompt_token_ids,
            options={"skip_special_tokens": False},
        )

    async def advance(self, turn: ModelTurn) -> Transition:
        score = score_closure(self.prompt.prefix + turn.text, self.prompt.expected, self.prompt.format)
        self.result = GradeResult(
            Outcome.GRADED,
            float(score.success),
            passed=score.success,
            diagnostics={
                "closure": self.prompt.model_dump(mode="json", exclude={"prompt_token_ids"}),
                "syntax_complete": score.syntax_complete,
                "content_preserved": score.content_preserved,
                "stop_reason": turn.stop_reason,
            },
        )
        metric = f"closure/{self.prompt.family}/{self.prompt.control}"
        return Transition(
            done=True,
            reward=float(score.success),
            grade=self.result,
            metrics={
                f"{metric}/success": float(score.success),
                f"{metric}/syntax_complete": float(score.syntax_complete),
                f"{metric}/content_preserved": float(score.content_preserved),
            },
        )

    async def grade(self, messages: tuple[dict[str, Any], ...], *, stop_reason: str) -> GradeResult:
        return self.result

    async def close(self) -> None:
        pass
