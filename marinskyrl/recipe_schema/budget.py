"""The authored context declaration and its launch-side token limits."""

from typing import Annotated, Self

from pydantic import Field, PositiveInt, model_validator

from .model import Section


class ContextBudget(Section):
    """Reserve a complete response within one rollout request window."""

    request_window_tokens: PositiveInt
    max_new_tokens_per_turn: PositiveInt
    max_turns: PositiveInt
    generated_budget_fraction: Annotated[int | float, Field(gt=0, le=1)] = 0.5
    overlong_cache_fraction: Annotated[int | float, Field(ge=0, le=1)] = 0.25

    @model_validator(mode="after")
    def _response_fits(self) -> Self:
        if self.request_window_tokens <= self.max_new_tokens_per_turn:
            raise ValueError("request_window_tokens must exceed max_new_tokens_per_turn")
        return self

    @property
    def max_input_tokens(self) -> int:
        return self.request_window_tokens - self.max_new_tokens_per_turn

    @property
    def opencode_limit_output(self) -> int:
        return min(self.max_new_tokens_per_turn, max(1, self.max_input_tokens - 1))

    @property
    def opencode_limit_context(self) -> int:
        output = self.opencode_limit_output
        margin = min(1024, max(0, self.max_input_tokens - output - 1))
        return max(1, self.max_input_tokens - output - margin)

    @property
    def generated_tokens_per_trajectory(self) -> int:
        if self.max_turns == 1:
            return self.max_new_tokens_per_turn
        return max(1, int(self.request_window_tokens * self.generated_budget_fraction))

    @property
    def overlong_cache_tokens(self) -> int:
        return int(self.generated_tokens_per_trajectory * self.overlong_cache_fraction)

    def as_dict(self) -> dict[str, int | float]:
        return {
            **self.model_dump(),
            "max_input_tokens": self.max_input_tokens,
            "generated_tokens_per_trajectory": self.generated_tokens_per_trajectory,
            "overlong_cache_tokens": self.overlong_cache_tokens,
            "opencode_limit_context": self.opencode_limit_context,
            "opencode_limit_output": self.opencode_limit_output,
        }
