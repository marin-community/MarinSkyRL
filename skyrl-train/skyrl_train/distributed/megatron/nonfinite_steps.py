import math
from dataclasses import dataclass
from enum import StrEnum


class NonfiniteStepAction(StrEnum):
    APPLY = "apply"
    SKIP = "skip"
    FAIL = "fail"


@dataclass(frozen=True)
class OptimizerStepResult:
    grad_norm: float | None
    applied: bool


def nonfinite_step_action(
    grad_norm: float, found_inf: bool, consecutive_skipped: int, limit: int | None
) -> NonfiniteStepAction:
    """Permit at most limit consecutive nonfinite attempts before failing training."""
    if math.isfinite(grad_norm) and not found_inf:
        return NonfiniteStepAction.APPLY
    if limit is None or consecutive_skipped >= limit:
        return NonfiniteStepAction.FAIL
    return NonfiniteStepAction.SKIP
