import bisect
import time
from collections.abc import Callable, Iterable

import numpy as np

UNSAMPLED_POLICY_STEP = -1


class PolicyStepStamps:
    """Per-request runs of (token count, policy step), timed on the EngineCore's monotonic clock."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._installed_at: list[float] = []
        self._steps: list[int] = []
        self._runs: dict[str, list[tuple[int, int]]] = {}

    def install(self, policy_step: int) -> None:
        self._installed_at.append(self._clock())
        self._steps.append(policy_step)

    def observe(self, request_id: str, token_count: int, sampled_at: float) -> None:
        """Attribute ``token_count`` new tokens to the step installed before the engine sampled them."""
        assert sampled_at <= self._clock(), "an engine step cannot be sampled in the future"
        index = bisect.bisect_right(self._installed_at, sampled_at)
        step = self._steps[index - 1] if index else UNSAMPLED_POLICY_STEP
        self._runs.setdefault(request_id, []).append((token_count, step))

    def finish(self, request_id: str) -> np.ndarray:
        runs = self._runs.pop(request_id, [])
        steps = np.fromiter((step for _, step in runs), dtype=np.int32, count=len(runs))
        return np.repeat(steps, [count for count, _ in runs])

    def discard(self, request_ids: Iterable[str]) -> None:
        for request_id in request_ids:
            self._runs.pop(request_id, None)
