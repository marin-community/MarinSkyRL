import bisect
import time
from collections.abc import Callable, Collection, Iterable

import numpy as np

UNSAMPLED_POLICY_STEP = -1


class PolicyStepStamps:
    """Per-request runs of (token count, policy step), timed on the EngineCore's monotonic clock."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._installed_at: list[float] = []
        self._steps: list[int] = []
        self._runs: dict[str, list[list[int]]] = {}
        self._unclaimed: set[str] = set()

    def install(self, policy_step: int) -> None:
        self._installed_at.append(self._clock())
        self._steps.append(policy_step)

    def observe(self, request_id: str, token_count: int, sampled_at: float) -> None:
        """Attribute ``token_count`` new tokens to the step installed before the engine sampled them."""
        assert sampled_at <= self._clock(), "an engine step cannot be sampled in the future"
        index = bisect.bisect_right(self._installed_at, sampled_at)
        step = self._steps[index - 1] if index else UNSAMPLED_POLICY_STEP
        runs = self._runs.setdefault(request_id, [])
        if runs and runs[-1][1] == step:
            runs[-1][0] += token_count
        else:
            runs.append([token_count, step])

    def finish(self, request_id: str) -> np.ndarray | None:
        """Pop the request's stamps, or None when none were observed or they were forgotten."""
        runs = self._runs.pop(request_id, None)
        if runs is None:
            return None
        steps = np.fromiter((step for _, step in runs), dtype=np.int32, count=len(runs))
        return np.repeat(steps, [count for count, _ in runs])

    def discard(self, request_ids: Iterable[str]) -> None:
        for request_id in request_ids:
            self._runs.pop(request_id, None)

    def forget_unclaimed(self, live_request_ids: Collection[str]) -> None:
        """Forget requests that had already left the engine at the previous call and were not claimed since."""
        for request_id in self._unclaimed:
            self._runs.pop(request_id, None)
        self._unclaimed = self._runs.keys() - live_request_ids
