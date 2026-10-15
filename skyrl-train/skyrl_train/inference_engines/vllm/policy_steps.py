import bisect
import time
from collections.abc import Collection

import numpy as np

from skyrl_train.inference_engines.base import UNSTAMPED_POLICY_STEP


class PolicyStepStamps:
    """Records, for each request, the policy step installed when the engine sampled each of its tokens."""

    def __init__(self) -> None:
        self._installed_at: list[float] = []
        self._steps: list[int] = []
        self._runs: dict[str, list[list[int]]] = {}
        self._unclaimed: set[str] = set()

    def install(self, policy_step: int, installed_at: float) -> None:
        """Stamp tokens sampled after the monotonic time ``installed_at`` with ``policy_step``."""
        self._installed_at.append(installed_at)
        self._steps.append(policy_step)

    def observe(self, request_id: str, token_count: int, sampled_at: float) -> None:
        """Attribute ``token_count`` new tokens to the step installed before the engine sampled them."""
        assert sampled_at <= time.monotonic(), "an engine step cannot be sampled in the future"
        index = bisect.bisect_right(self._installed_at, sampled_at)
        step = self._steps[index - 1] if index else UNSTAMPED_POLICY_STEP
        runs = self._runs.setdefault(request_id, [])
        if runs and runs[-1][1] == step:
            runs[-1][0] += token_count
        else:
            runs.append([token_count, step])

    def finish(self, request_id: str, response_length: int) -> list[int]:
        """Pop the stamps of the request's first ``response_length`` returned tokens."""
        runs = self._runs.pop(request_id, [])
        steps = np.fromiter((step for _, step in runs), dtype=np.int32, count=len(runs))
        stamps = np.repeat(steps, [count for count, _ in runs])
        assert len(stamps) >= response_length, (
            f"request {request_id} returned more tokens than its engine steps sampled"
        )
        return stamps[:response_length].tolist()

    def forget_unclaimed(self, live_request_ids: Collection[str]) -> None:
        """Forget requests that had already left the engine at the previous call and were not claimed since."""
        for request_id in self._unclaimed:
            self._runs.pop(request_id, None)
        self._unclaimed = self._runs.keys() - live_request_ids
