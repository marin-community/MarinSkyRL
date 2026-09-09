"""Mutual exclusion between PPO updates and a frozen weight-sync diagnostic."""

from contextlib import contextmanager
from threading import Lock


class PolicyWeightAccess:
    def __init__(self):
        self._lock = Lock()
        self.owner = None

    @contextmanager
    def hold(self, owner: str):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError(f"Policy weights are already owned by {self.owner}")
        self.owner = owner
        try:
            yield
        finally:
            self.owner = None
            self._lock.release()
