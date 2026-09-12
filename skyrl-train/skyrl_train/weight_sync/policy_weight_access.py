"""Mutual exclusion between PPO updates and frozen weight-sync sessions."""

from contextlib import contextmanager
from threading import Lock
from uuid import uuid4


class PolicyWeightAccess:
    def __init__(self):
        self._lock = Lock()
        self.owner = None
        self._token = None

    def acquire(self, owner: str) -> str:
        """Create a session that may be released by a later RPC on another thread."""
        if not isinstance(owner, str) or not owner:
            raise ValueError("Policy-weight ownership needs a nonempty name")
        with self._lock:
            if self._token is not None:
                raise RuntimeError(f"Policy weights are already owned by {self.owner}")
            self._token = uuid4().hex
            self.owner = owner
            return self._token

    def transfer(self, token: str, owner: str):
        """Move an existing lease between phases without admitting an update."""
        if not isinstance(owner, str) or not owner:
            raise ValueError("Policy-weight ownership needs a nonempty name")
        with self._lock:
            if self._token is None or token != self._token:
                raise RuntimeError("Policy-weight session token is absent or does not match")
            self.owner = owner

    def release(self, token: str):
        """Only the matching session may end ownership; stale calls fail closed."""
        with self._lock:
            if self._token is None or token != self._token:
                raise RuntimeError("Policy-weight session token is absent or does not match")
            self.owner = None
            self._token = None

    @contextmanager
    def hold(self, owner: str):
        token = self.acquire(owner)
        try:
            yield
        finally:
            self.release(token)
