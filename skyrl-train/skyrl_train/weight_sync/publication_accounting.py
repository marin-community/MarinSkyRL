"""Request identities for auditing abort/resume across weight syncs."""

import asyncio

from dataclasses import dataclass, field


@dataclass
class PublicationRequestAccounting:
    """Drainable event ledger owned by one engine event loop.

    Logical attempts use vLLM's request ID. An attempt remains active until its
    generator exits, including attempts queued behind a paused scheduler. A pause
    records both that set and the narrower native frontend set, so late additions
    cannot silently disappear from the accounting identity.
    """

    active: set[str] = field(default_factory=set)
    started: list[str] = field(default_factory=list)
    terminal: list[dict] = field(default_factory=list)
    pauses: list[dict] = field(default_factory=list)
    pause_count: int = 0
    idle: asyncio.Event = field(default_factory=asyncio.Event, repr=False)

    def __post_init__(self) -> None:
        if not self.active:
            self.idle.set()

    async def wait_for_idle(self, timeout_seconds: float) -> None:
        if timeout_seconds <= 0:
            raise ValueError("terminal acknowledgement timeout must be positive")
        await asyncio.wait_for(self.idle.wait(), timeout=timeout_seconds)

    def start(self, request_id: str) -> None:
        if request_id in self.active:
            raise ValueError("duplicate active request identity")
        self.idle.clear()
        self.active.add(request_id)
        self.started.append(request_id)

    def finish(
        self,
        request_id: str,
        *,
        reason: str,
        tokens: int,
        first_token_time: float | None,
        policy_version_at_first_token: int | None = None,
    ) -> None:
        if request_id not in self.active:
            raise ValueError("terminal request has no active identity")
        self.active.remove(request_id)
        if not self.active:
            self.idle.set()
        self.terminal.append(
            {
                "request_id": request_id,
                "reason": reason,
                "tokens": tokens,
                "native_first_token_time": first_token_time,
                # vLLM uses zero as the unobserved first-token sentinel. Keep
                # its raw evidence while exposing canonical absence to auditors.
                "first_token_time": None if tokens == 0 and first_token_time == 0.0 else first_token_time,
                "policy_version_at_first_token": policy_version_at_first_token,
            }
        )

    def begin_pause(self, *, frontend_ids: list[str], monotonic_time: float) -> None:
        if not set(frontend_ids) <= self.active:
            raise ValueError("native frontend includes an untracked request")
        self.pause_count += 1
        self.pauses.append(
            {
                "pause_index": self.pause_count,
                "monotonic_time": monotonic_time,
                "active_before": sorted(self.active),
                "frontend_before": sorted(frontend_ids),
            }
        )

    def drain(self) -> dict:
        result = {
            "active_ids": sorted(self.active),
            "started_ids": self.started,
            "terminal": self.terminal,
            "pauses": self.pauses,
            "pause_count": self.pause_count,
        }
        self.started, self.terminal, self.pauses = [], [], []
        return result
