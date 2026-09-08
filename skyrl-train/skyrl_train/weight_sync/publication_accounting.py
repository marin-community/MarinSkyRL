"""Request identities for auditing abort/resume across weight syncs."""

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

    def start(self, request_id: str) -> None:
        if request_id in self.active:
            raise ValueError("duplicate active request identity")
        self.active.add(request_id)
        self.started.append(request_id)

    def finish(self, request_id: str, *, reason: str, tokens: int, first_token_time: float | None) -> None:
        if request_id not in self.active:
            raise ValueError("terminal request has no active identity")
        self.active.remove(request_id)
        self.terminal.append(
            {"request_id": request_id, "reason": reason, "tokens": tokens, "first_token_time": first_token_time}
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
