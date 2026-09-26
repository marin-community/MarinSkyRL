"""Record which policy version sampled each span of a response across weight syncs."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TypedDict


# Each response's spans on engine outputs, model client outputs and chat choices; vLLM never emits it.
RESPONSE_POLICY_VERSION_SEGMENTS_KEY = "response_policy_version_segments"
# The trajectory batch field holding each response's spans aligned with its training tokens.
BEHAVIOR_POLICY_VERSION_SEGMENTS_KEY = "behavior_policy_version_segments"


class PolicyVersionSegment(TypedDict):
    """One contiguous response span sampled from one installed policy."""

    start: int
    token_count: int
    policy_version: int | None


def append_span(spans: list[PolicyVersionSegment], start: int, count: int, version: int | None) -> None:
    """Append ``count`` tokens from ``start``, merging them into an adjacent span of the same version."""
    if count == 0:
        return
    last = spans[-1] if spans else None
    if last is not None and last["policy_version"] == version and last["start"] + last["token_count"] == start:
        spans[-1] = {**last, "token_count": last["token_count"] + count}
    else:
        spans.append({"start": start, "token_count": count, "policy_version": version})


def truncate_spans(spans: Iterable[PolicyVersionSegment], response_length: int) -> list[PolicyVersionSegment]:
    """Clip spans to a response cut to ``response_length`` tokens."""
    return [
        {**span, "token_count": min(span["token_count"], response_length - span["start"])}
        for span in spans
        if span["start"] < response_length
    ]


def trained_tokens_versioned(spans: Iterable[PolicyVersionSegment], loss_mask: Sequence[int]) -> bool:
    """Whether a span with a known version covers every token the loss trains on."""
    versioned = set()
    for span in spans:
        if span["policy_version"] is not None:
            versioned.update(range(span["start"], span["start"] + span["token_count"]))
    return all(index in versioned for index, trained in enumerate(loss_mask) if trained)


def oldest_policy_version(rows: Iterable[Iterable[PolicyVersionSegment]]) -> int | None:
    """The oldest known version across rows of spans; None when no span carries one."""
    versions = (span["policy_version"] for spans in rows for span in spans)
    return min((version for version in versions if version is not None), default=None)


@dataclass
class PolicyVersionHistory:
    """Policy versions installed on one engine, each with the ``time.monotonic()`` reading at which it went live."""

    boundaries: list[tuple[float, int]] = field(default_factory=list)

    def record_resume(self, boundary: float, version: int) -> None:
        if self.boundaries and (boundary <= self.boundaries[-1][0] or version < self.boundaries[-1][1]):
            raise ValueError("policy version boundaries and versions must not move backwards")
        self.boundaries.append((boundary, version))

    def version_at(self, first_token_ts: float | None, *, submitted_at: float, returned_at: float) -> int | None:
        """The version installed at a request's first token, which vLLM stamps on the EngineCore's monotonic clock."""
        if not first_token_ts:
            return None
        if not submitted_at <= first_token_ts <= returned_at:
            raise RuntimeError(
                "vLLM first_token_ts is outside the request's lifetime; "
                "first_token_admission needs each EngineCore on its actor's host"
            )
        return next((version for boundary, version in reversed(self.boundaries) if first_token_ts >= boundary), None)
