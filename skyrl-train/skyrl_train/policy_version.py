"""Compact serving-policy provenance across asynchronous weight syncs."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TypedDict

import numpy as np


BEHAVIOR_POLICY_VERSION_SEGMENTS_KEY = "behavior_policy_version_segments"


class PolicyVersionSegment(TypedDict):
    """One contiguous response span sampled from one installed policy."""

    start: int
    token_count: int
    policy_version: int | None


def append_policy_version_span(
    segments: list[PolicyVersionSegment], *, start: int, token_count: int, policy_version: int | None
) -> None:
    """Append one ordered sampled span, coalescing an adjacent equal version."""

    if type(start) is not int or start < 0:
        raise ValueError("policy-version segment start must be a nonnegative integer")
    if type(token_count) is not int or token_count < 0:
        raise ValueError("policy-version segment token_count must be a nonnegative integer")
    if policy_version is not None and (type(policy_version) is not int or policy_version < 0):
        raise ValueError("policy-version segment policy_version must be a nonnegative integer or None")
    if token_count == 0:
        return
    previous_end = segments[-1]["start"] + segments[-1]["token_count"] if segments else 0
    if start < previous_end:
        raise ValueError("policy-version segments must be ordered and non-overlapping")
    if segments and start == previous_end and segments[-1]["policy_version"] == policy_version:
        segments[-1] = {
            "start": segments[-1]["start"],
            "token_count": segments[-1]["token_count"] + token_count,
            "policy_version": policy_version,
        }
    else:
        segments.append({"start": start, "token_count": token_count, "policy_version": policy_version})


def append_policy_version_segment(
    segments: list[PolicyVersionSegment], *, token_count: int, policy_version: int | None
) -> None:
    """Append a contiguous retry segment after the existing response prefix."""

    start = segments[-1]["start"] + segments[-1]["token_count"] if segments else 0
    append_policy_version_span(
        segments,
        start=start,
        token_count=token_count,
        policy_version=policy_version,
    )


def validate_policy_version_segments(
    segments: list[PolicyVersionSegment],
    *,
    response_length: int,
    require_known: bool,
    required_mask=None,
) -> None:
    """Validate ordered, non-overlapping response spans and required coverage."""

    if not isinstance(segments, list):
        raise ValueError("policy-version segments must be a list")
    if type(response_length) is not int or response_length < 0:
        raise ValueError("policy-version response_length must be a nonnegative integer")
    if required_mask is None:
        required_mask = [True] * response_length
    if len(required_mask) != response_length:
        raise ValueError("required policy-version coverage mask must align with the response")
    covered = [False] * response_length
    known = [False] * response_length
    previous_end = 0
    previous_version: int | None | object = object()
    for segment in segments:
        if not isinstance(segment, dict) or set(segment) != {"start", "token_count", "policy_version"}:
            raise ValueError("each policy-version segment must contain start, token_count and policy_version")
        start = segment["start"]
        token_count = segment["token_count"]
        policy_version = segment["policy_version"]
        if type(start) is not int or start < previous_end:
            raise ValueError("policy-version segments must be ordered and non-overlapping")
        if type(token_count) is not int or token_count <= 0:
            raise ValueError("policy-version segment token_count must be a positive integer")
        if policy_version is not None and (type(policy_version) is not int or policy_version < 0):
            raise ValueError("policy-version segment policy_version must be a nonnegative integer or None")
        if previous_version == policy_version:
            if start == previous_end:
                raise ValueError("adjacent policy-version segments must be coalesced")
        previous_version = policy_version
        end = start + token_count
        if end > response_length:
            raise ValueError("policy-version segment extends beyond the response")
        covered[start:end] = [True] * token_count
        known[start:end] = [policy_version is not None] * token_count
        previous_end = end
    if any(required and not present for required, present in zip(required_mask, covered, strict=True)):
        raise ValueError("policy-version segments do not cover every required response token")
    if require_known and any(required and not present for required, present in zip(required_mask, known, strict=True)):
        raise ValueError("learner rollouts require a known installed policy version for every selected token")


def expand_policy_version_segments(
    rows: list[list[PolicyVersionSegment]],
    response_mask,
    *,
    required_mask=None,
):
    """Expand compact row segments at the dense learner boundary only."""

    if len(rows) != response_mask.shape[0]:
        raise ValueError("policy-version segment rows must align with the learner batch")
    dense = np.full(response_mask.shape, -1, dtype=np.int64)
    if required_mask is not None and required_mask.shape != response_mask.shape:
        raise ValueError("required policy-version mask must match response_mask")
    for row_index, (segments, mask) in enumerate(zip(rows, response_mask, strict=True)):
        response_length = int(np.sum(mask > 0))
        row_required = (
            [bool(value) for value in required_mask[row_index, :response_length]] if required_mask is not None else None
        )
        validate_policy_version_segments(
            segments,
            response_length=response_length,
            require_known=True,
            required_mask=row_required,
        )
        for segment in segments:
            count = segment["token_count"]
            start = segment["start"]
            if segment["policy_version"] is not None:
                dense[row_index, start : start + count] = int(segment["policy_version"])
    return dense


def policy_version_bounds(rows: list[list[PolicyVersionSegment]]) -> tuple[int, int] | None:
    """Return oldest/newest known versions, or None when no sampled token exists."""

    versions = [
        segment["policy_version"]
        for segments in rows
        for segment in segments
        if segment["token_count"] > 0 and segment["policy_version"] is not None
    ]
    return (min(versions), max(versions)) if versions else None


def truncate_policy_version_segments(
    segments: list[PolicyVersionSegment], response_length: int
) -> list[PolicyVersionSegment]:
    """Clip compact provenance to a truncated response without changing versions."""

    result: list[PolicyVersionSegment] = []
    for segment in segments:
        if segment["start"] >= response_length:
            break
        count = min(segment["token_count"], response_length - segment["start"])
        append_policy_version_span(
            result,
            start=segment["start"],
            token_count=count,
            policy_version=segment["policy_version"],
        )
    return result


@dataclass
class PolicyVersionHistory:
    """Map the engine's first-token clock to the policy version installed at that instant."""

    boundaries: list[tuple[float, int]] = field(default_factory=list)

    def record_resume(self, boundary: float, version: int) -> None:
        if not math.isfinite(boundary) or boundary <= 0 or type(version) is not int or version < 0:
            raise ValueError("policy version boundary and version must be finite and nonnegative")
        if self.boundaries and (boundary <= self.boundaries[-1][0] or version < self.boundaries[-1][1]):
            raise ValueError("policy version boundaries and versions must not move backwards")
        self.boundaries.append((boundary, version))

    def at_first_token(self, timestamp: float | None) -> int | None:
        if timestamp is None or not math.isfinite(timestamp) or timestamp <= 0:
            return None
        for boundary, version in reversed(self.boundaries):
            if timestamp >= boundary:
                return version
        return None
