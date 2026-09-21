"""Compact serving-policy provenance across asynchronous weight syncs."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import TypedDict


BEHAVIOR_POLICY_VERSION_SEGMENTS_KEY = "behavior_policy_version_segments"
# The per-choice key an OpenAI chat completion carries its spans under.
POLICY_VERSION_SEGMENTS_KEY = "policy_version_segments"


class PolicyVersionSegment(TypedDict):
    """One contiguous response span sampled from one installed policy."""

    start: int
    token_count: int
    policy_version: int | None


def append_policy_version_segment_at(
    segments: list[PolicyVersionSegment], *, start: int, token_count: int, policy_version: int | None
) -> None:
    """Append one sampled span that begins at ``start``, coalescing an adjacent equal version."""

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


def append_contiguous_policy_version_segment(
    segments: list[PolicyVersionSegment], *, token_count: int, policy_version: int | None
) -> None:
    """Thin wrapper on :func:`append_policy_version_segment_at` that starts the span where the prefix ends."""

    start = segments[-1]["start"] + segments[-1]["token_count"] if segments else 0
    append_policy_version_segment_at(
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


def policy_version_bounds(rows: list[list[PolicyVersionSegment]]) -> tuple[int, int] | None:
    """Return oldest/newest known versions, or None when no sampled token exists."""

    versions = [
        segment["policy_version"]
        for segments in rows
        for segment in segments
        if segment["token_count"] > 0 and segment["policy_version"] is not None
    ]
    return (min(versions), max(versions)) if versions else None


def consumed_token_age_metrics(
    response_ids: list[list[int]],
    loss_masks: list[list[int]],
    version_rows: list[list[PolicyVersionSegment]],
    consuming_step: int,
) -> dict[str, float]:
    """Measure optimizer-consumption age per selected token, including mixed-version responses."""
    if not (len(response_ids) == len(loss_masks) == len(version_rows)) or consuming_step < 1:
        raise ValueError("consumed token ages require aligned rows and a positive optimizer step")
    age_counts: Counter[int] = Counter()
    for response, mask, segments in zip(response_ids, loss_masks, version_rows, strict=True):
        validate_policy_version_segments(
            segments,
            response_length=len(response),
            require_known=True,
            required_mask=[bool(value) for value in mask],
        )
        for segment in segments:
            version = segment["policy_version"]
            if version is None:
                continue
            age = consuming_step - version - 1
            if age < 0:
                raise ValueError("sampled policy version is newer than the optimizer-consumption step")
            start = segment["start"]
            selected_tokens = sum(bool(value) for value in mask[start : start + segment["token_count"]])
            age_counts[age] += selected_tokens
    total = sum(age_counts.values())
    if total == 0:
        raise ValueError("consumed token ages require at least one selected token")
    p90_threshold = 0.9 * total
    cumulative = 0
    p90_age = 0
    for age, count in sorted(age_counts.items()):
        cumulative += count
        if cumulative >= p90_threshold:
            p90_age = age
            break
    return {
        "async/consumed_token_age_mean": sum(age * count for age, count in age_counts.items()) / total,
        "async/consumed_token_age_p90": float(p90_age),
        "async/consumed_token_age_max": float(max(age_counts)),
        "async/consumed_token_age_stale_fraction": sum(count for age, count in age_counts.items() if age > 0) / total,
        "async/consumed_token_age_at_least_four_fraction": sum(count for age, count in age_counts.items() if age >= 4)
        / total,
        "async/consumed_token_age_tokens": float(total),
    }


def truncate_policy_version_segments(
    segments: list[PolicyVersionSegment], response_length: int
) -> list[PolicyVersionSegment]:
    """Clip compact provenance to a truncated response without changing versions."""

    result: list[PolicyVersionSegment] = []
    for segment in segments:
        if segment["start"] >= response_length:
            break
        count = min(segment["token_count"], response_length - segment["start"])
        append_policy_version_segment_at(
            result,
            start=segment["start"],
            token_count=count,
            policy_version=segment["policy_version"],
        )
    return result


@dataclass
class PolicyVersionHistory:
    """Policy versions installed on one engine, each with the ``time.monotonic()`` reading at which it became live."""

    boundaries: list[tuple[float, int]] = field(default_factory=list)

    def record_resume(self, boundary: float, version: int) -> None:
        """Record that ``version`` serves every token sampled at or after ``boundary``."""
        if not math.isfinite(boundary) or boundary <= 0 or type(version) is not int or version < 0:
            raise ValueError("policy version boundary and version must be finite and nonnegative")
        if self.boundaries and (boundary <= self.boundaries[-1][0] or version < self.boundaries[-1][1]):
            raise ValueError("policy version boundaries and versions must not move backwards")
        self.boundaries.append((boundary, version))

    def at_first_token(self, timestamp: float | None) -> int | None:
        """Version installed when a token was sampled at ``timestamp``; None before the first boundary."""
        if timestamp is None or not math.isfinite(timestamp) or timestamp <= 0:
            return None
        for boundary, version in reversed(self.boundaries):
            if timestamp >= boundary:
                return version
        return None
