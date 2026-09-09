"""Map engine-native first-token clocks to acknowledged publication boundaries."""

import math
from dataclasses import dataclass, field


@dataclass
class PublicationVersionHistory:
    # Both boundaries and first-token timestamps must use the same engine clock.
    boundaries: list[tuple[float, int]] = field(default_factory=list)

    def record_resume(self, boundary: float, version: int) -> None:
        if not math.isfinite(boundary) or boundary <= 0 or type(version) is not int or version < 0:
            raise ValueError("publication boundary and version must be finite and nonnegative")
        if self.boundaries and (boundary <= self.boundaries[-1][0] or version < self.boundaries[-1][1]):
            raise ValueError("publication boundaries and versions must not move backwards")
        self.boundaries.append((boundary, version))

    def at_first_token(self, timestamp: float | None) -> int | None:
        if timestamp is None or not math.isfinite(timestamp) or timestamp <= 0:
            return None
        for boundary, version in reversed(self.boundaries):
            if timestamp >= boundary:
                return version
        # An output sampled before the first measured boundary keeps its submission stamp.
        return None


def earliest_sampled_policy_version(response_ids: list[list[int]], versions: list[int | None]) -> int | None:
    """Return the earliest emitted-token version only when every sampled row is known."""
    if len(response_ids) != len(versions):
        raise ValueError("sampled-token versions must align with response rows")
    sampled = [version for ids, version in zip(response_ids, versions) if ids]
    if any(version is not None and (type(version) is not int or version < 0) for version in sampled):
        raise ValueError("sampled-token versions must be nonnegative integers")
    if not sampled or any(version is None for version in sampled):
        return None
    return min(sampled)
