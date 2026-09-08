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
