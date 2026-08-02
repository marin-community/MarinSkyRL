"""Ray object-store paths shared by the Iris launcher and node controller."""

import os
from dataclasses import dataclass
from enum import StrEnum

DEFAULT_RAY_SPILL_DIR = "/tmp/skyrl-ray-spill"


class RaySpillBackend(StrEnum):
    LOCAL = "local"
    R2 = "r2"


@dataclass(frozen=True)
class RaySpillTarget:
    """One validated local directory or remote R2 prefix."""

    backend: RaySpillBackend
    location: str

    def __post_init__(self) -> None:
        if self.backend is RaySpillBackend.LOCAL:
            if resolve_ray_spill_dir(self.location) != self.location:
                raise ValueError(f"Local Ray spill target must be resolved, got {self.location!r}")
        elif self.backend is RaySpillBackend.R2:
            if not self.location.startswith("s3://"):
                raise ValueError(f"R2 Ray spill target must use s3://, got {self.location!r}")
        else:
            raise ValueError(f"Unsupported Ray spill backend: {self.backend}")


def resolve_ray_spill_dir(path: str) -> str:
    """Resolve and validate a node-local Ray spill directory."""
    resolved = os.path.expandvars(os.path.expanduser(path))
    if not os.path.isabs(resolved) or "://" in resolved:
        raise ValueError(f"Ray spill directory must be an absolute local path, got {resolved!r}")
    return resolved
