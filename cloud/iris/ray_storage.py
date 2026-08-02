"""Ray object-store paths shared by the Iris launcher and node controller."""

import json
import os
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

DEFAULT_RAY_SPILL_DIR = "/tmp/skyrl-ray-spill"
RAY_SPILL_BUFFER_SIZE = 100 * 1024 * 1024
_RAY_LOCAL_SPILL_FLAG = "--object-spilling-directory"


class RaySpillBackend(StrEnum):
    LOCAL = "local"
    R2 = "r2"


class RaySpillTarget(Protocol):
    location: str

    def head_flags(self) -> list[str]: ...

    def worker_flags(self) -> list[str]: ...

    def description(self) -> str: ...


@dataclass(frozen=True)
class LocalRaySpillTarget:
    location: str

    def __post_init__(self) -> None:
        if resolve_ray_spill_dir(self.location) != self.location:
            raise ValueError(f"Local Ray spill target must be resolved, got {self.location!r}")

    def head_flags(self) -> list[str]:
        return [f"{_RAY_LOCAL_SPILL_FLAG}={self.location}"]

    def worker_flags(self) -> list[str]:
        return [f"{_RAY_LOCAL_SPILL_FLAG}={self.location}"]

    def description(self) -> str:
        return f"Ray object spilling -> launcher-owned local scratch {self.location}"


@dataclass(frozen=True)
class R2RaySpillTarget:
    location: str

    def __post_init__(self) -> None:
        if not self.location.startswith("s3://"):
            raise ValueError(f"R2 Ray spill target must use s3://, got {self.location!r}")

    def head_flags(self) -> list[str]:
        spilling_config = json.dumps(
            {
                "type": "smart_open",
                "params": {"uri": self.location, "buffer_size": RAY_SPILL_BUFFER_SIZE},
            }
        )
        system_config = json.dumps({"object_spilling_config": spilling_config, "min_spilling_size": 0})
        return [f"--system-config={system_config}"]

    def worker_flags(self) -> list[str]:
        return []

    def description(self) -> str:
        return f"Ray object spilling -> R2 prefix {self.location} (remote backend)"


def resolve_ray_spill_dir(path: str) -> str:
    """Resolve and validate a node-local Ray spill directory."""
    resolved = os.path.expandvars(os.path.expanduser(path))
    if not os.path.isabs(resolved) or "://" in resolved:
        raise ValueError(f"Ray spill directory must be an absolute local path, got {resolved!r}")
    return resolved
