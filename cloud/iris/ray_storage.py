"""Ray object-store paths shared by the Iris launcher and node controller."""

import os

DEFAULT_RAY_SPILL_DIR = "/tmp/skyrl-ray-spill"


def resolve_ray_spill_dir(path: str) -> str:
    """Resolve and validate a node-local Ray spill directory."""
    resolved = os.path.expandvars(os.path.expanduser(path))
    if not os.path.isabs(resolved) or "://" in resolved:
        raise ValueError(f"Ray spill directory must be an absolute local path, got {resolved!r}")
    return resolved
