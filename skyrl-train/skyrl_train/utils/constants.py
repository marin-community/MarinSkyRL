DEFAULT_RAY_PLACEMENT_GROUP_TIMEOUT_SECONDS = 180
"""
Timeout for allocating the placement group for different actors in SkyRL
"""

# Canonical worker-NCCL-collective timeout. ONE source of truth for the default
# and the accessor (was previously divergent: constants default 600 vs utils.py
# `max(1200, env-or-1200)`). Raised to 1800s (30 min) — the long MoE weight-sync
# gather + first-step forward on 80B routinely exceeds the old 600s watchdog and
# SIGABRTs the gang. ``trainer.distributed.worker_collective_timeout_seconds``
# is the runtime authority; this constant supplies lower-level API defaults.
DEFAULT_WORKER_NCCL_TIMEOUT_IN_S = 1800
DEFAULT_NCCL_MONITOR_HEARTBEAT_TIMEOUT = 300


def get_worker_nccl_timeout_s(value: int | None = None) -> int:
    """Validate a configured worker collective timeout."""
    timeout = DEFAULT_WORKER_NCCL_TIMEOUT_IN_S if value is None else int(value)
    if timeout <= 0:
        raise ValueError("trainer.distributed.worker_collective_timeout_seconds must be positive")
    return timeout


def get_nccl_monitor_heartbeat_timeout(value: int | None = None) -> int:
    """Validate the requested NCCL monitor heartbeat deadline."""
    timeout = DEFAULT_NCCL_MONITOR_HEARTBEAT_TIMEOUT if value is None else value
    if timeout <= 0:
        raise ValueError("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC must be positive")
    return timeout
