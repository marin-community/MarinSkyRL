DEFAULT_RAY_PLACEMENT_GROUP_TIMEOUT_SECONDS = 180
"""
Timeout for allocating the placement group for different actors in SkyRL
"""

# ``trainer.distributed.worker_collective_timeout_seconds`` is the runtime
# authority; this constant keeps lower-level API defaults aligned with it.
DEFAULT_WORKER_NCCL_TIMEOUT_IN_S = 300
DEFAULT_NCCL_MONITOR_HEARTBEAT_TIMEOUT = 300


def validate_worker_collective_timeout_seconds(value: int | None = None) -> int:
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
