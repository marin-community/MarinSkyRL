import os

#
SKYRL_RAY_PG_TIMEOUT_IN_S = int(os.environ.get("SKYRL_RAY_PG_TIMEOUT_IN_S", 180))
"""
Timeout for allocating the placement group for different actors in SkyRL
"""

# Canonical worker-NCCL-collective timeout. ONE source of truth for the default
# and the accessor (was previously divergent: constants default 600 vs utils.py
# `max(1200, env-or-1200)`). Raised to 1800s (30 min) — the long MoE weight-sync
# gather + first-step forward on 80B routinely exceeds the old 600s watchdog and
# SIGABRTs the gang; every live iris config already sets >=1800 so this is a
# no-op there and only protects a config that forgot the line. Env var is the
# override (env > default); set it lower for a quick test.
DEFAULT_WORKER_NCCL_TIMEOUT_IN_S = 1800
DEFAULT_NCCL_MONITOR_HEARTBEAT_TIMEOUT = 300
DEFAULT_NCCL_TRACE_BUFFER_SIZE = 20000


def get_worker_nccl_timeout_s() -> int:
    """Resolve the worker NCCL-collective timeout (seconds): env override, else default."""
    return int(os.environ.get("SKYRL_WORKER_NCCL_TIMEOUT_IN_S", DEFAULT_WORKER_NCCL_TIMEOUT_IN_S))


def get_nccl_monitor_heartbeat_timeout(value: str | int | None = None) -> int:
    """Validate the requested NCCL monitor heartbeat deadline."""
    timeout = int(DEFAULT_NCCL_MONITOR_HEARTBEAT_TIMEOUT if value is None else value)
    if timeout <= 0:
        raise ValueError("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC must be positive")
    return timeout


def nccl_communicator_timeout_environment(timeout_seconds: int) -> dict[str, str]:
    """Return nonblocking NCCL settings; reject non-positive timeouts."""
    if timeout_seconds <= 0:
        raise ValueError("NCCL communicator timeout must be positive")
    return {
        "TORCH_NCCL_USE_COMM_NONBLOCKING": "1",
        "TORCH_NCCL_NONBLOCKING_TIMEOUT": str(timeout_seconds),
    }


SKYRL_WORKER_NCCL_TIMEOUT_IN_S = get_worker_nccl_timeout_s()
"""
Timeout for initializing the NCCL process group for the worker, defaults to 30 minutes.
"""

# For some reason the `LD_LIBRARY_PATH` is not exported to the worker with .env file.
SKYRL_LD_LIBRARY_PATH_EXPORT = str(os.environ.get("SKYRL_LD_LIBRARY_PATH_EXPORT", "False")).lower() in (
    "true",
    "1",
    "yes",
)
"""
Whether to export ``LD_LIBRARY_PATH`` environment variable from the driver to the workers with Ray's runtime env.

For example, if you are using RDMA, you may need to customize the ``LD_LIBRARY_PATH`` to include the RDMA libraries (Ex: EFA on AWS).
"""

SKYRL_PYTHONPATH_EXPORT = str(os.environ.get("SKYRL_PYTHONPATH_EXPORT", "False")).lower() in (
    "true",
    "1",
    "yes",
)
"""
Whether to export ``PYTHONPATH`` environment variable from the driver to the workers with Ray's runtime env.

See https://github.com/ray-project/ray/issues/56697 for details on why this is needed.
"""
