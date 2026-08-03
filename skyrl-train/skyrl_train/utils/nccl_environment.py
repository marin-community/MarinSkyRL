import os

from loguru import logger

from skyrl_train.utils.constants import (
    DEFAULT_NCCL_TRACE_BUFFER_SIZE,
    get_nccl_monitor_heartbeat_timeout,
    get_worker_nccl_timeout_s,
)


def nccl_diagnostics_environment(*, heartbeat_timeout_seconds: int) -> dict[str, str]:
    """Return PyTorch NCCL watchdog and flight-recorder variables for workers."""

    return {
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        "TORCH_NCCL_DUMP_ON_TIMEOUT": "1",
        "TORCH_NCCL_ENABLE_MONITORING": "1",
        "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC": f"{heartbeat_timeout_seconds:d}",
        # Torch 2.9 renamed this setting while retaining the old name as an alias.
        # Keep both until every supported policy image uses the new spelling.
        "TORCH_NCCL_TRACE_BUFFER_SIZE": str(DEFAULT_NCCL_TRACE_BUFFER_SIZE),
        "TORCH_FR_BUFFER_SIZE": str(DEFAULT_NCCL_TRACE_BUFFER_SIZE),
    }


def worker_nccl_environment() -> dict[str, str]:
    """Resolve worker collective deadlines, diagnostics, and dump destinations."""

    dump_path = (
        os.environ.get("TORCH_FR_DUMP_TEMP_FILE")
        or os.environ.get("TORCH_NCCL_DEBUG_INFO_TEMP_FILE")
        or "/tmp/nccl_fr_rank"
    )
    collective_timeout_seconds = get_worker_nccl_timeout_s()
    heartbeat_timeout_value = os.environ.get("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC")
    requested_heartbeat_timeout_seconds = get_nccl_monitor_heartbeat_timeout(
        None if heartbeat_timeout_value is None else int(heartbeat_timeout_value)
    )
    heartbeat_timeout_seconds = min(requested_heartbeat_timeout_seconds, collective_timeout_seconds)
    if heartbeat_timeout_seconds != requested_heartbeat_timeout_seconds:
        logger.warning(
            "Capping TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC from {} to the {}-second collective timeout",
            requested_heartbeat_timeout_seconds,
            collective_timeout_seconds,
        )

    environment = {
        "TORCH_FR_DUMP_TEMP_FILE": dump_path,
        "TORCH_NCCL_DEBUG_INFO_TEMP_FILE": dump_path,
        "SKYRL_WORKER_NCCL_TIMEOUT_IN_S": str(collective_timeout_seconds),
    }
    environment.update(nccl_diagnostics_environment(heartbeat_timeout_seconds=heartbeat_timeout_seconds))
    return environment
