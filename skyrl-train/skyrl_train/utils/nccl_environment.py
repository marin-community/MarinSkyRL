import os

from loguru import logger

from skyrl_train.nccl_diagnostics import nccl_diagnostics_environment
from skyrl_train.env_vars import FR_DUMP_TEMP_FILE_ENV, NCCL_DEBUG_INFO_TEMP_FILE_ENV
from skyrl_train.utils.constants import (
    get_nccl_monitor_heartbeat_timeout,
    get_worker_nccl_timeout_s,
)


def worker_nccl_environment() -> dict[str, str]:
    """Resolve worker collective deadlines, diagnostics, and dump destinations."""

    dump_path = (
        os.environ.get(FR_DUMP_TEMP_FILE_ENV) or os.environ.get(NCCL_DEBUG_INFO_TEMP_FILE_ENV) or "/tmp/nccl_fr_rank"
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
        FR_DUMP_TEMP_FILE_ENV: dump_path,
        NCCL_DEBUG_INFO_TEMP_FILE_ENV: dump_path,
        "SKYRL_WORKER_NCCL_TIMEOUT_IN_S": str(collective_timeout_seconds),
    }
    environment.update(nccl_diagnostics_environment(heartbeat_timeout_seconds=heartbeat_timeout_seconds))
    return environment
