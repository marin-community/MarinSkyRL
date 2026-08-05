"""Dependency-light construction of the ProcessGroupNCCL diagnostic environment."""

DEFAULT_NCCL_TRACE_BUFFER_SIZE = 20_000


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
