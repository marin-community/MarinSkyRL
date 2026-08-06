"""Trainer-facing import surface for the canonical pre-launch environment manager."""

from cloud.iris.env_vars import (
    DEBUG_ARTIFACT_DIR_ENV as DEBUG_ARTIFACT_DIR_ENV,
    DEBUG_MODE_ENV as DEBUG_MODE_ENV,
    DEFAULT_NCCL_TRACE_BUFFER_SIZE as DEFAULT_NCCL_TRACE_BUFFER_SIZE,
    ENV_VAR_SPECS as ENV_VAR_SPECS,
    FR_DUMP_TEMP_FILE_ENV as FR_DUMP_TEMP_FILE_ENV,
    NCCL_DEBUG_INFO_TEMP_FILE_ENV as NCCL_DEBUG_INFO_TEMP_FILE_ENV,
    DistributedDebugMode as DistributedDebugMode,
    EnvVarManager as EnvVarManager,
    EnvVarScope as EnvVarScope,
    EnvVarSource as EnvVarSource,
    ensure_debug_artifact_directories as ensure_debug_artifact_directories,
    managed_environment_names as managed_environment_names,
    nccl_diagnostics_environment as nccl_diagnostics_environment,
    write_process_manifest as write_process_manifest,
)
