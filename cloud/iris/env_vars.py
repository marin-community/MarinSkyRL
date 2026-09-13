"""Compatibility imports for the shared MarinSkyRL runtime environment contract."""

import sys

from marinskyrl.runtime_environment import (
    DEBUG_ARTIFACT_DIR_ENV as DEBUG_ARTIFACT_DIR_ENV,
    DEBUG_MODE_ENV as DEBUG_MODE_ENV,
    DEFAULT_NCCL_TRACE_BUFFER_SIZE as DEFAULT_NCCL_TRACE_BUFFER_SIZE,
    ENV_VAR_SPECS as ENV_VAR_SPECS,
    EXECUTION_UID_ENV as EXECUTION_UID_ENV,
    FR_DUMP_TEMP_FILE_ENV as FR_DUMP_TEMP_FILE_ENV,
    HF_HUB_OFFLINE_ENV as HF_HUB_OFFLINE_ENV,
    LIVE_STACK_INTERVAL_ENV as LIVE_STACK_INTERVAL_ENV,
    NCCL_DEBUG_INFO_TEMP_FILE_ENV as NCCL_DEBUG_INFO_TEMP_FILE_ENV,
    NUMA_AFFINITY_ENV as NUMA_AFFINITY_ENV,
    PYTHONFAULTHANDLER_ENV as PYTHONFAULTHANDLER_ENV,
    RAY_USE_UVLOOP_ENV as RAY_USE_UVLOOP_ENV,
    RUN_ID_ENV as RUN_ID_ENV,
    TELEMETRY_ENDPOINT_ENV as TELEMETRY_ENDPOINT_ENV,
    UV_USE_IO_URING_ENV as UV_USE_IO_URING_ENV,
    VLLM_BATCH_INVARIANT_ENV as VLLM_BATCH_INVARIANT_ENV,
    DebugMode as DebugMode,
    DistributedDebugMode as DistributedDebugMode,
    EnvVarManager as EnvVarManager,
    EnvVarScope as EnvVarScope,
    EnvVarSource as EnvVarSource,
    EnvVarSpec as EnvVarSpec,
    EnvVarWriter as EnvVarWriter,
    ensure_debug_artifact_directories as ensure_debug_artifact_directories,
    grug_gpu_gate_environment as grug_gpu_gate_environment,
    managed_environment_names as managed_environment_names,
    nccl_diagnostics_environment as nccl_diagnostics_environment,
    ray_cluster_owner_environment as ray_cluster_owner_environment,
    temporarily_unset_managed_environment as temporarily_unset_managed_environment,
    wandb_launch_environment as wandb_launch_environment,
    write_process_manifest as write_process_manifest,
)
from marinskyrl.runtime_environment import _main


if __name__ == "__main__":
    _main(sys.argv)
