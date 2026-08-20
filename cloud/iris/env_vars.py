"""Canonical ownership and projection of MarinSkyRL environment variables."""

from __future__ import annotations

import json
import os
import re
import shlex
import site
import socket
import sys
from contextlib import contextmanager
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any


class EnvVarScope(StrEnum):
    DRIVER = "driver"
    RAY_WORKER = "ray_worker"
    INFERENCE_WORKER = "inference_worker"
    TASK_RUNTIME = "task_runtime"


class DistributedDebugMode(StrEnum):
    OFF = "off"
    DISTRIBUTED = "distributed"


class EnvVarSource(StrEnum):
    CONFIG = "config"
    DERIVED = "derived"
    EXTERNAL = "external"
    SECRET = "secret"


class EnvVarWriter(StrEnum):
    MANAGER = "manager"
    PYTHON_ASSIGNMENT = "python-assignment"
    PYTHON_PUTENV = "python-putenv"
    PYTHON_SETDEFAULT = "python-setdefault"
    PYTHON_ENV_MAPPING = "python-env-mapping"
    PYTHON_ENV_UPDATE = "python-env-update"
    PYTHON_ENV_KEYWORD = "python-env-keyword"
    PYTHON_ENV_RETURN = "python-env-return"
    YAML_EXTRA_ENV = "yaml-extra-env"
    SHELL_EXPORT = "shell-export"
    DOCKER_ENV = "docker-env"


@dataclass(frozen=True)
class EnvVarSpec:
    name: str
    owner: str
    source: EnvVarSource
    scopes: frozenset[EnvVarScope]
    writers: frozenset[EnvVarWriter] = frozenset({EnvVarWriter.MANAGER})


ALL_RUNTIME_SCOPES = frozenset(EnvVarScope)

DEBUG_MODE_ENV = "SKYRL_DEBUG_MODE"
DEBUG_ARTIFACT_DIR_ENV = "SKYRL_DEBUG_ARTIFACT_DIR"
COLLECTIVE_PHASE_DIAGNOSTICS_ENV = "SKYRL_COLLECTIVE_PHASE_DIAGNOSTICS"
FR_DUMP_TEMP_FILE_ENV = "TORCH_FR_DUMP_TEMP_FILE"
NCCL_DEBUG_INFO_TEMP_FILE_ENV = "TORCH_NCCL_DEBUG_INFO_TEMP_FILE"
PYTHONPATH_ENV = "PYTHONPATH"
VLLM_USE_V1_ENV = "VLLM_USE_V1"
VLLM_USE_DEEP_GEMM_ENV = "VLLM_USE_DEEP_GEMM"
VLLM_BATCH_INVARIANT_ENV = "VLLM_BATCH_INVARIANT"
VLLM_ALLOW_INSECURE_SERIALIZATION_ENV = "VLLM_ALLOW_INSECURE_SERIALIZATION"
WANDB_ENTITY_ENV = "WANDB_ENTITY"
HF_HUB_OFFLINE_ENV = "HF_HUB_OFFLINE"
LD_LIBRARY_PATH_ENV = "LD_LIBRARY_PATH"
NVRTC_HOME_ENV = "NVRTC_HOME"
RAY_CLUSTER_OWNER_ENV = "SKYRL_RAY_CLUSTER_OWNER"
NUMA_AFFINITY_ENV = "SKYRL_ENABLE_NUMA_AFFINITY"
TELEMETRY_ENDPOINT_ENV = "SKYRL_TELEMETRY_ENDPOINT"
RUN_ID_ENV = "SKYRL_RUN_ID"
EXECUTION_UID_ENV = "SKYRL_EXECUTION_UID"
DEFAULT_NCCL_TRACE_BUFFER_SIZE = 20_000


ENV_VAR_SPECS = (
    EnvVarSpec(DEBUG_MODE_ENV, "trainer.debug_mode", EnvVarSource.CONFIG, ALL_RUNTIME_SCOPES),
    EnvVarSpec(DEBUG_ARTIFACT_DIR_ENV, "trainer.debug_mode", EnvVarSource.DERIVED, ALL_RUNTIME_SCOPES),
    EnvVarSpec("NCCL_DEBUG", "trainer.debug_mode", EnvVarSource.DERIVED, ALL_RUNTIME_SCOPES),
    EnvVarSpec("NCCL_DEBUG_SUBSYS", "trainer.debug_mode", EnvVarSource.DERIVED, ALL_RUNTIME_SCOPES),
    EnvVarSpec("NCCL_DEBUG_FILE", "trainer.debug_mode", EnvVarSource.DERIVED, ALL_RUNTIME_SCOPES),
    EnvVarSpec("PYTHONFAULTHANDLER", "trainer.debug_mode", EnvVarSource.DERIVED, ALL_RUNTIME_SCOPES),
    EnvVarSpec(
        COLLECTIVE_PHASE_DIAGNOSTICS_ENV,
        "trainer.collective_phase_diagnostics",
        EnvVarSource.DERIVED,
        ALL_RUNTIME_SCOPES,
    ),
    EnvVarSpec("TORCH_FR_BUFFER_SIZE", "trainer.debug_mode", EnvVarSource.DERIVED, ALL_RUNTIME_SCOPES),
    EnvVarSpec("TORCH_CPP_LOG_LEVEL", "trainer.debug_mode", EnvVarSource.DERIVED, ALL_RUNTIME_SCOPES),
    EnvVarSpec(FR_DUMP_TEMP_FILE_ENV, "trainer.debug_mode", EnvVarSource.DERIVED, ALL_RUNTIME_SCOPES),
    EnvVarSpec("TORCH_NCCL_ASYNC_ERROR_HANDLING", "trainer.debug_mode", EnvVarSource.DERIVED, ALL_RUNTIME_SCOPES),
    EnvVarSpec(NCCL_DEBUG_INFO_TEMP_FILE_ENV, "trainer.debug_mode", EnvVarSource.DERIVED, ALL_RUNTIME_SCOPES),
    EnvVarSpec("TORCH_NCCL_DESYNC_DEBUG", "trainer.debug_mode", EnvVarSource.DERIVED, ALL_RUNTIME_SCOPES),
    EnvVarSpec("TORCH_NCCL_DUMP_ON_TIMEOUT", "trainer.debug_mode", EnvVarSource.DERIVED, ALL_RUNTIME_SCOPES),
    EnvVarSpec("TORCH_NCCL_ENABLE_MONITORING", "trainer.debug_mode", EnvVarSource.DERIVED, ALL_RUNTIME_SCOPES),
    EnvVarSpec("TORCH_NCCL_ENABLE_TIMING", "trainer.debug_mode", EnvVarSource.DERIVED, ALL_RUNTIME_SCOPES),
    EnvVarSpec("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "trainer.debug_mode", EnvVarSource.DERIVED, ALL_RUNTIME_SCOPES),
    EnvVarSpec("TORCH_NCCL_TRACE_BUFFER_SIZE", "trainer.debug_mode", EnvVarSource.DERIVED, ALL_RUNTIME_SCOPES),
    EnvVarSpec("TORCH_NCCL_TRACE_CPP_STACK", "trainer.debug_mode", EnvVarSource.DERIVED, ALL_RUNTIME_SCOPES),
    EnvVarSpec("TORCH_SHOW_CPP_STACKTRACES", "trainer.debug_mode", EnvVarSource.DERIVED, ALL_RUNTIME_SCOPES),
    EnvVarSpec("TORCH_SYMBOLIZE_MODE", "trainer.debug_mode", EnvVarSource.DERIVED, ALL_RUNTIME_SCOPES),
    EnvVarSpec(PYTHONPATH_ENV, "ci.marin_nightly.grug", EnvVarSource.EXTERNAL, frozenset({EnvVarScope.DRIVER})),
    EnvVarSpec(VLLM_USE_V1_ENV, "ci.marin_nightly.grug", EnvVarSource.EXTERNAL, frozenset({EnvVarScope.DRIVER})),
    EnvVarSpec(
        VLLM_USE_DEEP_GEMM_ENV,
        "ci.marin_nightly.grug",
        EnvVarSource.EXTERNAL,
        frozenset({EnvVarScope.DRIVER}),
    ),
    EnvVarSpec(
        VLLM_BATCH_INVARIANT_ENV,
        "trainer.algorithm.batch_invariant",
        EnvVarSource.CONFIG,
        frozenset({EnvVarScope.RAY_WORKER, EnvVarScope.INFERENCE_WORKER}),
    ),
    EnvVarSpec(
        VLLM_ALLOW_INSECURE_SERIALIZATION_ENV,
        "generator.fuse_weights",
        EnvVarSource.CONFIG,
        frozenset({EnvVarScope.RAY_WORKER, EnvVarScope.INFERENCE_WORKER}),
    ),
    EnvVarSpec(WANDB_ENTITY_ENV, "launch.wandb_entity", EnvVarSource.EXTERNAL, frozenset({EnvVarScope.TASK_RUNTIME})),
    EnvVarSpec(
        HF_HUB_OFFLINE_ENV,
        "checkpoint_export.hf_hub_publish",
        EnvVarSource.EXTERNAL,
        frozenset({EnvVarScope.DRIVER}),
    ),
    EnvVarSpec(
        LD_LIBRARY_PATH_ENV,
        "runtime.bootstrap",
        EnvVarSource.EXTERNAL,
        frozenset({EnvVarScope.RAY_WORKER, EnvVarScope.TASK_RUNTIME}),
    ),
    EnvVarSpec(
        NVRTC_HOME_ENV,
        "runtime.bootstrap",
        EnvVarSource.EXTERNAL,
        frozenset({EnvVarScope.RAY_WORKER, EnvVarScope.TASK_RUNTIME}),
    ),
    EnvVarSpec(
        RAY_CLUSTER_OWNER_ENV,
        "runtime.ray_cluster",
        EnvVarSource.EXTERNAL,
        frozenset({EnvVarScope.DRIVER}),
    ),
    EnvVarSpec(
        NUMA_AFFINITY_ENV,
        "trainer.placement.enable_numa_affinity",
        EnvVarSource.CONFIG,
        frozenset({EnvVarScope.RAY_WORKER, EnvVarScope.INFERENCE_WORKER}),
    ),
    # The endpoint and the run identity are properties of the whole run, so every
    # process that emits has to agree on them. The execution uid identifies one task
    # attempt: it is resolved per task and must not be broadcast from the driver's
    # node to actors on other nodes, so it stops at the driver. A process that reaches
    # a Ray worker without it derives its own from IRIS_TASK_ID.
    EnvVarSpec(
        TELEMETRY_ENDPOINT_ENV,
        "runtime.telemetry",
        EnvVarSource.EXTERNAL,
        ALL_RUNTIME_SCOPES,
    ),
    EnvVarSpec(
        RUN_ID_ENV,
        "runtime.telemetry",
        EnvVarSource.EXTERNAL,
        ALL_RUNTIME_SCOPES,
    ),
    EnvVarSpec(
        EXECUTION_UID_ENV,
        "runtime.telemetry",
        EnvVarSource.EXTERNAL,
        frozenset({EnvVarScope.TASK_RUNTIME, EnvVarScope.DRIVER}),
    ),
)

_PYTHON_WRITERS = frozenset(
    {
        EnvVarWriter.PYTHON_ASSIGNMENT,
        EnvVarWriter.PYTHON_PUTENV,
        EnvVarWriter.PYTHON_SETDEFAULT,
        EnvVarWriter.PYTHON_ENV_MAPPING,
        EnvVarWriter.PYTHON_ENV_UPDATE,
        EnvVarWriter.PYTHON_ENV_KEYWORD,
        EnvVarWriter.PYTHON_ENV_RETURN,
    }
)
_RUNTIME_BOUNDARY_WRITERS = _PYTHON_WRITERS | {
    EnvVarWriter.YAML_EXTRA_ENV,
    EnvVarWriter.SHELL_EXPORT,
    EnvVarWriter.DOCKER_ENV,
}
_BUILD_BOUNDARY_WRITERS = frozenset({EnvVarWriter.SHELL_EXPORT, EnvVarWriter.DOCKER_ENV})

# Explicit interfaces owned outside MarinSkyRL. These names may be projected at
# process, task, container, or build boundaries, but they are not training controls.
_SECRET_BOUNDARIES = {
    "DAYTONA_API_KEY",
    "KUBECONFIG",
    "MLFLOW_TRACKING_TOKEN",
    "WANDB_API_KEY",
}
_BUILD_BOUNDARIES = {
    "CPATH",
    "CUDNN_PATH",
    "DEBIAN_FRONTEND",
    "DOCKER_CONFIG",
    "LANG",
    "LANGUAGE",
    "LC_ALL",
    "MAX_JOBS",
    "NVCC_THREADS",
    "TORCH_CUDA_ARCH_LIST",
    "UV_HTTP_TIMEOUT",
    "UV_PROJECT_ENVIRONMENT",
    "UV_RETRIES",
    "VLLM_FORK_DIR",
    "VLLM_TARGET_DEVICE",
    "WHEEL_VENV",
}
_RUNTIME_BOUNDARIES = {
    "AWS_CONFIG_FILE",
    "AWS_DEFAULT_REGION",
    "AWS_ENDPOINT_URL",
    "AWS_REGION",
    "AWS_S3_ADDRESSING_STYLE",
    "CUDA_DEVICE_MAX_CONNECTIONS",
    "CUDA_DEVICE_ORDER",
    "CUDA_MODULE_LOADING",
    "CUDA_VISIBLE_DEVICES",
    "DATA_DIR",
    "FSSPEC_S3",
    "GLOO_SOCKET_IFNAME",
    "HARBOR_DISTRIBUTED_CONTAINERS",
    "HARBOR_MODEL_ENDPOINT",
    "HF_HUB_DOWNLOAD_TIMEOUT",
    "HF_HUB_OFFLINE",
    "LD_LIBRARY_PATH",
    "LOCAL_RANK",
    "MASTER_ADDR",
    "MASTER_PORT",
    "MLFLOW_TRACKING_URI",
    "NCCL_CUMEM_ENABLE",
    "NCCL_DEBUG",
    "NCCL_DEBUG_SUBSYS",
    "NCCL_NVLS_ENABLE",
    "NCCL_P2P_DISABLE",
    "NCCL_SHM_DISABLE",
    "NCCL_SOCKET_FAMILY",
    "NCCL_SOCKET_IFNAME",
    "NUM_INFERENCE_ENGINES",
    "NVTE_FUSED_ATTN",
    "OMP_NUM_THREADS",
    "OTAGENT_LITERAL_LOG_PATH",
    "OT_AGENT_IRIS_RAY_PORT",
    "OT_AGENT_IRIS_RENDEZVOUS_DIR",
    "POLICY_NUM_NODES",
    "PYTHONPATH",
    "PYTHONUNBUFFERED",
    "PYTORCH_CUDA_ALLOC_CONF",
    "RANK",
    "RAY_ADDRESS",
    "RAY_USE_UVLOOP",
    "RL_ENV_DIR",
    "RL_SYNC",
    "SKYRL_HOME",
    "TENSOR_PARALLEL_SIZE",
    "TEST_FILE",
    "TF_CPP_MIN_LOG_LEVEL",
    "TORCH_FR_BUFFER_SIZE",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING",
    "TORCH_NCCL_AVOID_RECORD_STREAMS",
    "TORCH_NCCL_DEBUG_INFO_TEMP_FILE",
    "TORCH_NCCL_DESYNC_DEBUG",
    "TORCH_NCCL_DUMP_ON_TIMEOUT",
    "TORCH_NCCL_ENABLE_MONITORING",
    "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC",
    "TORCH_NCCL_TRACE_CPP_STACK",
    "TRAIN_FILE",
    "TRANSFORMERS_OFFLINE",
    "UV_USE_IO_URING",
    "VLLM_ALLOW_INSECURE_SERIALIZATION",
    "VLLM_ALLOW_ROUTED_EXPERTS_DCP",
    "VLLM_ALLOW_RUNTIME_LORA_UPDATING",
    "VLLM_ALLREDUCE_USE_SYMM_MEM",
    "VLLM_DISABLE_COMPILE_CACHE",
    "VLLM_ENABLE_V1_MULTIPROCESSING",
    "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS",
    "VLLM_MQ_MAX_CHUNK_BYTES_MB",
    "VLLM_RAY_BUNDLE_INDICES",
    "VLLM_RAY_PER_WORKER_GPUS",
    "VLLM_ROUTED_EXPERTS_SIDE_TIMEOUT_SECONDS",
    "VLLM_USE_DEEP_GEMM",
    "VLLM_USE_FLASHINFER_SAMPLER",
    "VLLM_USE_V1",
    "WANDB_DIR",
    "WORLD_SIZE",
}

_BOUNDARY_WRITERS = {
    **{name: (EnvVarSource.SECRET, _RUNTIME_BOUNDARY_WRITERS) for name in _SECRET_BOUNDARIES},
    **{name: (EnvVarSource.EXTERNAL, _BUILD_BOUNDARY_WRITERS) for name in _BUILD_BOUNDARIES},
    **{name: (EnvVarSource.EXTERNAL, _RUNTIME_BOUNDARY_WRITERS) for name in _RUNTIME_BOUNDARIES},
}
ENV_VAR_SPECS = tuple(
    replace(spec, writers=writers | {EnvVarWriter.MANAGER}) if spec.name in _BOUNDARY_WRITERS else spec
    for spec in ENV_VAR_SPECS
    for _, writers in [_BOUNDARY_WRITERS.get(spec.name, (spec.source, spec.writers))]
)
_DECLARED_NAMES = {spec.name for spec in ENV_VAR_SPECS}
ENV_VAR_SPECS += tuple(
    EnvVarSpec(name, "external.runtime", source, frozenset(), writers)
    for name, (source, writers) in sorted(_BOUNDARY_WRITERS.items())
    if name not in _DECLARED_NAMES
)

_SPECS_BY_NAME = {spec.name: spec for spec in ENV_VAR_SPECS}
if len(_SPECS_BY_NAME) != len(ENV_VAR_SPECS):
    raise RuntimeError("Environment variable names must have exactly one EnvVarManager owner")

_REMOTE_PATH = re.compile(r"^[a-z][a-z0-9+.-]*://", re.IGNORECASE)
_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9_.-]+")
_NCCL_SETUP_SUBSYSTEMS = "INIT,BOOTSTRAP,ENV,NET,GRAPH,TUNING"


def ray_cluster_owner_environment() -> dict[str, str]:
    """Identify the Iris task runtime as the owner of its attached Ray cluster."""
    return {RAY_CLUSTER_OWNER_ENV: "iris-task-runtime"}


def _config_value(config: Any, dotted_path: str, default: Any = None) -> Any:
    value = config
    for component in dotted_path.split("."):
        if value is None:
            return default
        if isinstance(value, Mapping):
            value = value.get(component)
        else:
            value = getattr(value, component, None)
    return default if value is None else value


def _safe_component(value: str) -> str:
    cleaned = _SAFE_COMPONENT.sub("-", value).strip("-.")
    return cleaned or "run"


class EnvVarManager:
    """Resolve managed variables once and project them into runtime scopes."""

    def __init__(self, values: Mapping[str, str]):
        unknown = set(values) - set(_SPECS_BY_NAME)
        if unknown:
            raise ValueError(f"Unregistered managed environment variables: {sorted(unknown)}")
        self._values = dict(values)

    @classmethod
    def from_config(cls, config: Any, *, environ: Mapping[str, str] | None = None) -> "EnvVarManager":
        ambient = os.environ if environ is None else environ
        values = {}
        if library_path := ambient.get(LD_LIBRARY_PATH_ENV):
            values[LD_LIBRARY_PATH_ENV] = library_path
        if nvrtc_home := ambient.get(NVRTC_HOME_ENV):
            values[NVRTC_HOME_ENV] = nvrtc_home
        # The task runtime resolved these from its own Iris context before Ray started.
        # Carry them into the scopes the trainer's Ray actors and inference engines read.
        for telemetry_name in (TELEMETRY_ENDPOINT_ENV, RUN_ID_ENV, EXECUTION_UID_ENV):
            if telemetry_value := ambient.get(telemetry_name):
                values[telemetry_name] = telemetry_value
        if _config_value(config, "trainer.algorithm.batch_invariant", False):
            values[VLLM_BATCH_INVARIANT_ENV] = "1"
        if _config_value(config, "generator.fuse_weights", False):
            values[VLLM_ALLOW_INSECURE_SERIALIZATION_ENV] = "1"
        if _config_value(config, "trainer.placement.enable_numa_affinity", False):
            values[NUMA_AFFINITY_ENV] = "1"
        if _config_value(config, "trainer.collective_phase_diagnostics", False):
            values[COLLECTIVE_PHASE_DIAGNOSTICS_ENV] = "1"
        raw_mode = _config_value(config, "trainer.debug_mode", "off")
        try:
            mode = DistributedDebugMode(str(raw_mode))
        except ValueError as error:
            choices = ", ".join(mode.value for mode in DistributedDebugMode)
            raise ValueError(f"trainer.debug_mode must be one of: {choices}; got {raw_mode!r}") from error
        if mode is DistributedDebugMode.OFF:
            return cls(values)

        artifact_root = ambient.get(DEBUG_ARTIFACT_DIR_ENV) or cls._artifact_root(config)
        values.update(cls._distributed_values(artifact_root))
        return cls(values)

    @classmethod
    def for_distributed_launch(cls, *, job_name: str, artifact_root: str | None = None) -> "EnvVarManager":
        root = artifact_root or f"/tmp/skyrl-debug/{_safe_component(job_name)}"
        return cls(cls._distributed_values(root))

    @classmethod
    def for_frozen_cuda_runtime(
        cls,
        site_packages: list[str],
    ) -> "EnvVarManager":
        """Resolve Python-wheel CUDA library paths for task and Ray worker processes."""
        nvidia_roots = [Path(root) / "nvidia" for root in site_packages if (Path(root) / "nvidia").is_dir()]
        library_paths = sorted(path for root in nvidia_roots for path in root.glob("*/lib") if path.is_dir())
        nvrtc_homes = [root / "cuda_nvrtc" for root in nvidia_roots if (root / "cuda_nvrtc" / "lib").is_dir()]
        if not library_paths:
            raise RuntimeError("The frozen GPU runtime has no Python-wheel CUDA library directories")
        if len(nvrtc_homes) != 1:
            raise RuntimeError(f"The frozen GPU runtime must have exactly one NVRTC home; found {nvrtc_homes}")

        library_path = os.pathsep.join(str(path) for path in library_paths)
        return cls({LD_LIBRARY_PATH_ENV: library_path, NVRTC_HOME_ENV: str(nvrtc_homes[0])})

    def write_shell_activation(self, path: Path, scope: EnvVarScope) -> None:
        """Write managed values as a sourceable shell activation file."""
        values = self.environment_for(scope)
        lines = []
        for name, value in sorted(values.items()):
            suffix = f"${{{name}:+:${name}}}" if name == LD_LIBRARY_PATH_ENV else ""
            lines.append(f"export {name}={shlex.quote(value)}{suffix}\n")
        path.write_text("".join(lines))

    @staticmethod
    def _artifact_root(config: Any) -> str:
        checkpoint_path = str(_config_value(config, "trainer.ckpt_path", "") or "")
        run_name = _safe_component(str(_config_value(config, "trainer.run_name", "run")))
        if checkpoint_path and not _REMOTE_PATH.match(checkpoint_path):
            checkpoint = Path(checkpoint_path).expanduser()
            return str(checkpoint.parent / "debug")
        return f"/tmp/skyrl-debug/{run_name}"

    @staticmethod
    def _distributed_values(artifact_root: str) -> dict[str, str]:
        root = str(Path(artifact_root).expanduser())
        flight_prefix = str(Path(root) / "flight_recorder" / "nccl_fr_rank_")
        return {
            **nccl_diagnostics_environment(heartbeat_timeout_seconds=300),
            DEBUG_MODE_ENV: DistributedDebugMode.DISTRIBUTED.value,
            DEBUG_ARTIFACT_DIR_ENV: root,
            "NCCL_DEBUG": "INFO",
            "NCCL_DEBUG_SUBSYS": _NCCL_SETUP_SUBSYSTEMS,
            "NCCL_DEBUG_FILE": str(Path(root) / "nccl" / "nccl.%h.%p.log"),
            "PYTHONFAULTHANDLER": "1",
            COLLECTIVE_PHASE_DIAGNOSTICS_ENV: "1",
            "TORCH_CPP_LOG_LEVEL": "INFO",
            FR_DUMP_TEMP_FILE_ENV: flight_prefix,
            NCCL_DEBUG_INFO_TEMP_FILE_ENV: flight_prefix,
            "TORCH_NCCL_DESYNC_DEBUG": "1",
            "TORCH_NCCL_ENABLE_TIMING": "1",
            "TORCH_NCCL_TRACE_CPP_STACK": "1",
            "TORCH_SHOW_CPP_STACKTRACES": "1",
            "TORCH_SYMBOLIZE_MODE": "fast",
        }

    def environment_for(self, scope: EnvVarScope) -> dict[str, str]:
        return {name: value for name, value in self._values.items() if scope in _SPECS_BY_NAME[name].scopes}

    def apply_to_process(
        self, scope: EnvVarScope, *, environ: MutableMapping[str, str] | None = None
    ) -> dict[str, str]:
        target = os.environ if environ is None else environ
        values = self.environment_for(scope)
        target.update(values)
        if artifact_root := values.get(DEBUG_ARTIFACT_DIR_ENV):
            ensure_debug_artifact_directories(artifact_root)
        return values


@contextmanager
def temporarily_unset_managed_environment(
    name: str,
    scope: EnvVarScope,
    *,
    environ: MutableMapping[str, str] | None = None,
):
    """Temporarily remove one registered variable and restore its exact prior state."""
    spec = _SPECS_BY_NAME.get(name)
    if spec is None or scope not in spec.scopes:
        raise ValueError(f"{name!r} is not registered for the {scope.value} environment scope")
    target = os.environ if environ is None else environ
    previous = target.pop(name, None)
    try:
        yield
    finally:
        if previous is not None:
            target[name] = previous


def ensure_debug_artifact_directories(artifact_root: str) -> None:
    root = Path(artifact_root)
    for child in ("collective_phases", "flight_recorder", "nccl", "processes", "runs"):
        (root / child).mkdir(parents=True, exist_ok=True)


def nccl_diagnostics_environment(*, heartbeat_timeout_seconds: int) -> dict[str, str]:
    """Return the centrally owned watchdog and bounded flight-recorder settings."""
    return {
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        "TORCH_NCCL_DUMP_ON_TIMEOUT": "1",
        "TORCH_NCCL_ENABLE_MONITORING": "1",
        "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC": f"{heartbeat_timeout_seconds:d}",
        "TORCH_NCCL_TRACE_BUFFER_SIZE": str(DEFAULT_NCCL_TRACE_BUFFER_SIZE),
        "TORCH_FR_BUFFER_SIZE": str(DEFAULT_NCCL_TRACE_BUFFER_SIZE),
    }


def write_process_manifest(
    role: str,
    *,
    environment: Mapping[str, str] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> Path:
    """Persist one collision-free process receipt when distributed debug mode is active."""
    values = os.environ if environment is None else environment
    artifact_root = values.get(DEBUG_ARTIFACT_DIR_ENV)
    if not artifact_root:
        raise RuntimeError(f"{DEBUG_ARTIFACT_DIR_ENV} is required for a debug process manifest")
    ensure_debug_artifact_directories(artifact_root)
    hostname = socket.gethostname()
    path = Path(artifact_root) / "processes" / f"{_safe_component(role)}.{hostname}.{os.getpid()}.json"
    managed = {name: values[name] for name in sorted(_SPECS_BY_NAME) if name in values}
    payload = {
        "schema_version": 1,
        "role": role,
        "hostname": hostname,
        "pid": os.getpid(),
        "debug_mode": managed.get(DEBUG_MODE_ENV),
        "environment": managed,
        "metadata": dict(metadata or {}),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n")
    temporary.replace(path)
    return path


def managed_environment_names(scope: EnvVarScope | None = None) -> frozenset[str]:
    if scope is None:
        return frozenset(_SPECS_BY_NAME)
    return frozenset(spec.name for spec in ENV_VAR_SPECS if scope in spec.scopes)


def grug_gpu_gate_environment(repository_root: str, *, environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Build the environment for the Grug rollout/train/broadcast GPU gate."""
    ambient = os.environ if environ is None else environ
    source_paths = f"{repository_root}/skyrl-gym:{repository_root}/skyrl-train"
    if pythonpath := ambient.get(PYTHONPATH_ENV):
        source_paths = f"{source_paths}:{pythonpath}"
    return {
        PYTHONPATH_ENV: source_paths,
        VLLM_USE_V1_ENV: "1",
        VLLM_USE_DEEP_GEMM_ENV: "0",
    }


def wandb_launch_environment(*, entity: str | None, environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Resolve the W&B entity with explicit launch configuration taking precedence."""
    ambient = os.environ if environ is None else environ
    return {WANDB_ENTITY_ENV: entity or ambient.get(WANDB_ENTITY_ENV, "dogml")}


def _main(argv: list[str]) -> None:
    if len(argv) == 3 and argv[1] == "write-frozen-cuda-runtime":
        manager = EnvVarManager.for_frozen_cuda_runtime(site.getsitepackages())
        manager.write_shell_activation(Path(argv[2]), EnvVarScope.TASK_RUNTIME)
        return
    if len(argv) >= 5 and argv[1] == "run-grug-gpu-gate" and argv[3] == "--":
        environment = dict(os.environ)
        environment.update(grug_gpu_gate_environment(argv[2]))
        os.execvpe(argv[4], argv[4:], environment)
    raise SystemExit(
        f"usage: {argv[0]} write-frozen-cuda-runtime ACTIVATION_FILE | "
        "run-grug-gpu-gate REPOSITORY_ROOT -- COMMAND [ARG ...]"
    )


if __name__ == "__main__":
    _main(sys.argv)
