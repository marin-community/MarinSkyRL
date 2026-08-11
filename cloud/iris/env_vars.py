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
from dataclasses import dataclass
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


@dataclass(frozen=True)
class EnvVarSpec:
    name: str
    owner: str
    source: EnvVarSource
    scopes: frozenset[EnvVarScope]


ALL_RUNTIME_SCOPES = frozenset(EnvVarScope)

DEBUG_MODE_ENV = "SKYRL_DEBUG_MODE"
DEBUG_ARTIFACT_DIR_ENV = "SKYRL_DEBUG_ARTIFACT_DIR"
FR_DUMP_TEMP_FILE_ENV = "TORCH_FR_DUMP_TEMP_FILE"
NCCL_DEBUG_INFO_TEMP_FILE_ENV = "TORCH_NCCL_DEBUG_INFO_TEMP_FILE"
PYTHONPATH_ENV = "PYTHONPATH"
VLLM_USE_V1_ENV = "VLLM_USE_V1"
VLLM_USE_DEEP_GEMM_ENV = "VLLM_USE_DEEP_GEMM"
WANDB_ENTITY_ENV = "WANDB_ENTITY"
HF_HUB_OFFLINE_ENV = "HF_HUB_OFFLINE"
LD_LIBRARY_PATH_ENV = "LD_LIBRARY_PATH"
NVRTC_HOME_ENV = "NVRTC_HOME"
DEFAULT_NCCL_TRACE_BUFFER_SIZE = 20_000


ENV_VAR_SPECS = (
    EnvVarSpec(DEBUG_MODE_ENV, "trainer.debug_mode", EnvVarSource.CONFIG, ALL_RUNTIME_SCOPES),
    EnvVarSpec(DEBUG_ARTIFACT_DIR_ENV, "trainer.debug_mode", EnvVarSource.DERIVED, ALL_RUNTIME_SCOPES),
    EnvVarSpec("NCCL_DEBUG", "trainer.debug_mode", EnvVarSource.DERIVED, ALL_RUNTIME_SCOPES),
    EnvVarSpec("NCCL_DEBUG_SUBSYS", "trainer.debug_mode", EnvVarSource.DERIVED, ALL_RUNTIME_SCOPES),
    EnvVarSpec("NCCL_DEBUG_FILE", "trainer.debug_mode", EnvVarSource.DERIVED, ALL_RUNTIME_SCOPES),
    EnvVarSpec("PYTHONFAULTHANDLER", "trainer.debug_mode", EnvVarSource.DERIVED, ALL_RUNTIME_SCOPES),
    EnvVarSpec("SKYRL_COLLECTIVE_PHASE_DIAGNOSTICS", "trainer.debug_mode", EnvVarSource.DERIVED, ALL_RUNTIME_SCOPES),
    EnvVarSpec("SKYRL_POLICY_HOST_RAM_MONITOR", "trainer.debug_mode", EnvVarSource.DERIVED, ALL_RUNTIME_SCOPES),
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
)

_SPECS_BY_NAME = {spec.name: spec for spec in ENV_VAR_SPECS}
if len(_SPECS_BY_NAME) != len(ENV_VAR_SPECS):
    raise RuntimeError("Environment variable names must have exactly one EnvVarManager owner")

_REMOTE_PATH = re.compile(r"^[a-z][a-z0-9+.-]*://", re.IGNORECASE)
_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9_.-]+")
_NCCL_SETUP_SUBSYSTEMS = "INIT,BOOTSTRAP,ENV,NET,GRAPH,TUNING"


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
        raw_mode = ambient.get(DEBUG_MODE_ENV, _config_value(config, "trainer.debug_mode", "off"))
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
            "SKYRL_COLLECTIVE_PHASE_DIAGNOSTICS": "1",
            "SKYRL_POLICY_HOST_RAM_MONITOR": "1",
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
        if values:
            ensure_debug_artifact_directories(values[DEBUG_ARTIFACT_DIR_ENV])
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
