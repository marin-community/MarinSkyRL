"""Stable JSON request and response types for MarinSkyRL jobs."""

from __future__ import annotations

import hashlib
import json
import posixpath
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from cloud.iris.runtime_environment import RuntimeProfile
from marinskyrl.training_completion import CompletionMode, NativeCheckpoint

PROTOCOL_VERSION = 2


class AttemptState(StrEnum):
    PREPARED = "prepared"
    SUBMITTED = "submitted"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class LaunchMode(StrEnum):
    PREPARE = "prepare"
    DETACH = "detach"
    WAIT = "wait"


@dataclass(frozen=True)
class RuntimeIdentity:
    commit: str
    profile: RuntimeProfile


@dataclass(frozen=True)
class ModelLocator:
    uri: str
    identity: str
    local_path: str
    tokenizer_uri: str
    tokenizer_revision: str


@dataclass(frozen=True)
class DataLocator:
    uri: str
    identity: str
    local_path: str
    relative_path: str


@dataclass(frozen=True)
class SkyRLRolePlan:
    colocate_all: bool
    policy_num_nodes: int
    policy_num_gpus_per_node: int
    num_inference_engines: int
    inference_engine_tensor_parallel_size: int
    train_batch_size: int
    policy_mini_batch_size: int
    micro_train_batch_size_per_gpu: int
    n_samples_per_prompt: int


@dataclass(frozen=True)
class SkyRLTopology:
    num_nodes: int
    gpus_per_node: int
    gpu_variant: str
    role_plan: SkyRLRolePlan


@dataclass(frozen=True)
class SkyRLOutputPaths:
    checkpoint_root: str
    export_root: str
    attempts_root: str
    resolved_config_uri: str
    terminal_manifest_uri: str


@dataclass(frozen=True)
class SkyRLLaunchRequest:
    run_id: str
    attempt_id: str
    config_yaml: str
    runtime: RuntimeIdentity
    model: ModelLocator
    train_data: tuple[DataLocator, ...]
    validation_data: tuple[DataLocator, ...]
    topology: SkyRLTopology
    output: SkyRLOutputPaths
    seed: int
    overrides: tuple[str, ...]
    completion_mode: CompletionMode = CompletionMode.CHECKPOINT
    checkpoint_retention_days: int | None = None


@dataclass(frozen=True)
class IrisLaunchOptions:
    cluster: str
    cluster_config: str
    cpu: float
    memory: str
    disk: str
    target_cluster: str | None
    parent_cluster_config: str | None
    priority: str
    max_retries: int
    job_name: str
    wandb_entity: str | None
    timeout_seconds: int = 0

    def __post_init__(self) -> None:
        if self.timeout_seconds < 0:
            raise ValueError("timeout_seconds must be nonnegative (0 disables the job deadline)")


@dataclass(frozen=True)
class SkyRLJobSpec:
    request: SkyRLLaunchRequest
    execution: IrisLaunchOptions
    schema_version: int = PROTOCOL_VERSION


@dataclass(frozen=True)
class SkyRLTrainingResult:
    global_step: int
    receipt_uri: str
    resolved_config_uri: str
    checkpoint: NativeCheckpoint | None


@dataclass(frozen=True)
class SkyRLExportPaths:
    export_root: str
    attempts_root: str
    terminal_manifest_uri: str


@dataclass(frozen=True)
class SkyRLExportRequest:
    training_manifest_uri: str
    attempt_id: str
    output: SkyRLExportPaths


@dataclass(frozen=True)
class SkyRLExportSpec:
    request: SkyRLExportRequest
    execution: IrisLaunchOptions
    schema_version: int = PROTOCOL_VERSION


@dataclass(frozen=True)
class SkyRLModel:
    policy_export_uri: str
    global_step: int
    tokenizer_uri: str
    tokenizer_revision: str
    checkpoint_root: str
    terminal_manifest_uri: str


@dataclass(frozen=True)
class SkyRLLaunchResponse:
    run_id: str
    attempt_id: str
    state: AttemptState
    iris_job_id: str | None
    iris_job_state: str | None
    runtime: RuntimeIdentity
    training: SkyRLTrainingResult | None
    failure: str | None


@dataclass(frozen=True)
class SkyRLExportResponse:
    run_id: str
    attempt_id: str
    state: AttemptState
    runtime: RuntimeIdentity
    training_iris_job_id: str | None
    model: SkyRLModel | None
    failure: str | None
    reused_export: bool = False


def training_receipt_uri(request: SkyRLLaunchRequest) -> str:
    return posixpath.join(
        posixpath.dirname(request.output.terminal_manifest_uri), "receipts", f"{request.attempt_id}.json"
    )


def training_request_fingerprint(request: SkyRLLaunchRequest) -> str:
    payload = {
        "schema_version": PROTOCOL_VERSION,
        "request": asdict(request),
        "receipt_uri": training_receipt_uri(request),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _validate_protocol(value: dict[str, Any]) -> None:
    if value.get("schema_version") != PROTOCOL_VERSION:
        raise ValueError(f"MarinSkyRL protocol schema_version must be {PROTOCOL_VERSION}")


def export_spec(value: dict[str, Any]) -> SkyRLExportSpec:
    _validate_protocol(value)
    request = value["request"]
    return SkyRLExportSpec(
        request=SkyRLExportRequest(
            training_manifest_uri=request["training_manifest_uri"],
            attempt_id=request["attempt_id"],
            output=SkyRLExportPaths(**request["output"]),
        ),
        execution=IrisLaunchOptions(**value["execution"]),
    )


def job_spec(value: dict[str, Any]) -> SkyRLJobSpec:
    """Parse one JSON-compatible job specification."""
    _validate_protocol(value)
    request = value["request"]
    return SkyRLJobSpec(
        request=SkyRLLaunchRequest(
            run_id=request["run_id"],
            attempt_id=request["attempt_id"],
            config_yaml=request["config_yaml"],
            runtime=RuntimeIdentity(
                commit=request["runtime"]["commit"],
                profile=RuntimeProfile(request["runtime"]["profile"]),
            ),
            model=ModelLocator(**request["model"]),
            train_data=tuple(DataLocator(**locator) for locator in request["train_data"]),
            validation_data=tuple(DataLocator(**locator) for locator in request["validation_data"]),
            topology=SkyRLTopology(
                num_nodes=request["topology"]["num_nodes"],
                gpus_per_node=request["topology"]["gpus_per_node"],
                gpu_variant=request["topology"]["gpu_variant"],
                role_plan=SkyRLRolePlan(**request["topology"]["role_plan"]),
            ),
            output=SkyRLOutputPaths(**request["output"]),
            seed=int(request["seed"]),
            overrides=tuple(request.get("overrides", ())),
            completion_mode=CompletionMode(request["completion_mode"]),
            checkpoint_retention_days=request["checkpoint_retention_days"],
        ),
        execution=IrisLaunchOptions(**value["execution"]),
    )
