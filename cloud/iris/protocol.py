"""Stable JSON request and response types for MarinSkyRL jobs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from cloud.iris.runtime_environment import RuntimeProfile


class AttemptState(StrEnum):
    PREPARED = "prepared"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


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


@dataclass(frozen=True)
class SkyRLJobSpec:
    request: SkyRLLaunchRequest
    execution: IrisLaunchOptions


@dataclass(frozen=True)
class SkyRLModel:
    policy_export_uri: str
    global_step: int
    tokenizer_uri: str
    tokenizer_revision: str
    checkpoint_root: str
    terminal_manifest_uri: str


@dataclass(frozen=True)
class SkyRLTerminalResponse:
    run_id: str
    attempt_id: str
    state: AttemptState
    iris_job_id: str | None
    iris_job_state: str | None
    runtime: RuntimeIdentity
    model: SkyRLModel | None
    failure: str | None


def job_spec(value: dict[str, Any]) -> SkyRLJobSpec:
    """Parse one JSON-compatible job specification."""
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
        ),
        execution=IrisLaunchOptions(**value["execution"]),
    )
