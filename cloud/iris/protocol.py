"""Stable JSON request and response types for MarinSkyRL jobs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from cloud.iris.runtime_environment import RuntimeProfile
from marinskyrl.task_sources import DataSource, data_source


class AttemptState(StrEnum):
    PREPARED = "prepared"
    SUBMITTED = "submitted"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class LaunchMode(StrEnum):
    PREPARE = "prepare"
    DETACH = "detach"
    WAIT = "wait"


class ModelRoleKind(StrEnum):
    POLICY = "policy"
    REFERENCE = "reference"
    CRITIC = "critic"
    ROLLOUT = "rollout"
    DRAFT_TRAINER = "draft_trainer"
    TEACHER = "teacher"


class RoleExecution(StrEnum):
    LOCAL = "local"
    REMOTE = "remote"


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
class ModelRoleClaim:
    """One active logical model role and its resolved execution claim."""

    role_id: str
    kind: ModelRoleKind
    execution: RoleExecution
    backend: str
    colocation_group: str | None
    num_nodes: int
    gpus_per_node: int
    replicas: int
    tensor_parallel_size: int
    pipeline_parallel_size: int
    data_parallel_size: int
    expert_parallel_size: int

    def __post_init__(self) -> None:
        if not self.role_id or not self.backend:
            raise ValueError("model role claims require non-empty role_id and backend")
        if self.execution is RoleExecution.REMOTE:
            if self.colocation_group is not None or self.num_nodes != 0 or self.gpus_per_node != 0:
                raise ValueError(f"remote role {self.role_id!r} cannot reserve an Iris bundle")
            if self.replicas <= 0:
                raise ValueError(f"remote role {self.role_id!r} requires at least one endpoint replica")
            return
        if not self.colocation_group:
            raise ValueError(f"local role {self.role_id!r} requires a named colocation group")
        geometry = (
            self.num_nodes,
            self.gpus_per_node,
            self.replicas,
            self.tensor_parallel_size,
            self.pipeline_parallel_size,
            self.data_parallel_size,
            self.expert_parallel_size,
        )
        if min(geometry) <= 0:
            raise ValueError(f"local role {self.role_id!r} requires positive execution geometry")
        if self.tensor_parallel_size > self.num_nodes * self.gpus_per_node:
            raise ValueError(
                f"local role {self.role_id!r} tensor parallelism {self.tensor_parallel_size} exceeds its "
                f"{self.num_nodes}x{self.gpus_per_node} physical footprint"
            )


@dataclass(frozen=True)
class RoleBundle:
    """One physical whole-node footprint shared by compatible local roles."""

    name: str
    role_ids: tuple[str, ...]
    num_nodes: int
    gpus_per_node: int

    def __post_init__(self) -> None:
        if not self.name or not self.role_ids or min(self.num_nodes, self.gpus_per_node) <= 0:
            raise ValueError("role bundles require a name, roles, and positive whole-node geometry")
        if len(set(self.role_ids)) != len(self.role_ids):
            raise ValueError(f"role bundle {self.name!r} contains duplicate roles")


@dataclass(frozen=True)
class SkyRLRolePlan:
    claims: tuple[ModelRoleClaim, ...]
    bundles: tuple[RoleBundle, ...]
    train_batch_size: int
    policy_mini_batch_size: int
    micro_train_batch_size_per_gpu: int
    n_samples_per_prompt: int

    def __post_init__(self) -> None:
        role_ids = [claim.role_id for claim in self.claims]
        if len(set(role_ids)) != len(role_ids):
            raise ValueError("role plan contains duplicate role claims")
        bundle_names = [bundle.name for bundle in self.bundles]
        if len(set(bundle_names)) != len(bundle_names):
            raise ValueError("role plan contains duplicate physical bundles")
        claims_by_id = {claim.role_id: claim for claim in self.claims}
        bundled_roles = [role_id for bundle in self.bundles for role_id in bundle.role_ids]
        local_roles = [claim.role_id for claim in self.claims if claim.execution is RoleExecution.LOCAL]
        if sorted(bundled_roles) != sorted(local_roles):
            raise ValueError("role bundles must cover every local role exactly once and no remote roles")
        for bundle in self.bundles:
            for role_id in bundle.role_ids:
                claim = claims_by_id[role_id]
                if claim.colocation_group != bundle.name:
                    raise ValueError(f"role {role_id!r} does not name its containing bundle {bundle.name!r}")
                if (claim.num_nodes, claim.gpus_per_node) != (bundle.num_nodes, bundle.gpus_per_node):
                    raise ValueError(f"role {role_id!r} physical geometry does not match bundle {bundle.name!r}")
        gpu_widths = {bundle.gpus_per_node for bundle in self.bundles}
        if len(gpu_widths) > 1:
            raise ValueError(f"Iris role bundles require one GPUs-per-node value; got {sorted(gpu_widths)}")

    def claim(self, role_id: str | ModelRoleKind) -> ModelRoleClaim:
        normalized_role_id = role_id.value if isinstance(role_id, ModelRoleKind) else role_id
        matches = [claim for claim in self.claims if claim.role_id == normalized_role_id]
        if len(matches) != 1:
            raise ValueError(f"role plan must contain exactly one {normalized_role_id!r} claim; found {len(matches)}")
        return matches[0]

    @property
    def colocate_all(self) -> bool:
        """Whether the policy-side roles and rollout share one bundle."""
        rollout = self.claim(ModelRoleKind.ROLLOUT)
        if rollout.execution is RoleExecution.REMOTE:
            return False
        groups = {
            claim.colocation_group
            for claim in self.claims
            if claim.execution is RoleExecution.LOCAL
            and claim.kind not in {ModelRoleKind.TEACHER, ModelRoleKind.DRAFT_TRAINER}
        }
        return len(groups) == 1


@dataclass(frozen=True)
class SkyRLTopology:
    num_nodes: int
    gpus_per_node: int
    gpu_variant: str
    role_plan: SkyRLRolePlan

    def __post_init__(self) -> None:
        planned_nodes = sum(bundle.num_nodes for bundle in self.role_plan.bundles)
        if self.num_nodes != planned_nodes:
            raise ValueError(f"topology num_nodes={self.num_nodes} does not match role bundles={planned_nodes}")
        planned_gpu_widths = {bundle.gpus_per_node for bundle in self.role_plan.bundles}
        if planned_gpu_widths and planned_gpu_widths != {self.gpus_per_node}:
            raise ValueError(
                f"topology gpus_per_node={self.gpus_per_node} does not match role bundles={sorted(planned_gpu_widths)}"
            )


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
    train_data: tuple[DataSource, ...]
    validation_data: tuple[DataSource, ...]
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
    wandb_entity: str | None


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
class SkyRLLaunchResponse:
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
            train_data=tuple(data_source(source) for source in request["train_data"]),
            validation_data=tuple(data_source(source) for source in request["validation_data"]),
            topology=SkyRLTopology(
                num_nodes=request["topology"]["num_nodes"],
                gpus_per_node=request["topology"]["gpus_per_node"],
                gpu_variant=request["topology"]["gpu_variant"],
                role_plan=_role_plan(request["topology"]["role_plan"]),
            ),
            output=SkyRLOutputPaths(**request["output"]),
            seed=int(request["seed"]),
            overrides=tuple(request.get("overrides", ())),
        ),
        execution=IrisLaunchOptions(**value["execution"]),
    )


def _role_plan(value: dict[str, Any]) -> SkyRLRolePlan:
    if "claims" not in value:
        return _legacy_role_plan(value)
    return SkyRLRolePlan(
        claims=tuple(
            ModelRoleClaim(
                role_id=claim["role_id"],
                kind=ModelRoleKind(claim["kind"]),
                execution=RoleExecution(claim["execution"]),
                backend=claim["backend"],
                colocation_group=claim["colocation_group"],
                num_nodes=int(claim["num_nodes"]),
                gpus_per_node=int(claim["gpus_per_node"]),
                replicas=int(claim["replicas"]),
                tensor_parallel_size=int(claim["tensor_parallel_size"]),
                pipeline_parallel_size=int(claim.get("pipeline_parallel_size", 1)),
                data_parallel_size=int(claim.get("data_parallel_size", 1)),
                expert_parallel_size=int(claim.get("expert_parallel_size", 1)),
            )
            for claim in value["claims"]
        ),
        bundles=tuple(
            RoleBundle(
                name=bundle["name"],
                role_ids=tuple(bundle["role_ids"]),
                num_nodes=int(bundle["num_nodes"]),
                gpus_per_node=int(bundle["gpus_per_node"]),
            )
            for bundle in value["bundles"]
        ),
        train_batch_size=int(value["train_batch_size"]),
        policy_mini_batch_size=int(value["policy_mini_batch_size"]),
        micro_train_batch_size_per_gpu=int(value["micro_train_batch_size_per_gpu"]),
        n_samples_per_prompt=int(value["n_samples_per_prompt"]),
    )


def _legacy_role_plan(value: dict[str, Any]) -> SkyRLRolePlan:
    """Upgrade pre-bundle requests still retained by Iris.

    Cloud/Iris owns this compatibility path. Remove it after requests and terminal
    manifests written before 2026-09-14 have passed their configured retention window.
    """
    policy_nodes = int(value["policy_num_nodes"])
    gpus_per_node = int(value["policy_num_gpus_per_node"])
    rollout_replicas = int(value["num_inference_engines"])
    colocate_all = bool(value["colocate_all"])
    policy_group = "all" if colocate_all else ModelRoleKind.POLICY.value
    rollout_group = "all" if colocate_all else ModelRoleKind.ROLLOUT.value
    claims = (
        ModelRoleClaim(
            role_id=ModelRoleKind.POLICY.value,
            kind=ModelRoleKind.POLICY,
            execution=RoleExecution.LOCAL,
            backend="legacy",
            colocation_group=policy_group,
            num_nodes=policy_nodes,
            gpus_per_node=gpus_per_node,
            replicas=policy_nodes * gpus_per_node,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=policy_nodes * gpus_per_node,
            expert_parallel_size=1,
        ),
        ModelRoleClaim(
            role_id=ModelRoleKind.REFERENCE.value,
            kind=ModelRoleKind.REFERENCE,
            execution=RoleExecution.LOCAL,
            backend="legacy",
            colocation_group=policy_group,
            num_nodes=policy_nodes,
            gpus_per_node=gpus_per_node,
            replicas=policy_nodes * gpus_per_node,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=policy_nodes * gpus_per_node,
            expert_parallel_size=1,
        ),
        ModelRoleClaim(
            role_id=ModelRoleKind.ROLLOUT.value,
            kind=ModelRoleKind.ROLLOUT,
            execution=RoleExecution.LOCAL,
            backend="legacy",
            colocation_group=rollout_group,
            num_nodes=policy_nodes if colocate_all else rollout_replicas,
            gpus_per_node=gpus_per_node,
            replicas=rollout_replicas,
            tensor_parallel_size=int(value["inference_engine_tensor_parallel_size"]),
            pipeline_parallel_size=1,
            data_parallel_size=int(value.get("inference_engine_data_parallel_size", 1)),
            expert_parallel_size=int(value.get("inference_engine_expert_parallel_size", 1)),
        ),
    )
    bundles = (
        RoleBundle(
            name=policy_group,
            role_ids=(
                (ModelRoleKind.POLICY.value, ModelRoleKind.REFERENCE.value, ModelRoleKind.ROLLOUT.value)
                if colocate_all
                else (ModelRoleKind.POLICY.value, ModelRoleKind.REFERENCE.value)
            ),
            num_nodes=policy_nodes,
            gpus_per_node=gpus_per_node,
        ),
    )
    if not colocate_all:
        bundles += (
            RoleBundle(
                name=rollout_group,
                role_ids=(ModelRoleKind.ROLLOUT.value,),
                num_nodes=rollout_replicas,
                gpus_per_node=gpus_per_node,
            ),
        )
    return SkyRLRolePlan(
        claims=claims,
        bundles=bundles,
        train_batch_size=int(value["train_batch_size"]),
        policy_mini_batch_size=int(value["policy_mini_batch_size"]),
        micro_train_batch_size_per_gpu=int(value["micro_train_batch_size_per_gpu"]),
        n_samples_per_prompt=int(value["n_samples_per_prompt"]),
    )
