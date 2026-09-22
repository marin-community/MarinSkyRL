#!/usr/bin/env python3
"""Derive and submit an Iris GPU job from one validated Hydra launch config.

The resolved document supplies allocation, routing, task environment, storage,
and Ray settings. This module projects those values into the Iris SDK and sends
the same document to every task; it has no user-facing option parser.
"""

from __future__ import annotations

import base64
import contextlib
import datetime
import hashlib
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional
from urllib.parse import unquote, urlparse

import yaml
from iris.client.client import IrisClient, Job
from iris.cluster.constraints import (
    CLUSTER_CONSTRAINT_KEY,
    Constraint,
    ConstraintOp,
    infer_preemptible_constraint,
    preemptible_constraint,
)
from iris.cluster.platforms.k8s.coreweave_topology import gpu_gang_coscheduling_level
from iris.cluster.types import CoschedulingConfig, ResourceSpec, gpu_device
from iris.resources.state import JobState
from iris.rpc import job_pb2
from omegaconf import DictConfig, OmegaConf

from cloud.iris.paths import PROJECT_ROOT
from cloud.iris.launch_config import SubmissionMode, load_launch_config, validate_launch_config
from cloud.iris.export_hf_checkpoint import export_terminal_policy
from cloud.iris.ingress_utils import (
    PARENT_CONTROLLER_CONFIG_ENV,
    PARENT_CONTROLLER_CONFIG_YAML_ENV,
    PARENT_CREDENTIALS_JSON_ENV,
    PARENT_IAP_TOKEN_ENV,
)
from cloud.iris.ray_storage import (
    RaySpillBackend,
    resolve_ray_spill_target,
)
from cloud.iris.rl_config_translation import RL_CONFIG_PAYLOAD_ENV, RL_CONFIG_TASK_DIR
from marinskyrl.resource_locator import (
    is_cloud_uri,
    join_resource_path,
)
from cloud.iris.secrets_env import load_secrets_env_into_os_environ
from cloud.iris.runtime_bundle import build_runtime_bundle
from marinskyrl.environment_contract import (
    DEBUG_ARTIFACT_DIR_ENV,
    DEBUG_MODE_ENV,
    DebugMode,
    EnvVarManager,
    EnvVarScope,
    wandb_launch_environment,
)
from cloud.iris.runtime_environment import (
    CHECKPOINT_EXPORT_ENTRYPOINT,
    MARINSKYRL_ACTIVATION_FILE,
    MARINSKYRL_TASK_ROOT,
    RuntimeProfile,
    task_setup_script,
)

# Memory and disk requests may be resolved from the selected cluster's live nodes
# before Iris submission.
DAYTONA_RL_SECRET_PROJECT = "hai-gcp-models"
DAYTONA_RL_SECRET_NAME = "DAYTONA_RL_API_KEY"
DAYTONA_RL_SECRET_VERSION = "1"


# The RL Daytona org enforces a 40-snapshot quota. Harbor mints one "harbor__*" env
# snapshot per trial (auto_snapshot=true); once the org is over quota, snapshot creation
# fails and harbor's fallthrough attempts a declarative sandbox build, which this org
# forbids, so every trial then dies unscored with DaytonaValidationError and the job trains
# on all-zero rewards. Purging harbor-minted snapshots idle past this age before every
# launch keeps quota headroom so harbor's worker-side minting can self-heal.
DAYTONA_RL_SNAPSHOT_QUOTA = 40
HARBOR_SNAPSHOT_NAME_PREFIX = "harbor__"
STALE_SNAPSHOT_MAX_AGE = datetime.timedelta(hours=2)
AUTOMATIC_RESOURCE_REQUEST = "auto"
MEMORY_RESOURCE = "memory"
DISK_RESOURCE = "ephemeral-storage"
# Leave the remainder of live allocatable RAM and disk to kubelet, daemonsets,
# and filesystem overhead.
NODE_RESOURCE_FRACTION = 0.80


@dataclass(frozen=True)
class RLStoragePaths:
    """Resolved storage paths used for launch diagnostics and retention."""

    checkpoint_root: str
    export_root: str
    trace_root: str
    trajectory_root: str
    ray_log_root: str
    resume_checkpoint_count: int


def _is_checkpoint_export(args: SimpleNamespace) -> bool:
    return getattr(args, "entrypoint", None) == CHECKPOINT_EXPORT_ENTRYPOINT


def _parse_quantity_to_gib(q: str) -> float:
    """Parse a Kubernetes or Iris byte quantity to GiB."""
    q = q.strip()
    for suf, mult in (("Ki", 2**10), ("Mi", 2**20), ("Gi", 2**30), ("Ti", 2**40), ("Pi", 2**50)):
        if q.endswith(suf):
            return float(q[: -len(suf)]) * mult / 2**30
    for suf, mult in (
        ("KB", 1e3),
        ("MB", 1e6),
        ("GB", 1e9),
        ("TB", 1e12),
        ("PB", 1e15),
        ("k", 1e3),
        ("M", 1e6),
        ("G", 1e9),
        ("T", 1e12),
        ("P", 1e15),
    ):
        if q.endswith(suf):
            return float(q[: -len(suf)]) * mult / 2**30
    return float(q) / 2**30  # plain bytes


@dataclass(frozen=True)
class ResourceQuantities:
    """Per-node memory and disk quantities in GiB."""

    memory_gib: float
    disk_gib: float


@dataclass(frozen=True)
class NodeResourceBudget:
    """Live headroom and policy-capped automatic requests for one node."""

    headroom: ResourceQuantities
    automatic: ResourceQuantities


@dataclass(frozen=True)
class ClusterResourceSnapshot:
    """Available matching nodes observed in one Kubernetes query."""

    context: str
    gpu_variant: str
    gpus_per_node: int
    nodes: tuple[NodeResourceBudget, ...]


@dataclass(frozen=True)
class ResolvedResourceRequests:
    """Per-node memory and disk requests ready for Iris submission."""

    memory: str
    disk: str


@dataclass(frozen=True)
class IrisLaunchOutcome:
    """Result of one Iris submission attempt."""

    job_id: str
    job_state: str
    exit_code: int


class _LauncherTermination(BaseException):
    """A process termination signal converted into supervised job cancellation."""

    def __init__(self, signum: int):
        self.signum = signum
        super().__init__(f"launcher received signal {signum}")


@contextlib.contextmanager
def _supervised_termination_signals() -> Iterator[None]:
    previous = signal.getsignal(signal.SIGTERM)

    def request_termination(signum, _frame) -> None:
        raise _LauncherTermination(signum)

    signal.signal(signal.SIGTERM, request_termination)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def _cancel_iris_job_tree(job: Job, job_id: str, cause: BaseException) -> None:
    print(f"[rl-iris] Cancelling job tree {job_id}...", file=sys.stderr, flush=True)
    try:
        job.cancel()
    except BaseException as cleanup_error:
        cause.add_note(f"Failed to confirm cancellation of Iris job tree {job_id}: {cleanup_error}")
        raise cause from cleanup_error
    print(f"[rl-iris] Cancelled job tree {job_id}.", file=sys.stderr, flush=True)


def supervise_iris_job(job: Job) -> IrisLaunchOutcome:
    """Wait for a terminal state, cancelling the job tree if supervision cannot continue."""
    job_id = str(job.job_id)
    try:
        with _supervised_termination_signals():
            status = job.wait(stream_logs=True, timeout=float("inf"), raise_on_failure=False)
    except (KeyboardInterrupt, _LauncherTermination) as interruption:
        _cancel_iris_job_tree(job, job_id, interruption)
        exit_code = 130 if isinstance(interruption, KeyboardInterrupt) else 128 + interruption.signum
        return IrisLaunchOutcome(job_id=job_id, job_state=JobState.KILLED.value, exit_code=exit_code)
    except BaseException as error:
        _cancel_iris_job_tree(job, job_id, error)
        raise
    return IrisLaunchOutcome(
        job_id=job_id,
        job_state=status.state.value,
        exit_code=0 if status.state is JobState.SUCCEEDED else 1,
    )


def _gpu_resources(gpu_variant: str, gpu_count: int, *, cpu: float, memory: str, disk: str) -> ResourceSpec:
    resources = ResourceSpec(cpu=cpu, memory=memory, disk=disk)
    resources.device = gpu_device(gpu_variant, gpu_count)
    return resources


def _gpu_multinode(gpu_variant: str, gpu_count: int, replicas: int) -> CoschedulingConfig | None:
    if replicas <= 1:
        return None
    level = gpu_gang_coscheduling_level(gpu_variant, gpu_count, replicas)
    return CoschedulingConfig(group_by=level)


def _gpu_constraints(
    resources: job_pb2.ResourceSpecProto,
    *,
    replicas: int,
    preemptible: bool | None,
    target_cluster: str | None,
) -> list[Constraint]:
    constraints = []
    if preemptible is not None:
        constraints.append(preemptible_constraint(preemptible))
    if target_cluster:
        constraints.append(Constraint.create(key=CLUSTER_CONSTRAINT_KEY, op=ConstraintOp.EQ, value=target_cluster))
    inferred = infer_preemptible_constraint(resources, replicas, constraints)
    if inferred is not None:
        constraints.append(inferred)
    return constraints


def _model_path(uri: str) -> str:
    """Resolve a typed model URI into the task runtime's model reference."""
    if is_cloud_uri(uri):
        return uri
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        if parsed.netloc not in ("", "localhost"):
            raise ValueError(f"Model file URI must be local: {uri!r}")
        return unquote(parsed.path)
    return uri


def _iris_submission_state(config_path: Path, config: DictConfig) -> SimpleNamespace:
    """Build the in-memory Iris launch state from one validated Hydra config."""
    validate_launch_config(config)
    raw = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    assert isinstance(raw, dict)
    run = raw["run"]
    runtime = raw["runtime"]
    iris = raw["iris"]
    allocation = iris["allocation"]
    ray = raw["ray"]
    artifacts = raw["artifacts"]
    inputs = raw["inputs"]
    ingress = raw["ingress"]
    skyrl = raw["skyrl"]
    model = inputs["model"]
    model_path = _model_path(model["uri"])
    terminal_bench = skyrl.get("terminal_bench_config") or {}
    trajectory_retention = (skyrl.get("generator") or {}).get("trajectory_retention") or {}
    contents = config_path.read_bytes()
    digest = hashlib.sha256(contents).hexdigest()[:16]
    suffix = config_path.suffix or ".yaml"
    task_config = BundledLaunchConfig(
        task_path=f"{RL_CONFIG_TASK_DIR}/{digest}{suffix}",
        payload=base64.b64encode(contents).decode("ascii"),
    )
    submission = run["submission"]
    args = SimpleNamespace(
        rl_config=str(config_path),
        rl_config_launch=task_config,
        entrypoint=runtime["entrypoint"],
        model_path=model_path,
        resume_checkpoints_to_keep=int(artifacts["resume_checkpoint_count"]),
        num_nodes=int(allocation["num_nodes"]),
        gpus_per_node=int(allocation["gpus_per_node"]),
        gpu_variant=allocation["gpu_variant"],
        cpu=float(allocation["cpu"]),
        memory=allocation["memory"],
        disk=allocation["disk"],
        ray_port=int(ray["port"]),
        ray_spill_dir=ray["spill_dir"],
        ray_spill_backend=RaySpillBackend(ray["spill_backend"]),
        rendezvous_dir=ray["rendezvous_dir"],
        cluster=iris["cluster"],
        cluster_config=iris["cluster_config"],
        runtime_commit=runtime["launcher_commit"],
        runtime_profile=RuntimeProfile(runtime["profile"]),
        job_name=iris["job_name"],
        priority=iris["priority"],
        max_retries=int(iris["max_retries"]),
        timeout=int(iris["timeout"]),
        no_wait=submission == SubmissionMode.DETACH,
        dry_run=submission == SubmissionMode.PREPARE,
        preemptible=None,
        ingress_mode=ingress["mode"],
        ingress_host=ingress["host"] or None,
        target_cluster=iris["target_cluster"],
        parent_cluster_config=iris["parent_cluster_config"],
        secrets_env=None,
        wandb_entity=iris["wandb_entity"],
    )
    args.storage_paths = RLStoragePaths(
        checkpoint_root=artifacts["checkpoint_root"],
        export_root=artifacts["export_root"],
        trace_root=terminal_bench.get("trials_dir") or join_resource_path(artifacts["attempts_root"], "trace_jobs"),
        trajectory_root=trajectory_retention.get("output_path")
        or join_resource_path(artifacts["attempts_root"], "trajectories"),
        ray_log_root=ray["log_dir"],
        resume_checkpoint_count=args.resume_checkpoints_to_keep,
    )
    validate_controller_ingress_reachability(args)
    return args


class IrisBackend:
    """Validate and submit one resolved Hydra launch config."""

    def validate(self, config_path: Path) -> None:
        _iris_submission_state(config_path, load_launch_config(config_path))

    def launch(self, config_path: Path) -> IrisLaunchOutcome:
        config = load_launch_config(config_path)
        args = _iris_submission_state(config_path, config)
        if config.run.submission == SubmissionMode.PREPARE:
            raise ValueError("Prepare mode validates a launch without submitting it")
        with contextlib.redirect_stdout(sys.stderr):
            return launch(args, config.runtime.launcher_commit)

    def export_terminal_policy(self, config_path: Path) -> None:
        """Export the terminal checkpoint described by a completed training config."""
        export_terminal_policy(config_path)


def _pod_resource_request_gib(pod: dict[str, Any], resource: str) -> float:
    """Return the scheduler-effective request for one pod resource."""
    spec = pod.get("spec", {})

    def request(container: dict[str, Any]) -> float:
        quantity = container.get("resources", {}).get("requests", {}).get(resource)
        return _parse_quantity_to_gib(quantity) if isinstance(quantity, str) else 0.0

    application_request = sum(request(container) for container in spec.get("containers", []))
    init_request = max((request(container) for container in spec.get("initContainers", [])), default=0.0)
    overhead = spec.get("overhead", {}).get(resource)
    overhead_request = _parse_quantity_to_gib(overhead) if isinstance(overhead, str) else 0.0
    return max(application_request, init_request) + overhead_request


def _node_is_ready(node: dict[str, Any]) -> bool:
    conditions = node.get("status", {}).get("conditions", [])
    return any(condition.get("type") == "Ready" and condition.get("status") == "True" for condition in conditions)


def _node_resource_budgets(
    items: list[dict[str, Any]],
    *,
    gpu_variant: str,
    gpus_per_node: int,
) -> list[NodeResourceBudget]:
    """Return live headroom and automatic request budgets for matching Ready nodes."""
    pod_requests: dict[str, ResourceQuantities] = {}
    for item in items:
        if item.get("kind") != "Pod" or item.get("status", {}).get("phase") in {"Succeeded", "Failed"}:
            continue
        node_name = item.get("spec", {}).get("nodeName")
        if not isinstance(node_name, str):
            continue
        requested = pod_requests.get(node_name, ResourceQuantities(memory_gib=0.0, disk_gib=0.0))
        pod_requests[node_name] = ResourceQuantities(
            memory_gib=requested.memory_gib + _pod_resource_request_gib(item, MEMORY_RESOURCE),
            disk_gib=requested.disk_gib + _pod_resource_request_gib(item, DISK_RESOURCE),
        )

    available = []
    for item in items:
        if item.get("kind") != "Node" or item.get("spec", {}).get("unschedulable") or not _node_is_ready(item):
            continue
        metadata = item.get("metadata", {})
        allocatable = item.get("status", {}).get("allocatable", {})
        product = metadata.get("labels", {}).get("nvidia.com/gpu.product", "")
        if allocatable.get("nvidia.com/gpu") != str(gpus_per_node) or gpu_variant.lower() not in product.lower():
            continue
        node_name = metadata.get("name")
        if not isinstance(node_name, str):
            continue
        memory_allocatable = _parse_quantity_to_gib(allocatable[MEMORY_RESOURCE])
        disk_allocatable = _parse_quantity_to_gib(allocatable[DISK_RESOURCE])
        requested = pod_requests.get(node_name, ResourceQuantities(memory_gib=0.0, disk_gib=0.0))
        headroom = ResourceQuantities(
            memory_gib=memory_allocatable - requested.memory_gib,
            disk_gib=disk_allocatable - requested.disk_gib,
        )
        available.append(
            NodeResourceBudget(
                headroom=headroom,
                automatic=ResourceQuantities(
                    memory_gib=min(memory_allocatable * NODE_RESOURCE_FRACTION, headroom.memory_gib),
                    disk_gib=min(disk_allocatable * NODE_RESOURCE_FRACTION, headroom.disk_gib),
                ),
            )
        )
    return available


def _inspect_cluster_resources(
    cluster_config_path: str, *, gpu_variant: str, gpus_per_node: int
) -> ClusterResourceSnapshot:
    """Read one live node-and-pod resource snapshot from the selected cluster."""
    cluster_config = _load_cluster_config(cluster_config_path)
    platform = cluster_config.get("platform")
    coreweave = platform.get("coreweave") if isinstance(platform, dict) else None
    kubeconfig = coreweave.get("kubeconfig_path") if isinstance(coreweave, dict) else None
    context = coreweave.get("kube_context") if isinstance(coreweave, dict) else None
    if not isinstance(kubeconfig, str) or not isinstance(context, str):
        raise SystemExit(
            "The selected cluster config needs platform.coreweave.kubeconfig_path and kube_context "
            "to derive automatic memory and disk requests; pass both explicitly."
        )

    try:
        out = subprocess.run(
            [
                "kubectl",
                "--kubeconfig",
                str(Path(kubeconfig).expanduser()),
                "--context",
                context,
                "get",
                "nodes,pods",
                "--all-namespaces",
                "-o",
                "json",
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        ).stdout
    except Exception as exc:  # noqa: BLE001 - convert cluster/tool failures into launch guidance
        raise SystemExit(
            f"Could not inspect {gpus_per_node}x{gpu_variant} nodes in kube context {context!r}: {exc}. "
            "Pass explicit --memory and --disk values."
        ) from exc
    items = json.loads(out)["items"]
    if not isinstance(items, list):
        raise ValueError("kubectl node-and-pod snapshot has a non-list items field")
    nodes = _node_resource_budgets(items, gpu_variant=gpu_variant, gpus_per_node=gpus_per_node)
    return ClusterResourceSnapshot(
        context=context,
        gpu_variant=gpu_variant,
        gpus_per_node=gpus_per_node,
        nodes=tuple(nodes),
    )


def _resource_request_is_automatic(request: str) -> bool:
    return request.strip().lower() == AUTOMATIC_RESOURCE_REQUEST


def _resolve_gang_resource_requests(
    snapshot: ClusterResourceSnapshot,
    *,
    num_nodes: int,
    memory_request: str,
    disk_request: str,
) -> ResolvedResourceRequests:
    """Select one resource pair that fits the requested gang in a live snapshot."""
    automatic_memory = _resource_request_is_automatic(memory_request)
    automatic_disk = _resource_request_is_automatic(disk_request)
    requested_memory_gib = None if automatic_memory else _parse_quantity_to_gib(memory_request)
    requested_disk_gib = None if automatic_disk else _parse_quantity_to_gib(disk_request)
    candidates = [
        resources
        for resources in snapshot.nodes
        if (requested_memory_gib is None or resources.headroom.memory_gib >= requested_memory_gib)
        and (requested_disk_gib is None or resources.headroom.disk_gib >= requested_disk_gib)
    ]
    if len(candidates) < num_nodes:
        raise SystemExit(
            f"Automatic resources need {num_nodes} Ready, schedulable "
            f"{snapshot.gpus_per_node}x{snapshot.gpu_variant} nodes, "
            f"but only {len(candidates)} of {len(snapshot.nodes)} matching nodes in kube context "
            f"{snapshot.context!r} fit the constraints memory={memory_request}, disk={disk_request}; "
            "adjust the resource requests."
        )

    if automatic_memory:
        selected = sorted(candidates, key=lambda resources: resources.automatic.memory_gib, reverse=True)[:num_nodes]
    else:
        selected = sorted(candidates, key=lambda resources: resources.automatic.disk_gib, reverse=True)[:num_nodes]
    memory_gib = int(min(resources.automatic.memory_gib for resources in selected))
    disk_gib = int(min(resources.automatic.disk_gib for resources in selected))
    if (automatic_memory and memory_gib < 1) or (automatic_disk and disk_gib < 1):
        automatic_resources = " and ".join(
            resource for resource, automatic in (("memory", automatic_memory), ("disk", automatic_disk)) if automatic
        )
        raise SystemExit(
            f"No positive automatic {automatic_resources} request fits {num_nodes} "
            f"{snapshot.gpus_per_node}x{snapshot.gpu_variant} nodes "
            f"in kube context {snapshot.context!r}; pass explicit --memory and --disk values."
        )
    resolved_memory = f"{memory_gib}Gi" if automatic_memory else memory_request
    resolved_disk = f"{disk_gib}Gi" if automatic_disk else disk_request
    return ResolvedResourceRequests(memory=resolved_memory, disk=resolved_disk)


def resolve_node_resource_requests(
    cluster_config_path: str,
    *,
    gpu_variant: str,
    gpus_per_node: int,
    num_nodes: int,
    memory_request: str,
    disk_request: str,
) -> ResolvedResourceRequests:
    """Return live admission-aware resource requests for a GPU gang."""
    snapshot = _inspect_cluster_resources(
        cluster_config_path,
        gpu_variant=gpu_variant,
        gpus_per_node=gpus_per_node,
    )
    resolved = _resolve_gang_resource_requests(
        snapshot,
        num_nodes=num_nodes,
        memory_request=memory_request,
        disk_request=disk_request,
    )
    print(
        f"[rl-iris] automatic node resources: largest requests with live headroom on {num_nodes} of "
        f"{len(snapshot.nodes)} matching nodes, capped at {NODE_RESOURCE_FRACTION:.0%} allocatable = "
        f"memory {resolved.memory}, disk {resolved.disk}",
        flush=True,
    )
    return resolved


RL_PYTHON = "python"
SKYRL_HOME = MARINSKYRL_TASK_ROOT
# Iris synchronizes the small controller bundle to /app. The setup phase checks
# out the same immutable MarinSkyRL commit and installs its locked training
# environment under /app/marinskyrl.
APP_DIR = "/app"


@dataclass(frozen=True)
class BundledLaunchConfig:
    """Task path and environment payload for one resolved launch config."""

    task_path: str
    payload: str

    def task_environment(self) -> dict[str, str]:
        """Return the environment needed to materialize the config."""
        return {RL_CONFIG_PAYLOAD_ENV: self.payload}


MARIN_LOGIN_RECORD_PATH = Path.home() / ".config" / "marin" / "credentials" / "marin.json"
_JOB_NAME_MAX_LENGTH = 63


def _resolve_parent_cluster_config(cluster_config: Optional[str]) -> Optional[str]:
    """Path to the PARENT (marin) cluster YAML for federated submission.

    The marin meta-scheduler config (marin.yaml) that owns iris.oa.dev and lists the
    CoreWeave clusters as delegation peers. Defaults to the ``marin.yaml`` sibling of
    ``--cluster-config`` (they live in the same ``lib/iris/config/`` dir); falls back
    to the same search roots as :func:`_resolve_cluster_config_default`.
    """
    if cluster_config:
        sib = Path(cluster_config).with_name("marin.yaml")
        if sib.exists():
            return str(sib)
    rel = "lib/iris/config/marin.yaml"
    for c in (
        Path.home() / "Documents/marin" / rel,
        Path("/Users/benjaminfeuer/Documents/marin") / rel,
        Path(os.environ.get("MARIN_ROOT", "")) / rel,
    ):
        if c.exists():
            return str(c)
    return None


def _load_cluster_config(cluster_config: str) -> dict[str, Any]:
    """Load the selected Iris cluster configuration for launch-time defaults."""
    try:
        with open(cluster_config) as f:
            loaded = yaml.safe_load(f)
    except OSError as exc:
        raise SystemExit(
            f"Could not load --cluster-config {cluster_config!r} to resolve RL launch defaults: {exc}. "
            "Pass an existing cluster config or explicit --cpu and --rendezvous-dir values."
        ) from exc
    if not isinstance(loaded, dict):
        raise SystemExit(f"--cluster-config {cluster_config!r} must contain a YAML mapping.")
    return loaded


def _cluster_gpu_cpu_capacity(cluster_config: dict[str, Any], *, gpu_variant: str, gpus_per_node: int) -> float:
    """Return CPU capacity for the matching GPU scale group in an Iris config."""
    scale_groups = cluster_config.get("scale_groups")
    if not isinstance(scale_groups, dict):
        raise SystemExit("The selected Iris cluster config has no scale_groups mapping; pass --cpu explicitly.")
    for scale_group in scale_groups.values():
        resources = scale_group.get("resources") if isinstance(scale_group, dict) else None
        if not isinstance(resources, dict):
            continue
        if (
            resources.get("device_type") == "gpu"
            and str(resources.get("device_variant", "")).lower() == gpu_variant.lower()
            and resources.get("device_count") == gpus_per_node
        ):
            cpu = resources.get("cpu")
            if isinstance(cpu, (int, float)) and cpu > 0:
                return float(cpu)
    raise SystemExit(
        f"The selected Iris cluster config has no {gpus_per_node}x{gpu_variant} GPU scale group. "
        "Choose a topology that the selected cluster advertises."
    )


def _daytona_rl_api_key_from_secret_manager() -> Optional[str]:
    """The pinned RL Daytona key from Secret Manager, or ``None`` if gcloud is
    missing/denied or the secret is empty. Never logged."""
    command = [
        "gcloud",
        "secrets",
        "versions",
        "access",
        DAYTONA_RL_SECRET_VERSION,
        f"--secret={DAYTONA_RL_SECRET_NAME}",
        f"--project={DAYTONA_RL_SECRET_PROJECT}",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _resolve_daytona_rl_api_key() -> str:
    """Resolve the RL-org Daytona key from the launch environment or Secret Manager.

    ``--secrets-env`` is loaded before this function runs. Keep its RL-specific name at
    the launch boundary, then let the caller expose the resolved value as
    ``DAYTONA_API_KEY`` because that is the name expected by Daytona and Harbor.
    """
    env_key = os.environ.get(DAYTONA_RL_SECRET_NAME)
    if env_key:
        print(
            f"[rl-iris] Daytona: using {DAYTONA_RL_SECRET_NAME} from --secrets-env.",
            flush=True,
        )
        return env_key
    value = _daytona_rl_api_key_from_secret_manager()
    if value:
        print(
            "[rl-iris] Daytona: agentic run uses canonical Google Secret Manager "
            f"{DAYTONA_RL_SECRET_NAME} version {DAYTONA_RL_SECRET_VERSION}.",
            flush=True,
        )
        return value
    raise SystemExit(
        "[rl-iris] no Daytona RL key available: provide DAYTONA_RL_API_KEY via --secrets-env "
        "or authenticate gcloud for the Marin project, then retry."
    )


def _daytona_client(api_key: str) -> Any:
    """Construct a Daytona SDK client for the RL org.

    The daytona SDK is an optional launch-host dependency, so it is imported lazily here
    rather than at module scope.
    """
    from daytona import Daytona, DaytonaConfig

    return Daytona(DaytonaConfig(api_key=api_key))


def _purge_stale_daytona_snapshots(api_key: str) -> None:
    """Delete stale harbor-minted Daytona snapshots on the RL org before a launch.

    A snapshot is a purge candidate iff its name starts with ``HARBOR_SNAPSHOT_NAME_PREFIX``
    and it has been idle (no use since ``last_used_at``, or since ``created_at`` if never
    used) for longer than ``STALE_SNAPSHOT_MAX_AGE``. Non-harbor (system) snapshots are
    never touched. See ``DAYTONA_RL_SNAPSHOT_QUOTA`` above for why this runs before every
    launch.
    """
    try:
        d = _daytona_client(api_key)
    except ImportError:
        # The daytona SDK is Linux-only (sys_platform == 'linux' in pyproject.toml).
        # On a macOS launch host it is absent; skip the purge rather than crashing.
        print(
            "[rl-iris] Daytona snapshot purge skipped: daytona SDK is not installed "
            "on this platform (expected on macOS launch hosts).",
            flush=True,
        )
        return
    now = datetime.datetime.now(datetime.timezone.utc)
    total = 0
    harbor_count = 0
    purged_count = 0
    page = 1
    while True:
        result = d.snapshot.list(page=page, limit=100)
        total = result.total
        for snapshot in result.items:
            if not snapshot.name.startswith(HARBOR_SNAPSHOT_NAME_PREFIX):
                continue
            harbor_count += 1
            last_active = snapshot.last_used_at or snapshot.created_at
            if now - last_active > STALE_SNAPSHOT_MAX_AGE:
                d.snapshot.delete(snapshot)
                purged_count += 1
        if page >= result.total_pages or not result.items:
            break
        page += 1
    kept_count = harbor_count - purged_count
    print(
        f"[rl-iris] Daytona snapshot purge: total={total} harbor={harbor_count} "
        f"purged={purged_count} kept={kept_count}",
        flush=True,
    )


def _effective_gdn_backend(args: SimpleNamespace) -> str:
    """Return the GDN backend declared by the resolved launch config."""
    raw_config = _load_rl_config_yaml(args.rl_config)
    return str((raw_config.get("generator") or {}).get("gdn_backend", "torch")).lower()


def _sanitize_job_name_component(value: str) -> str:
    """Make one human-readable Kubernetes job-name component."""
    value = value.strip().rstrip("/").split("/")[-1]
    value = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return value or "run"


def _cluster_dashboard_host(cluster_config_path: Optional[str]) -> Optional[str]:
    """Bare host of a cluster config's ``dashboard_url`` — the public host of the
    controller that OWNS endpoints registered on that cluster. None if unreadable."""
    if not cluster_config_path:
        return None
    try:
        import yaml
        from urllib.parse import urlparse

        with open(cluster_config_path) as f:
            raw = yaml.safe_load(f) or {}
        url = raw.get("dashboard_url")
        return urlparse(url).hostname if url else None
    except Exception:  # noqa: BLE001
        return None


def _rl_config_is_agentic(rl_config: Optional[str]) -> bool:
    """True when the rl_config drives an in-sandbox agent (opencode/harbor/terminal_bench)
    that must call BACK to the served model. Best-effort text scan."""
    try:
        if not rl_config or not os.path.isfile(rl_config):
            return False
        with open(rl_config, "r") as f:
            text = f.read().lower()
        return any(k in text for k in ("terminal_bench", "harbor", "opencode"))
    except OSError:
        return False


def validate_controller_ingress_reachability(args: SimpleNamespace) -> None:
    """Fail loud BEFORE submit when ``--ingress-mode controller`` would produce a
    capability URL a Daytona sandbox CANNOT reach — the Exp2 opencode-RL blocker
    (ported from commit 8fdabb12, extended for the federated remediation).

    opencode runs in a Daytona sandbox and reaches the co-located vLLM over the public
    internet at ``https://<ingress_host>/proxy/t/<token>/<endpoint>/v1``. The endpoint
    is REGISTERED on the controller of the cluster the job runs on and the token is
    minted with that controller's key, so the capability URL only resolves when
    ``<ingress_host>`` is a controller that can BOTH route to the endpoint AND be
    reached from Daytona:

      * A **directly-submitted CoreWeave** job cannot: the peer controller's own host
        (``dashboard_url``, e.g. ``iris-cw-us-east-02a.oa.dev``) is IP-locked to the
        marin egress; and iris.oa.dev (marin) only FEDERATES ``/proxy`` to a CoreWeave
        endpoint for a job it DELEGATED. A direct submit → iris.oa.dev has no route →
        404 → opencode never reaches vLLM → RecordProxy captures 0 traffic, the job
        burns an H100 node making 0 trials.
      * The **federated** path (``--target-cluster <peer>``) fixes it: marin delegates
        the job to the peer child, so ``has_received_job_from_peer`` passes and marin
        federation-proxies ``/proxy``. The endpoint is registered on the peer AND
        MIRRORED onto marin by FederationSync; the capability token is minted at the
        PARENT (iris.oa.dev) for the mirrored endpoint. So controller-ingress on
        CoreWeave is ALLOWED iff ``--target-cluster`` is set and ``--ingress-host`` is
        the marin host.

    Escape hatch (once a further remediation is wired): ``OTAGENT_ALLOW_INGRESS_HOST_MISMATCH=1``.
    """
    if getattr(args, "ingress_mode", "direct") != "controller":
        return
    if os.environ.get("OTAGENT_ALLOW_INGRESS_HOST_MISMATCH") == "1":
        print(
            "[rl-iris] WARNING: OTAGENT_ALLOW_INGRESS_HOST_MISMATCH=1 — skipping the "
            "controller-ingress reachability guard.",
            flush=True,
        )
        return

    cluster = str(getattr(args, "cluster", "") or "")
    ingress_host = str(getattr(args, "ingress_host", "") or "")
    target_cluster = str(getattr(args, "target_cluster", "") or "")
    dash_host = _cluster_dashboard_host(getattr(args, "cluster_config", None))
    is_coreweave = cluster.startswith("cw-") or (dash_host or "") not in ("", "iris.oa.dev")

    if is_coreweave:
        # The ONLY reachable CoreWeave topology: federated submission through marin.
        if not target_cluster:
            raise SystemExit(
                "[rl-iris] BLOCKED: --ingress-mode controller on a directly-submitted "
                f"CoreWeave job (--cluster={cluster or '?'}, controller host="
                f"{dash_host or '?'}) is NOT reachable from a Daytona sandbox.\n"
                "  The capability URL would 404: iris.oa.dev only federates /proxy for a "
                "job it DELEGATED, and the CoreWeave controller's own host is IP-locked. "
                "opencode would never reach vLLM (0 trials, RecordProxy captures nothing) "
                "— the 2026-07-16 Exp2 blocker.\n"
                "  Fix: pass --target-cluster " + (cluster or "<peer>") + " to federate "
                "the job through the marin meta-scheduler (keep --ingress-host iris.oa.dev), "
                "so marin delegates it to the peer and federation-proxies /proxy.\n"
                "  Override (only once another remediation is wired): "
                "OTAGENT_ALLOW_INGRESS_HOST_MISMATCH=1."
            )
        if ingress_host and ingress_host != "iris.oa.dev":
            raise SystemExit(
                f"[rl-iris] BLOCKED: federated CoreWeave controller-ingress needs "
                f"--ingress-host iris.oa.dev (the marin parent that owns the mirrored "
                f"endpoint + signs the token), got --ingress-host {ingress_host}. A "
                "peer-signed token 401s at iris.oa.dev (federation trust is "
                "unidirectional: cw trusts marin, not the reverse)."
            )
        return
    # Non-CoreWeave (e.g. a marin-local submission): the host must match the controller
    # that owns the endpoint.
    if ingress_host and dash_host and ingress_host != dash_host and not target_cluster:
        raise SystemExit(
            f"[rl-iris] BLOCKED: --ingress-host {ingress_host} does not match this "
            f"cluster's controller host {dash_host} (--cluster={cluster}). Override with "
            "OTAGENT_ALLOW_INGRESS_HOST_MISMATCH=1."
        )


@dataclass(frozen=True)
class FederatedParentCredentials:
    """Credential material validated on the launcher and forwarded to the peer task."""

    login_record_json: str | None = None
    iap_token: str | None = None


def prepare_federated_parent_credentials(args: SimpleNamespace) -> FederatedParentCredentials | None:
    """Validate and return credentials needed by a federated pod.

    Prefer the cached Marin login record so the pod can refresh credentials. When the
    launcher has ambient service-account credentials instead, forward its short-lived
    IAP token. Both paths mint locally before allocating GPUs.
    """
    if not getattr(args, "target_cluster", None) or getattr(args, "ingress_mode", "direct") != "controller":
        return None
    if not MARIN_LOGIN_RECORD_PATH.is_file():
        parent_config = getattr(args, "parent_cluster_config", None) or _resolve_parent_cluster_config(
            getattr(args, "cluster_config", None)
        )
        try:
            from iris.cli.connect import client_credentials
            from iris.cluster.config import load_config

            if parent_config is None:
                raise RuntimeError("no parent cluster config")
            credentials = client_credentials(load_config(parent_config), "marin")
            provider = credentials.iap_provider
            token = provider.get_token() if provider is not None else None
        except Exception as exc:
            raise SystemExit(
                "[rl-iris] BLOCKED: federated CoreWeave controller ingress requires either "
                f"the cached Marin IAP login record at {MARIN_LOGIN_RECORD_PATH} or ambient "
                "service-account credentials that can mint an IAP token. Run "
                "`iris --cluster=marin login`, or configure workload identity, then relaunch."
            ) from exc
        if not token:
            raise SystemExit("[rl-iris] BLOCKED: ambient service-account credentials returned an empty IAP token.")
        print(
            "[rl-iris] Federated parent-IAP preflight passed with ambient service-account credentials; "
            "forwarding a short-lived IAP token to the peer task.",
            flush=True,
        )
        return FederatedParentCredentials(iap_token=token)
    try:
        record = json.loads(MARIN_LOGIN_RECORD_PATH.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"[rl-iris] BLOCKED: {MARIN_LOGIN_RECORD_PATH} is not valid JSON. "
            "Run `iris --cluster=marin login` and relaunch."
        ) from exc

    if record.get("cluster") != "marin" or urlparse(str(record.get("endpoint", ""))).hostname != "iris.oa.dev":
        raise SystemExit(
            f"[rl-iris] BLOCKED: {MARIN_LOGIN_RECORD_PATH} is not a Marin iris.oa.dev login record. "
            "Run `iris --cluster=marin login` and relaunch."
        )
    refresh_token = record.get("edge_refresh_token")
    if not isinstance(refresh_token, str) or not refresh_token:
        raise SystemExit(
            f"[rl-iris] BLOCKED: {MARIN_LOGIN_RECORD_PATH} has no edge_refresh_token. "
            "Run `iris --cluster=marin login` and relaunch."
        )

    from rigging.auth import IapRefreshTokenProvider, MARIN_DESKTOP_OAUTH_CLIENT

    provider = IapRefreshTokenProvider(
        MARIN_DESKTOP_OAUTH_CLIENT.client_id,
        MARIN_DESKTOP_OAUTH_CLIENT.client_secret,
        refresh_token,
        login_hint="log in to cluster 'marin' to authenticate",
    )
    try:
        token = provider.get_token()
    except Exception as exc:
        raise SystemExit(
            "[rl-iris] BLOCKED: unable to mint an IAP token from the cached Marin login record. "
            "Run `iris --cluster=marin login` and relaunch."
        ) from exc
    if not token:
        raise SystemExit(
            "[rl-iris] BLOCKED: cached Marin login did not mint an IAP token. "
            "Run `iris --cluster=marin login` and relaunch."
        )
    print(
        "[rl-iris] Federated parent-IAP preflight passed; forwarding the cached Marin login record to the peer task.",
        flush=True,
    )
    return FederatedParentCredentials(login_record_json=json.dumps(record))


def build_debug_launch_env(args: SimpleNamespace) -> dict[str, str]:
    """Resolve the effective debug preset after the job name and RL config exist."""
    raw = _load_rl_config_yaml(args.rl_config)
    trainer = raw.get("trainer") or {}
    mode = str(trainer.get("debug_mode", DebugMode.LIGHT.value))
    try:
        resolved = DebugMode(mode)
    except ValueError as error:
        choices = ", ".join(item.value for item in DebugMode)
        raise ValueError(f"trainer.debug_mode must be one of: {choices}; got {mode!r}") from error
    if resolved is DebugMode.OFF:
        return {}
    phase_diagnostics = trainer.get("collective_phase_diagnostics") is not False
    return EnvVarManager.for_debug_launch(
        mode=resolved,
        job_name=args.job_name,
        collective_phase_diagnostics=phase_diagnostics,
    ).environment_for(EnvVarScope.TASK_RUNTIME)


def _load_yaml_mapping(config_path: str) -> dict[str, Any]:
    full = PROJECT_ROOT / config_path
    path = full if full.exists() else Path(config_path)
    with path.open() as source:
        raw = yaml.safe_load(source) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: launch config must contain a mapping")
    return raw


def _load_rl_config_yaml(config_path: str) -> dict[str, Any]:
    """Return the SkyRL subtree from a validated launch document."""
    raw = _load_yaml_mapping(config_path)
    skyrl = raw.get("skyrl")
    if not isinstance(skyrl, dict):
        raise ValueError(f"{config_path}: launch config must contain a skyrl mapping")
    return skyrl


def load_config_extra_env(rl_config_path: str) -> dict[str, str]:
    """Return normalized task environment variables declared by an RL config.

    Top-level ``extra_env`` values take precedence over a ported
    ``container.extra_env`` mapping. Unreadable or empty configurations return no
    overrides.
    """
    try:
        launch = _load_yaml_mapping(rl_config_path)
    except Exception as exc:  # noqa: BLE001
        print(f"[rl-iris] WARNING: could not read extra_env from {rl_config_path}: {exc}", file=sys.stderr)
        return {}
    runtime = launch.get("runtime") or {}
    extra = dict(runtime.get("task_env") or {})
    skyrl_controls = sorted(str(key) for key in extra if str(key).startswith("SKYRL_"))
    if skyrl_controls:
        raise ValueError(
            "RL config extra_env may not define MarinSkyRL behavior; use typed Hydra settings instead: "
            + ", ".join(skyrl_controls)
        )
    out: dict[str, str] = {}
    for k, v in extra.items():
        if v is None:
            continue
        if isinstance(v, bool):
            v = int(v)
        out[str(k)] = str(v)
    return out


def _job_scope_fr_dump_path(prefix: str, job_name: str) -> str:
    """Rewrite a JOB-SCOPED NCCL flight-recorder dump path so its slug segment is the
    ACTUAL job name, e.g. ``/tmp/fr_dumps/<slug>/nccl_fr_rank`` -> ``/tmp/fr_dumps/
    <job_name>/nccl_fr_rank``.

    WHY (2026-07-11 FR-slug bug): the 80B configs hardcode
    ``TORCH_NCCL_DEBUG_INFO_TEMP_FILE: /tmp/fr_dumps/80b-next-cp1/nccl_fr_rank`` in
    their ``extra_env:``, so a run launched under a DIFFERENT ``--job-name`` (e.g.
    ``80b-next-cp1-r3d2``) still wrote its FR dumps under the stale ``80b-next-cp1``
    slug (harmless there, but wrong — a future FR dump would land under the wrong
    slug). Deriving the slug from the live job name keeps the dump under the right
    per-job dir. The controller's ``ensure_fr_dump_dir`` mkdir -p's whatever dirname
    the cvar carries, so overriding the cvar here is sufficient.

    ONLY rewrites the job-scoped ``.../fr_dumps/<slug>/<file>`` pattern; a bare
    generic path (``/tmp/nccl_fr_rank``, which every non-80B iris config uses) has no
    slug segment and is returned UNCHANGED (byte-identical for those configs)."""
    parent = os.path.dirname(prefix)  # e.g. /tmp/fr_dumps/<slug>
    grandparent = os.path.dirname(parent)  # e.g. /tmp/fr_dumps
    if os.path.basename(grandparent) != "fr_dumps":
        return prefix  # not a job-scoped fr_dumps path; leave it
    return os.path.join(grandparent, job_name, os.path.basename(prefix))


def _build_task_shell(
    args: SimpleNamespace,
    controller_cmd: list[str],
    pythonpath: str,
) -> list[str]:
    """Wrap the node controller with per-pod storage and cache setup."""
    ctrl = shlex.join(controller_cmd)
    # Run outside task_runtime so every pod establishes Ray's filesystem contract
    # before importing or starting any MarinSkyRL controller code. The controller
    # repeats this check immediately before `ray start` as defense in depth.
    spill_preflight = resolve_ray_spill_target(
        args.rendezvous_dir, args.ray_spill_backend, args.ray_spill_dir
    ).shell_preflight()
    # TileLang JIT-cache warm-start shim for GDN/FlashQLA runs only.
    # generator.gdn_backend=flashqla lazily JIT-compiles the FlashQLA GatedDeltaNet TileLang
    # kernels on the first GPU forward into the node-local, ephemeral TileLang cache.
    # This brackets the train command with a per-pod, per-NODE cache sync (the bash
    # runs once per task pod / node, and
    # TileLang's cache is node-local, so one --down warms all 8 local GPU workers):
    #   --down BEFORE the controller -> pulls the keyed warm cache (seed cache.tgz +
    #          incremental per-hash-dir objects) into TILELANG_CACHE_DIR so TileLang
    #          hash-matches and skips the cold compile. A miss is a warn+continue no-op.
    #   --up   at EXIT (bash EXIT trap; fires on normal completion AND a `set -e`/crash
    #          exit) -> uploads NEWLY-compiled hash-dirs as per-hash objects (race-free
    #          across the ~16 writers — content-addressed, no cache.tgz overwrite).
    # The shim never fails the job (best-effort; exits 0 even on S3 error).
    # TILELANG_CACHE_DIR is exported (defaulting to TileLang's own default) so the shim
    # and the trainer's TileLang agree on the location; a config-set value wins.
    # TILELANG_CACHE_MODEL_PATH lets the shim derive the model component of the key.
    sync_py = "cloud/iris/tilelang_cache_sync.py"
    tl_down = f"{RL_PYTHON} {sync_py} --down || true"
    tl_up = f"{RL_PYTHON} {sync_py} --up || true"
    # The controller is run as a BACKGROUND child + `wait` (not `exec`) so we can
    # (a) run --up at exit via the bash EXIT trap and (b) FORWARD SIGTERM/SIGINT to
    # the controller — preserving the old `exec` graceful-shutdown path (rank-0's Ray
    # teardown + nonzero exit on preemption) that a plain child would lose. `wait` is
    # interrupted by the trapped signal (rc>128); we re-`wait` to reap the child's
    # real exit code after its forwarded-TERM shutdown.
    flashqla_enabled = _effective_gdn_backend(args) == "flashqla"
    gdn_branch = (
        f'export TILELANG_CACHE_DIR="${{TILELANG_CACHE_DIR:-/root/.tilelang/cache}}"; '
        f"export TILELANG_CACHE_MODEL_PATH={shlex.quote(args.model_path)}; "
        f"{tl_down}; "
        f"trap {shlex.quote(tl_up)} EXIT; "
        f'trap \'[ -n "$_child" ] && kill -TERM "$_child" 2>/dev/null\' TERM INT; '
        f"set +e; {ctrl} & _child=$!; "
        f'wait "$_child"; _rc=$?; '
        f'if [ $_rc -gt 128 ]; then wait "$_child" 2>/dev/null; _rc=$?; fi; '
        f"exit $_rc"
    )
    if not flashqla_enabled:
        gdn_branch = f"exec {ctrl}"
    bash = (
        f"set -e; {spill_preflight}cd {APP_DIR}; "
        f"export SKYRL_HOME={shlex.quote(SKYRL_HOME)}; "
        f"source {shlex.quote(MARINSKYRL_ACTIVATION_FILE)}; "
        f"export PYTHONPATH={shlex.quote(pythonpath)}:${{PYTHONPATH:-}}; "
        f"export VLLM_USE_V1=1; "
        f"{gdn_branch}"
    )
    return ["bash", "-c", bash]


def build_config_task_command(args: SimpleNamespace) -> list[str]:
    """Build the replica-controller command for the resolved launch config."""
    launch_config = args.rl_config_launch
    if not isinstance(launch_config, BundledLaunchConfig):
        raise RuntimeError("resolved launch config payload is missing")
    controller_cmd = [
        RL_PYTHON,
        "cloud/iris/task_runtime.py",
        "--config",
        launch_config.task_path,
    ]
    pythonpath = f"{APP_DIR}:{SKYRL_HOME}:{SKYRL_HOME}/skyrl-train"
    return _build_task_shell(args, controller_cmd, pythonpath)


def launch(args: SimpleNamespace, expected_launcher_commit: str) -> IrisLaunchOutcome:
    """Submit a resolved launch config and, unless detached, wait for its terminal state."""
    workspace = build_runtime_bundle(expected_launcher_commit)
    parent_credentials = prepare_federated_parent_credentials(args)

    if not args.job_name:
        args.job_name = f"rl-iris-{time.strftime('%Y%m%d-%H%M%S')}"

    # Load --secrets-env into os.environ on the launch host (so launch-host
    # hooks see it) AND collect them for injection into the task. Reuse the
    # (file overrides shell; same semantics as the iris launchers).
    load_secrets_env_into_os_environ(args.secrets_env)

    if not _is_checkpoint_export(args) and _rl_config_is_agentic(args.rl_config):
        daytona_api_key = _resolve_daytona_rl_api_key()
        os.environ["DAYTONA_API_KEY"] = daytona_api_key
        # The purge deletes stale snapshots across the shared RL org, so skip it on a
        # --dry-run — with the --secrets-env fallback above, a dry-run now reaches this
        # point instead of exiting at the Secret Manager call.
        if not args.dry_run:
            _purge_stale_daytona_snapshots(daytona_api_key)

    command = build_config_task_command(args)

    # Per-task resources: one whole selected GPU node per task.
    gpu_spec = f"{args.gpu_variant}x{args.gpus_per_node}"

    automatic_memory = _resource_request_is_automatic(str(args.memory))
    automatic_disk = _resource_request_is_automatic(str(args.disk))
    if automatic_memory or automatic_disk:
        resolved_resources = resolve_node_resource_requests(
            args.cluster_config,
            gpu_variant=args.gpu_variant,
            gpus_per_node=args.gpus_per_node,
            num_nodes=args.num_nodes,
            memory_request=str(args.memory),
            disk_request=str(args.disk),
        )
        args.memory = resolved_resources.memory
        args.disk = resolved_resources.disk

    user = os.environ.get("USER") or os.environ.get("USERNAME") or "user"
    print(f"[rl-iris] Job:        /{user}/{args.job_name}", flush=True)
    print(f"[rl-iris] Cluster:    {args.cluster}  ({args.cluster_config})", flush=True)
    print(f"[rl-iris] Runtime:    {args.runtime_commit} ({args.runtime_profile.value})", flush=True)
    print(
        f"[rl-iris] Topology:   {args.num_nodes} node(s) x {gpu_spec}  "
        f"(= {args.num_nodes * args.gpus_per_node} GPUs, exclusive, gang/leafgroup)",
        flush=True,
    )
    print(f"[rl-iris] Per node:   cpu={args.cpu} memory={args.memory} disk={args.disk}", flush=True)
    print(f"[rl-iris] Priority:   {args.priority}", flush=True)
    print(f"[rl-iris] RL config:  {args.rl_config}  model={args.model_path}", flush=True)
    storage_paths = args.storage_paths
    if not _is_checkpoint_export(args):
        print(
            f"[rl-iris] Resume:     {storage_paths.checkpoint_root} (keep {storage_paths.resume_checkpoint_count})",
            flush=True,
        )
        print(f"[rl-iris] Canonical:  {storage_paths.export_root}", flush=True)
        print(f"[rl-iris] Raw traces: {storage_paths.trace_root}", flush=True)
        print(f"[rl-iris] Trajectory: {storage_paths.trajectory_root}", flush=True)
        print(f"[rl-iris] Ray logs:   {storage_paths.ray_log_root}", flush=True)
        print(f"[rl-iris] Ray spill:  {args.ray_spill_backend.value}:{args.ray_spill_dir}", flush=True)
    print(f"[rl-iris] Rendezvous: {args.rendezvous_dir}", flush=True)
    print(f"[rl-iris] Command:    {shlex.join(command)}", flush=True)

    if args.dry_run:
        print("[rl-iris] --dry-run: not submitting", flush=True)
        return IrisLaunchOutcome(job_id="", job_state="prepared", exit_code=0)

    # Defer heavy iris imports so --dry-run / --help stay snappy.
    #
    # NOTE: post iris PR #6652 (pydantic config parsing) + #6730 (multi-backend
    # controller) the old submit API moved. The config is now a pydantic
    # ``IrisClusterConfig`` loaded via the MODULE-LEVEL ``load_config(path)``
    # (the ``IrisConfig`` class + its ``.load()`` / ``.provider_bundle()`` /
    # ``.proto`` are gone). The provider bundle is now built by the module-level
    # ``iris.cluster.composer.provider_bundle(config)``, and ``LocalCluster``
    # moved to ``iris.cluster.local_cluster``. The job-build helpers
    # (ResourceSpec / constraints / EnvironmentSpec / Entrypoint / job_pb2) and
    # the ``IrisClient.remote(...)`` /
    # ``client.submit(...)`` surface are UNCHANGED — see how the marin CLI itself
    # now submits in iris/cli/job.py + iris/cli/connect.py, which this mirrors.
    from iris.cluster.types import EnvironmentSpec, Entrypoint

    # Per-task resources: whole node, all GPUs (no co-tenant → exclusive).
    resources = _gpu_resources(
        args.gpu_variant,
        args.gpus_per_node,
        cpu=args.cpu,
        memory=args.memory,
        disk=args.disk,
    )

    # Multi-node gang: replicas=num_nodes; for GPUs with replicas>1 this returns
    # CoschedulingConfig(group_by="leafgroup") — co-schedule all nodes on one IB
    # leaf fabric, atomically (Kueue gang admission on cw-us-east-02a).
    replicas = args.num_nodes
    coscheduling = _gpu_multinode(args.gpu_variant, args.gpus_per_node, replicas)

    resources_proto = resources.to_proto()
    # --target-cluster (federated submission) appends a `cluster EQ <peer>` constraint
    # so the marin meta-scheduler DELEGATES the whole job to the peer child (see the
    # submission block below). None on the default direct path (byte-identical).
    constraints = _gpu_constraints(
        resources_proto,
        replicas=replicas,
        preemptible=args.preemptible,
        target_cluster=args.target_cluster,
    )

    priority_band = job_pb2.PriorityBand.Value(f"PRIORITY_BAND_{args.priority.upper()}")

    # Env: secrets file values + the standard RL/iris-serve signals. iris injects
    # IRIS_TASK_ID / IRIS_NUM_TASKS / IRIS_ADVERTISE_HOST per task automatically.
    env_vars: dict[str, str] = {}
    # Forward the RL config YAML's top-level `extra_env:` block (the Iris analog of
    # the SLURM container.extra_env exports — see load_config_extra_env). The
    # launcher's own signals (rendezvous/secrets, below) win on any collision.
    config_extra_env = load_config_extra_env(args.rl_config)
    if config_extra_env:
        env_vars.update(config_extra_env)
        print(f"[rl-iris] Config extra_env: {', '.join(sorted(config_extra_env))}", flush=True)
    debug_env = build_debug_launch_env(args)
    if debug_env:
        env_vars.update(debug_env)
        print(
            f"[rl-iris] Debug mode {debug_env[DEBUG_MODE_ENV]}: artifacts -> {debug_env[DEBUG_ARTIFACT_DIR_ENV]}",
            flush=True,
        )
    env_vars.update(args.rl_config_launch.task_environment())
    # ── Per-cluster infra-env DEFAULTS (fill-gap belt for cluster-specific footguns) ──────────
    # Some clusters need a specific network/NCCL interface that a cluster-AGNOSTIC RL config
    # won't (and shouldn't) carry. Fill it in here, keyed on --target-cluster, only if the
    # config's extra_env did not set it (lowest precedence — an explicit
    # value always wins). cw-rno2a: its host_network:true nodes expose IB/IPoIB (ibs*/ibp*) +
    # virtual ifaces, so NCCL AND Ray's raylet/GCS mis-detect the bootstrap interface and the
    # multi-node gang SILENTLY never forms (keep-6 2026-07-19: 6 arms idle-heartbeated 3.4h at
    # zero progress before this was diagnosed). Pin the bootstrap socket to the host ethernet PF
    # via the exclude pattern (value = cw-rno2a.yaml:128). No-op on cw-us-east-02a (auto-detect
    # already lands on the PF). Add a cluster row here rather than editing every RL config.
    _target_cluster = str(getattr(args, "target_cluster", "") or "")
    # Do NOT add GLOO_SOCKET_IFNAME here with this value. Gloo does not accept
    # NCCL's `^exclude` syntax — it wants a literal interface name and fails with
    # `Unable to find address for: ^ibs` at engine init. Gloo picking loopback on
    # multi-node gangs is still an open problem; the fix needs a real interface
    # name, which differs per node, so the exclusion trick does not transfer.
    _CLUSTER_ENV_DEFAULTS: dict[str, dict[str, str]] = {
        "cw-rno2a": {"NCCL_SOCKET_IFNAME": "^ibs,ibp,lo,docker,veth,cilium,lxc"},
    }
    for _k, _v in _CLUSTER_ENV_DEFAULTS.get(_target_cluster, {}).items():
        if _k not in env_vars:
            env_vars[_k] = _v
            print(f"[rl-iris] Cluster infra-env default for {_target_cluster}: {_k}={_v}", flush=True)
    # FR-slug fix: a config may hardcode a JOB-SCOPED NCCL flight-recorder dump path
    # (/tmp/fr_dumps/<slug>/nccl_fr_rank) with a STALE slug from the config it was
    # copied from. Re-scope the slug to the live --job-name so a future FR dump lands
    # under the right per-job dir (the controller mkdir -p's the cvar's dirname). No-op
    # for the bare generic /tmp/nccl_fr_rank path every non-80B config uses.
    for _fr_cvar in ("TORCH_NCCL_DEBUG_INFO_TEMP_FILE", "TORCH_FR_DUMP_TEMP_FILE"):
        _old = env_vars.get(_fr_cvar)
        if _old:
            _new = _job_scope_fr_dump_path(_old, args.job_name)
            if _new != _old:
                env_vars[_fr_cvar] = _new
                print(f"[rl-iris] FR-slug re-scope: {_fr_cvar} {_old} -> {_new}", flush=True)
    if args.rendezvous_dir:
        env_vars["OT_AGENT_IRIS_RENDEZVOUS_DIR"] = args.rendezvous_dir
    env_vars["OT_AGENT_IRIS_RAY_PORT"] = str(args.ray_port)
    # Forward the launch host's secrets (mirrors launch_eval_iris.py passthrough).
    #
    # IMPORTANT — do NOT forward AWS_*/R2_* here. The cw-us-east-02a cluster
    # projects an `iris-task-env` k8s Secret into EVERY task pod via `envFrom`
    # (because storage.remote_state_dir is an s3:// URI), and that secret already
    # carries the correct in-cluster R2 credentials + endpoint
    # (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_ENDPOINT_URL / AWS_REGION /
    # FSSPEC_S3). In K8s, explicit container `env` entries take precedence over
    # `envFrom`, so forwarding the launch host's AWS_* (which point at a
    # DIFFERENT account and lack AWS_ENDPOINT_URL) would CLOBBER the pod's
    # injected creds and make the s3://marin-us-east-02a rendezvous (multi-node)
    # silently target real AWS S3 instead of the cluster store. NOTE: the default
    # object store moved R2 (s3://marin-na) -> CW (s3://marin-us-east-02a) on
    # 2026-07-05 (marin c7caecc95a) — pods now inject CW creds+AWS_ENDPOINT_URL and
    # can no longer reach R2. Let the cluster-injected creds win; the
    # fsspec rendezvous in task_runtime.py uses default credential
    # discovery and picks them up.
    #
    # Daytona credentials MUST be forwarded: agentic RL (terminal_bench / Harbor)
    # builds a Daytona sandbox per trial, and iris injects only HF/WANDB into the
    # task pod — nothing else. Without DAYTONA_API_KEY the worker's harbor client
    # raises DaytonaAuthenticationError on every env build, so no sandbox comes
    # up, the verifier never runs, and EVERY trajectory finalizes as
    # VerificationNotCompletedError with reward 0 (observed zeroing an entire
    # reverify rollout). Mirror the base IrisLauncher passthrough set
    # so the same creds reach the RL worker.
    #
    # SkyRL passes trainer.project_name directly to wandb.init but does not pass an
    # entity. Resolve that missing dimension here: the explicit launch argument wins,
    # followed by the launch-host value and the Marin team default.
    for k in (
        "HF_TOKEN",
        "WANDB_API_KEY",
        "DAYTONA_API_KEY",
        "DAYTONA_JWT_TOKEN",
        "DAYTONA_ORGANIZATION_ID",
        "DAYTONA_API_URL",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "TOGETHER_API_KEY",
    ):
        v = os.environ.get(k)
        if v:
            env_vars[k] = v
    env_vars.update(wandb_launch_environment(entity=args.wandb_entity))

    # Federated controller-ingress pod plumbing (opencode-RL literal capture): the in-pod
    # worker mints the capability token at the PARENT (marin/iris.oa.dev) for the mirrored
    # endpoint, which needs (a) the parent cluster config path and (b) IAP credentials to
    # authenticate to iris.oa.dev. We forward the config path + any launch-host IAP cred
    # env so the in-pod _ParentControllerClient can re-mint the IAP OIDC token.
    #
    # The parent config file is not part of the task bundle, so forward its contents for
    # in-pod materialization. The credential preflight returns either a refreshable Marin
    # login record or a short-lived token minted from ambient service-account credentials.
    # Direct submission (no --target-cluster) forwards none of this.
    if getattr(args, "target_cluster", None) and getattr(args, "ingress_mode", "direct") == "controller":
        parent_cfg = (
            getattr(args, "parent_controller_config_in_pod", None)
            or args.parent_cluster_config
            or _resolve_parent_cluster_config(args.cluster_config)
        )
        if parent_cfg:
            env_vars[PARENT_CONTROLLER_CONFIG_ENV] = parent_cfg
            # marin.yaml is not part of the synced workspace, so the path above won't
            # resolve in-pod. Forward the
            # file CONTENT (write-from-env, mirroring the cached login record) so the
            # in-pod worker (materialize_parent_controller_config) writes it to a real
            # path and repoints the env. marin.yaml carries no secrets (signing_key is
            # a gcp-secret:// ref resolved server-side). When parent_cfg is an explicit
            # in-pod path (baked/synced), os.path.isfile is False on the launch host →
            # no content forwarded (operator owns materialization).
            if os.path.isfile(parent_cfg):
                with open(parent_cfg) as _pf:
                    env_vars[PARENT_CONTROLLER_CONFIG_YAML_ENV] = _pf.read()
        for k in (
            "IRIS_IAP_REFRESH_TOKEN",
            "GOOGLE_APPLICATION_CREDENTIALS",
            "IRIS_EDGE_REFRESH_TOKEN",
        ):
            v = os.environ.get(k)
            if v:
                env_vars[k] = v
        if parent_credentials is None:
            raise AssertionError("federated controller ingress must have validated parent credentials")
        if parent_credentials.login_record_json is not None:
            env_vars[PARENT_CREDENTIALS_JSON_ENV] = parent_credentials.login_record_json
        if parent_credentials.iap_token is not None:
            env_vars[PARENT_IAP_TOKEN_ENV] = parent_credentials.iap_token

    # Load the cluster config (pydantic IrisClusterConfig) and build the provider
    # bundle, then discover + tunnel to the controller. This mirrors the marin
    # CLI's own path (iris/cli/connect.py::require_controller_url): for a local
    # controller start an in-process LocalCluster; otherwise use the config's
    # controller_address() (defaults.worker.controller_address) if set, else fall
    # back to the backend's discover_controller(). cw-us-east-02a's controller
    # kind is "coreweave" (non-local, no IAP auth) → the discover path.
    #
    # FEDERATED submission (--target-cluster set): submit through the PARENT (marin)
    # meta-scheduler instead of the peer's own controller. We load marin.yaml (whose
    # dashboard_url is the IAP-gated iris.oa.dev) and tunnel THERE; the `cluster EQ
    # <peer>` constraint appended above makes marin delegate the whole job to the peer
    # child. This is what lets marin later federation-proxy /proxy to the peer's
    # (mirrored) endpoint — the only Daytona-reachable CoreWeave ingress topology.
    # Reaching iris.oa.dev requires IAP creds (`iris login` with an @openathena.ai
    # account, or an allowlisted service account); tunnel()/IrisClient handle the auth.
    submit_cluster_config = args.cluster_config
    if args.target_cluster:
        parent_cfg = args.parent_cluster_config or _resolve_parent_cluster_config(args.cluster_config)
        if not parent_cfg:
            raise SystemExit(
                "[rl-iris] --target-cluster set but no parent (marin) cluster config "
                "could be resolved. Pass --parent-cluster-config <path to marin.yaml>."
            )
        submit_cluster_config = parent_cfg
        print(
            f"[rl-iris] Federated submission: delegating to peer '{args.target_cluster}' "
            f"via the marin meta-scheduler ({parent_cfg}).",
            flush=True,
        )
    from contextlib import contextmanager as _contextmanager

    @_contextmanager
    def _direct_client():
        if ambient_client := _ambient_in_cluster_client(workspace):
            with ambient_client as client:
                yield client
            return
        from iris.cluster.composer import provider_bundle
        from iris.cluster.config import load_config
        from iris.cluster.local_cluster import LocalCluster

        # Direct submission to --cluster's own controller. On CoreWeave the loopback SSH
        # tunnel presents as the trusted local_admin identity (no IAP login needed) —
        # byte-identical to before.
        iris_config = load_config(submit_cluster_config)
        bundle = provider_bundle(iris_config)
        if iris_config.controller.controller_kind() == "local":
            controller_address = LocalCluster(iris_config).start()
        else:
            controller_address = iris_config.controller_address() or bundle.controller.discover_controller(
                iris_config.controller
            )
        with bundle.controller.tunnel(address=controller_address) as controller_url:
            yield IrisClient.remote(controller_url, workspace=workspace)

    if args.target_cluster:
        # Federated submission MUST carry the IAP *user* identity: the controller rejects
        # a loopback/local_admin tunnel identity for a federated job ("a local_admin
        # (CIDR/loopback) identity cannot submit a federated job"), because delegation
        # forwards the submitter's identity to the peer for its owner check. Connect to
        # the marin parent exactly as the `iris job run` CLI does — open_iris_client
        # threads the IAP ClientCredentials (iris JWT + IAP OIDC token) from the cached
        # `iris login`, so the submission carries the user identity rather than loopback.
        # Requires a completed `iris --cluster=marin login` (@openathena.ai).
        from iris.cli.connect import open_iris_client

        client_cm = open_iris_client(config_file=Path(submit_cluster_config), workspace=workspace)
    else:
        client_cm = _direct_client()

    with client_cm as client:
        entrypoint = Entrypoint.from_command(*command)
        job = client.submit(
            entrypoint=entrypoint,
            name=args.job_name,
            resources=resources,
            environment=EnvironmentSpec(
                env_vars=env_vars,
                extras=["gpu"],
                setup_scripts=[task_setup_script(args.runtime_commit, args.runtime_profile)],
            ),
            constraints=constraints or None,
            coscheduling=coscheduling,
            replicas=replicas,
            max_retries_failure=args.max_retries,
            max_task_failures=args.max_retries,
            priority_band=priority_band,
            timeout=None if args.timeout == 0 else _seconds_to_duration(args.timeout),
        )
        full_job_id = str(job.job_id)
        print(
            f"[rl-iris] Submitted: {full_job_id}  (replicas={replicas}, "
            f"coscheduling={getattr(coscheduling, 'group_by', None)})",
            flush=True,
        )

        if args.no_wait:
            return IrisLaunchOutcome(
                job_id=full_job_id,
                job_state="submitted",
                exit_code=0,
            )
        print(
            f"[rl-iris] Now streaming logs for {full_job_id}. This process runs until the job ends.\n"
            "[rl-iris] SIGINT and SIGTERM cancel the complete job tree; signals do not detach.\n"
            "[rl-iris] Use --no-wait to submit and return instead.",
            file=sys.stderr,
            flush=True,
        )
        outcome = supervise_iris_job(job)
        print(f"[rl-iris] Job exit: {outcome.exit_code}", flush=True)
        return outcome


def _ambient_in_cluster_client(workspace: Path) -> IrisClient | None:
    """Connect a nested launch to the controller assigned to its coordinator task."""
    controller_url = os.environ.get("IRIS_CONTROLLER_URL")
    if not controller_url:
        return None
    return IrisClient.in_cluster(controller_url, workspace=workspace)


def _seconds_to_duration(secs: int):
    from rigging.timing import Duration

    return Duration.from_seconds(secs)
