#!/usr/bin/env python3
"""Launch a MarinSkyRL RL training job on Marin's Iris GPU cluster (CoreWeave).

This is the GPU/Iris analog of ``rl/cloud/launch_rl_cloud.py`` (the SkyPilot RL
launcher). It combines:
  - the RL-job structure from ``launch_rl_cloud.py`` (gpu-rl venv, run_rl.py
    entrypoint, rl_config / model_path / train_data / overrides), and
  - the Iris SDK submission mechanics from ``eval/cloud/launch_eval_iris.py``
    (controller tunnel, IrisClient.submit, --secrets-env injection, --no-wait,
    job-name, max-retries, workspace source-sync to /app).

The target is GPU (not TPU), and the gpu-rl image is a conda-venv image
(/opt/openthoughts/envs/rl), so this launcher drives the iris SDK's GPU helpers
(build_resources(gpu=...), gpu_device, the leafgroup-coscheduling
``resolve_multinode_defaults``) directly rather than going through a TPU-shaped
base launcher.

Multi-node / gang scheduling
----------------------------
Iris HAS a native gang mechanism for GPUs (verified via `iris job run --help`
and lib/iris/src/iris/cli/job.py):
  - ``--gpu H100x8`` requests a whole CoreWeave node (8 H100 + IB) per task.
  - ``--replicas N`` (the `--help` text: "Number of tasks for gang scheduling")
    requests N such tasks.
  - For GPUs with replicas>1, ``resolve_multinode_defaults`` returns
    ``CoschedulingConfig(group_by="leafgroup")`` — the H100/InfiniBand
    colocation level — so all N replicas are co-scheduled together on the same
    IB leaf fabric, all-or-nothing.
  - The cw-us-east-02a cluster config enables **Kueue gang admission**
    (``kueue.cluster_queue: iris-cq``, ``host_network: true`` for NCCL/IB), so
    the N-task gang is admitted atomically: either all N whole nodes are
    granted or the job queues — true exclusive, co-scheduled multi-node.

So this launcher requests ``--num-nodes N`` whole H100x8 nodes EXCLUSIVELY: one
iris task per node (``replicas=N``), each holding all 8 GPUs of its node (no
co-tenants), coscheduled by leafgroup. The RL topology (one cross-node Ray
cluster, NCCL over IB) is wired by an in-container controller
(``cloud/iris/start_rl_iris_controller.py``): rank 0 starts the Ray head and
publishes its IP to a shared rendezvous; ranks 1..N-1 join; then rank 0 runs the
MarinSkyRL driver (``cloud.iris.run_rl --num_nodes N``) attached to that cluster.

Usage
-----
    set -a; source "${DC_AGENT_SECRET_ENV:?see .claude/secret.md}"; set +a

    python -m cloud.iris.launch_rl_iris \
        --rl_config cloud/iris/configs/<config>.yaml \
        --model_path Qwen/Qwen3-8B \
        --train_data '["mlfoundations-dev/dataset"]' \
        --num-nodes 4 \
        --job-name my-rl-iris-run \
        --no-wait
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import secrets
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, List, Optional
from urllib.parse import urlparse

import yaml

from cloud.iris.paths import PROJECT_ROOT
from cloud.iris.gpu_rl_images import image_for_cluster
from cloud.iris.secrets_env import load_secrets_env_into_os_environ

# Defaults for the CoreWeave H100 GPU cluster.
DEFAULT_CLUSTER = "cw-us-east-02a"
DEFAULT_GPU_VARIANT = "H100"
DEFAULT_GPUS_PER_NODE = 8  # gd-8xh100ib-i128 = 8x H100-80GB + IB
# These H100 nodes are requested WHOLE-NODE-EXCLUSIVE (no co-tenants) — so request ALL the
# node's allocatable resources; don't under-request (wasted capacity + a too-low --memory
# caused a container-cgroup OOM at FSDP weight-load on the 30B run). Node allocatable ≈ 128
# CPU / ~2014 GiB mem / 8 GPU.
#   - CPU 48 (NOT 64): ~64-68 of the 128 cores are persistent daemonset reservation, so a
#     request >~60 FAILS the single-IB-leaf gang admission (observed: 64 unplaceable, 48 admits).
#   - MEMORY 1700GB (≈1583 GiB cgroup) — RAISED from 1400GB on 2026-07-11 because 1400 was
#     UNDER-allocating. It must clear TWO opposing footguns:
#     (a) too LOW → container-cgroup OOM at the training forward. The old 512GB OOM'd at FSDP
#         weight-load; and 1400GB (≈1303 GiB cgroup) sits RIGHT AT the ~1303 GiB forward peak
#         at 8 ranks/node (the 80B cp1 128-GPU EP8×FSDP8 geometry) — the r2 rankspread run
#         measured a 1028 GiB forward peak at only 4 ranks/node (2026-07-11 diagnosis), so at
#         8 ranks/node the peak clears 1400's cgroup → no headroom = OOM risk; and
#     (b) too HIGH (1800GB ≈ 1676 GiB) → sits so close to node-allocatable (~2014 GiB) that
#         after daemonset/persistent-reservation overhead a leafgroup gang (all-or-nothing,
#         one IB leaf) can't fit all pods → Kueue SchedulingGated stall (cost multiple
#         60-120min stalls overnight 2026-06-26, on a 1-GPU probe AND 8-node gangs).
#     1700GB fits with headroom (≈1583 GiB < 2014 GiB allocatable) AND clears the forward
#     peak. Lower toward the real need on an admission stall; do NOT raise toward 1800.
#     (1000-1200GB suffices for 2-node smokes.) See .claude/ops/iris/coreweave_gpu_ops.md.
#   - DISK defaults to "auto" = 80% of the node's live allocatable ephemeral-storage (~27.2 TiB
#     → ~21 TiB). WHY NOT the old 512GB: the long MoE training step's Ray object store spills to
#     /tmp (a metered emptyDir that counts against the --disk ephemeral-storage limit), growing
#     to >1.6 TB and EVICTING the pod (2026-06-28). Whole-node-exclusive gangs have NO co-tenants,
#     so reserving disk is pure waste — claim ~80%. (R2 object-spilling, the durable fix, is also
#     on; this headroom is belt-and-suspenders.) Pass --disk explicitly to override.
# A whole H100 node exposes 128 CPUs in the Iris cluster configuration, but the
# CoreWeave scheduler leaves daemonset headroom. Requests above 48 have proven
# unplaceable, so use the cluster's advertised capacity only to reduce this
# safe ceiling for smaller future node types.
MAX_DEFAULT_CPU_PER_NODE = 48.0
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
DEFAULT_MEMORY_PER_NODE = "1700GB"
# --disk "auto" → DISK_FRACTION of the GPU node's live allocatable ephemeral-storage at launch
# (FALLBACK_DISK_GIB iff the node query fails). See _resolve_default_disk().
DEFAULT_DISK_PER_NODE = "auto"
DISK_FRACTION = 0.80
FALLBACK_DISK_GIB = 21800  # ~80% of the h100-8x ~27.2 TiB allocatable, used only if kubectl is unavailable
DEFAULT_PRIORITY = "interactive"
PRIORITY_NAMES = ("production", "interactive", "batch")


def _parse_quantity_to_gib(q: str) -> float:
    """Parse a k8s resource quantity (plain bytes, or Ki/Mi/Gi/Ti binary / k/M/G/T decimal suffix) to GiB."""
    q = q.strip()
    for suf, mult in (("Ki", 2**10), ("Mi", 2**20), ("Gi", 2**30), ("Ti", 2**40), ("Pi", 2**50)):
        if q.endswith(suf):
            return float(q[: -len(suf)]) * mult / 2**30
    for suf, mult in (("k", 1e3), ("M", 1e6), ("G", 1e9), ("T", 1e12), ("P", 1e15)):
        if q.endswith(suf):
            return float(q[: -len(suf)]) * mult / 2**30
    return float(q) / 2**30  # plain bytes


def _resolve_default_disk(fraction: float = DISK_FRACTION) -> str:
    """``fraction`` of the GPU node's LIVE allocatable ephemeral-storage, as a ``"<N>Gi"`` string.

    Whole-node-exclusive gangs have no co-tenants, so claim most of the node NVMe (the old fixed
    512GB default evicted long MoE steps once Ray's object store spilled to the metered /tmp).
    Queries kubectl for the MIN allocatable across 8-GPU nodes (never over-request a smaller node);
    falls back to FALLBACK_DISK_GIB if kubectl is unavailable (requires KUBECONFIG)."""
    try:
        out = subprocess.run(
            [
                "kubectl",
                "get",
                "nodes",
                "-o",
                r'jsonpath={range .items[*]}{.status.capacity.nvidia\.com/gpu}{" "}'
                r'{.status.allocatable.ephemeral-storage}{"\n"}{end}',
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        ).stdout
        allocs = [
            _parse_quantity_to_gib(p[1])
            for p in (line.split() for line in out.splitlines())
            if len(p) == 2 and p[0] == "8"
        ]
        if allocs:
            gib = int(min(allocs) * fraction)
            print(
                f"[rl-iris] --disk auto: {fraction:.0%} of node allocatable "
                f"(min {min(allocs):.0f}GiB across {len(allocs)} GPU nodes) = {gib}Gi",
                flush=True,
            )
            return f"{gib}Gi"
        print(
            f"[rl-iris] --disk auto: no 8-GPU nodes returned by kubectl; using fallback {FALLBACK_DISK_GIB}Gi",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001 - best-effort; fall back rather than block a launch
        print(
            f"[rl-iris] --disk auto: kubectl node query failed ({type(exc).__name__}: {exc}); "
            f"using fallback {FALLBACK_DISK_GIB}Gi",
            flush=True,
        )
    return f"{FALLBACK_DISK_GIB}Gi"


# The gpu-rl image's RL venv (deps-only: torch 2.11 + vLLM fork + skyrl editable).
RL_PYTHON = "/opt/openthoughts/envs/rl/bin/python"
SKYRL_HOME = "/opt/skyrl"
# In-container source sync target. iris syncs the launcher's `workspace`
# (this MarinSkyRL repo, PROJECT_ROOT — see IrisClient.remote(..., workspace=PROJECT_ROOT))
# to /app and sets IRIS_WORKDIR=/app. The runtime is self-contained here (cloud.iris.*) —
# no OpenThoughts-Agent workspace is required in-pod.
#
# The bundle wins for `cloud.iris.*` ONLY. It does NOT win for `skyrl_train.*`, even though
# PYTHONPATH lists /app/skyrl-train ahead of /opt/skyrl/skyrl-train: run_rl.py launches the
# trainer with `subprocess.Popen(cmd, cwd=$SKYRL_HOME/skyrl-train)`, and `python -m` puts the
# CWD first on sys.path, ahead of PYTHONPATH. So `skyrl_train.*` and `examples.terminal_bench.*`
# always resolve to the BAKED image tree.
#
# Path test: edited skyrl-train/ -> image rebuild required. Edited cloud/iris/ -> no rebuild.
# See .agents/ops/gpu-rl-image-build.md. An earlier version of this comment claimed the bundle
# also won for skyrl-train, and that cost a wasted 48-GPU launch.
APP_DIR = "/app"

# marin-iris wheel installed into the RL venv at pod bootstrap for the controller-ingress
# registration path (GAP D). The gpu-rl image bakes ONLY MarinSkyRL + harbor, never iris (a
# marin-monorepo pkg), so cloud.iris.ingress_utils' `import iris.cluster.client.* / iris.rpc.*`
# would ModuleNotFoundError in driver init. This dev build is validated against the live
# marin controller's registration/mint RPC protocol and parses the current parent marin.yaml
# `platform.gcp.registry_mirrors` field; override via --iris-ref.
DEFAULT_IRIS_VERSION = "marin-iris==0.2.54.dev202607210800"

MARIN_LOGIN_RECORD_PATH = Path.home() / ".config" / "marin" / "credentials" / "marin.json"
_JOB_NAME_MAX_LENGTH = 63


def _resolve_cluster_config_default(cluster: str = DEFAULT_CLUSTER) -> str:
    """Find the marin repo's ``<cluster>.yaml`` iris cluster config.

    ``cluster`` selects the CoreWeave cluster (e.g. ``cw-us-east-02a`` = 256×H100 default,
    ``cw-rno2a`` = the 512×H100 RNO2A cluster for the delphi pilot). Falls back to the
    bare relative path if no marin checkout is found (the caller can still pass an explicit
    ``--cluster-config``)."""
    rel = f"lib/iris/config/{cluster}.yaml"
    candidates = [
        Path.home() / "Documents/marin" / rel,
        Path("/Users/benjaminfeuer/Documents/marin") / rel,
        Path(os.environ.get("MARIN_ROOT", "")) / rel,
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return rel


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


def _cluster_storage_root(cluster_config: dict[str, Any]) -> str:
    """Return the durable object-store root containing the Iris controller state."""
    storage = cluster_config.get("storage")
    remote_state_dir = storage.get("remote_state_dir") if isinstance(storage, dict) else None
    if not isinstance(remote_state_dir, str) or not remote_state_dir.startswith(("s3://", "gs://")):
        raise SystemExit(
            "The selected Iris cluster config needs storage.remote_state_dir set to an s3:// or gs:// URI "
            "to derive --rendezvous-dir; pass --rendezvous-dir explicitly."
        )
    return remote_state_dir.rstrip("/").rsplit("/", 1)[0]


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


def _resolve_daytona_rl_api_key() -> str:
    """Read the pinned RL Daytona key from Secret Manager on the launch host.

    Iris task environments currently carry literal values, not secret references; the
    ``gcp-secret://`` resolver applies to Iris controller config only.  Resolve the
    canonical pinned secret here, retain it only in process memory, and let the existing
    Iris job-secret path inject it.  Do not include the value in diagnostics.
    """
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
    except (OSError, subprocess.SubprocessError) as exc:
        raise SystemExit(
            "[rl-iris] could not resolve the canonical DAYTONA_RL_API_KEY from Google Secret Manager; "
            "authenticate gcloud for the Marin project and retry."
        ) from exc
    value = result.stdout.strip()
    if not value:
        raise SystemExit(
            "[rl-iris] canonical DAYTONA_RL_API_KEY resolved empty from Google Secret Manager; "
            "check the pinned secret version."
        )
    return value


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
    d = _daytona_client(api_key)
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


def _validate_rl_config_topology(args: argparse.Namespace) -> None:
    """Reject a gang size incompatible with explicit trainer placement metadata.

    SkyRL placement is intentionally optional because colocated and legacy configs
    derive topology at runtime.  When both policy and reference node counts are
    declared, however, their disaggregated gang is a stable public contract.
    """
    try:
        with open(args.rl_config) as f:
            config = yaml.safe_load(f) or {}
    except OSError:
        return
    trainer = config.get("trainer") if isinstance(config, dict) else None
    placement = trainer.get("placement") if isinstance(trainer, dict) else None
    if not isinstance(placement, dict) or placement.get("colocate_all") is True:
        return

    policy_nodes = placement.get("policy_num_nodes")
    ref_nodes = placement.get("ref_num_nodes")
    policy_gpus = placement.get("policy_num_gpus_per_node")
    ref_gpus = placement.get("ref_num_gpus_per_node")
    if not all(isinstance(value, int) and value > 0 for value in (policy_nodes, ref_nodes)):
        return
    expected_nodes = policy_nodes + ref_nodes
    if args.num_nodes != expected_nodes:
        raise SystemExit(
            f"--num-nodes={args.num_nodes} conflicts with {args.rl_config}'s disaggregated placement "
            f"(policy_num_nodes + ref_num_nodes = {expected_nodes})."
        )
    declared_gpus = {value for value in (policy_gpus, ref_gpus) if isinstance(value, int) and value > 0}
    if declared_gpus and (len(declared_gpus) != 1 or args.gpus_per_node not in declared_gpus):
        raise SystemExit(
            f"--gpus-per-node={args.gpus_per_node} conflicts with {args.rl_config}'s placement "
            f"GPU count(s): {sorted(declared_gpus)}."
        )


def _rl_config_harness_name(rl_config: str) -> Optional[str]:
    """Read the configured Harbor harness name without constructing trainer state."""
    try:
        with open(rl_config) as f:
            config = yaml.safe_load(f) or {}
    except OSError:
        return None
    if not isinstance(config, dict):
        return None

    candidate_paths = (
        ("terminal_bench_config", "harbor", "name"),
        ("terminal_bench", "harbor", "name"),
        ("generator", "harbor", "harness", "name"),
        ("generator", "harbor", "name"),
    )
    for path in candidate_paths:
        current: Any = config
        for key in path:
            if not isinstance(current, dict):
                break
            current = current.get(key)
        else:
            if isinstance(current, str) and current.strip():
                return current.strip().lower()
    return None


def _rl_training_strategy(args: argparse.Namespace) -> Optional[str]:
    """Read the effective ``trainer.strategy`` the job will run under.

    A ``--skyrl_override trainer.strategy=...`` wins over the config file, because Hydra applies
    it later. Returns None when neither source declares one.
    """
    for override in reversed(getattr(args, "skyrl_override", None) or []):
        key, separator, value = str(override).partition("=")
        if separator and key.strip().lstrip("+~") == "trainer.strategy":
            return value.strip().strip("'\"").lower() or None

    try:
        with open(args.rl_config) as f:
            config = yaml.safe_load(f) or {}
    except OSError:
        return None
    trainer = config.get("trainer") if isinstance(config, dict) else None
    strategy = trainer.get("strategy") if isinstance(trainer, dict) else None
    return strategy.strip().lower() if isinstance(strategy, str) and strategy.strip() else None


def _sanitize_job_name_component(value: str) -> str:
    """Make one human-readable Kubernetes job-name component."""
    value = value.strip().rstrip("/").split("/")[-1]
    value = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return value or "run"


def derive_default_job_name(
    args: argparse.Namespace,
    *,
    timestamp: Optional[str] = None,
    nonce: Optional[str] = None,
) -> str:
    """Build a unique, valid Iris job name from the selected RL config and model."""
    config_name = _sanitize_job_name_component(Path(args.rl_config).stem)
    model_name = _sanitize_job_name_component(args.model_path)
    timestamp = timestamp or time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    nonce = nonce or secrets.token_hex(3)
    suffix = f"-{timestamp}-{nonce}"
    prefix = f"rl-{config_name}-{model_name}"
    return f"{prefix[: _JOB_NAME_MAX_LENGTH - len(suffix)].rstrip('-')}{suffix}"


def resolve_launch_defaults(args: argparse.Namespace) -> None:
    """Resolve cluster-dependent and harness-dependent defaults before validation."""
    if not args.job_name:
        args.job_name = derive_default_job_name(args)

    _validate_rl_config_topology(args)
    cluster_config = _load_cluster_config(args.cluster_config)
    capacity = _cluster_gpu_cpu_capacity(
        cluster_config,
        gpu_variant=args.gpu_variant,
        gpus_per_node=args.gpus_per_node,
    )

    if args.cpu is None:
        args.cpu = min(capacity, MAX_DEFAULT_CPU_PER_NODE)

    if args.num_nodes > 1 and not args.rendezvous_dir:
        storage_root = _cluster_storage_root(cluster_config)
        args.rendezvous_dir = f"{storage_root}/rendezvous/{args.job_name}"

    if args.record_literal is None:
        harness = _rl_config_harness_name(args.rl_config)
        args.record_literal = harness is None or harness.replace("_", "-") != "terminus-2"

    if args.task_image is None:
        strategy = _rl_training_strategy(args)
        execution_cluster = args.target_cluster or args.cluster
        try:
            args.task_image = image_for_cluster(execution_cluster, strategy).reference
        except ValueError as error:
            raise SystemExit(f"{error}; pass --task-image explicitly.") from error


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


def _rl_config_needs_controller_ingress(rl_config: Optional[str]) -> bool:
    """True ONLY for the OPENCODE harness, which needs the co-located RecordProxy literal
    bridge (token-id/logprob capture for TIS) + the cross-cluster ``/proxy/t`` capability
    URL. Other agentic harnesses (terminus-2) do NOT: they call the served model over the
    DIRECT marinskyrl HTTP endpoint (``ingress_mode=direct``), the historical path, and
    must NOT be force-routed through controller-ingress (the federated ``/proxy/t`` path
    breaks non-streaming terminus-2 -> upstream/stream timeout). Detect the ACTIVE harbor
    harness (``name: opencode``), not mere presence of harbor/terminal_bench blocks."""
    try:
        if not rl_config or not os.path.isfile(rl_config):
            return False
        with open(rl_config, "r") as f:
            text = f.read().lower()
        # Active harness declared as `name: opencode` in the harbor block (ignore comments).
        return any(line.strip().startswith("name:") and "opencode" in line for line in text.splitlines())
    except OSError:
        return False


def autoconfigure_ingress(args: argparse.Namespace) -> None:
    """Derive the controller-ingress config from the target cluster so an agentic CoreWeave
    launch JUST WORKS from ``--target-cluster`` alone — no manual ``--ingress-mode`` /
    ``--ingress-host``.

    Rationale: controller-ingress is required ONLY for the OPENCODE harness (the co-located
    RecordProxy literal bridge + the cross-cluster ``/proxy/t`` capability URL). It is NOT
    the only reachable topology on CoreWeave — the terminus-2 harness reaches the served
    model over the DIRECT marinskyrl HTTP endpoint (``ingress_mode=direct``, the historical
    path) and MUST NOT be force-routed through controller-ingress (federated ``/proxy/t``
    breaks non-streaming terminus-2). So we auto-enable controller ONLY for opencode; for
    that case the ingress host is cluster-determined (``iris.oa.dev``), removing the
    ``--ingress-host`` mismatch error class. Prefer default > flag > env var."""
    target = str(getattr(args, "target_cluster", "") or "")
    cluster = str(getattr(args, "cluster", "") or "")
    is_cw = target.startswith("cw-") or cluster.startswith("cw-")
    # Resolve the "auto" sentinel (the default). An EXPLICIT --ingress-mode direct|controller
    # is ALWAYS honored (explicit flag beats derivation); only "auto" is derived here.
    mode = getattr(args, "ingress_mode", "auto")
    if mode == "auto":
        # Auto-enable controller ONLY for an opencode rl_config on CoreWeave (needs the
        # literal bridge + /proxy/t). terminus-2 & everything else -> direct (marinskyrl HTTP).
        if is_cw and _rl_config_needs_controller_ingress(getattr(args, "rl_config", None)):
            args.ingress_mode = "controller"
            print(
                "[rl-iris] auto: --ingress-mode=controller (opencode rl_config on a CoreWeave target)",
                flush=True,
            )
        else:
            args.ingress_mode = "direct"
    if not is_cw:
        return  # non-CoreWeave: host derivation below is controller/CoreWeave-only
    # (2) The ingress host is cluster-determined on CoreWeave: the marin parent iris.oa.dev.
    if getattr(args, "ingress_mode", "direct") == "controller":
        prev = getattr(args, "ingress_host", None)
        if prev not in (None, "", "iris.oa.dev"):
            print(
                f"[rl-iris] auto: overriding --ingress-host {prev} -> iris.oa.dev "
                "(CoreWeave federated parent; the host is cluster-determined, not a free choice)",
                flush=True,
            )
        elif prev is None:
            print(
                "[rl-iris] auto: --ingress-host=iris.oa.dev (derived from --target-cluster; "
                "CoreWeave federated parent)",
                flush=True,
            )
        args.ingress_host = "iris.oa.dev"


def validate_controller_ingress_reachability(args: argparse.Namespace) -> None:
    """Fail loud BEFORE submit when ``--ingress-mode controller`` would produce a
    capability URL a Daytona sandbox CANNOT reach — the Exp2 opencode-RL blocker
    (ported from OT-Agent 8fdabb12, extended for the federated remediation).

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


def prepare_federated_parent_credentials(args: argparse.Namespace) -> str | None:
    """Validate and return the cached Marin IAP login needed by a federated pod.

    A CoreWeave task has neither a cached human login nor a Marin-allowlisted service
    account. Controller ingress therefore cannot mint a parent capability token unless
    the launcher forwards the operator's cached Marin IAP login record. Mint an IAP
    token here, before submitting or allocating GPUs, so a stale or absent record fails
    locally instead of after the endpoint-registration wait in the task.
    """
    if not getattr(args, "target_cluster", None) or getattr(args, "ingress_mode", "direct") != "controller":
        return None
    if not MARIN_LOGIN_RECORD_PATH.is_file():
        raise SystemExit(
            "[rl-iris] BLOCKED: federated CoreWeave controller ingress requires the cached "
            f"Marin IAP login record at {MARIN_LOGIN_RECORD_PATH}. "
            "Run `iris --cluster=marin login` and relaunch."
        )
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
    return json.dumps(record)


def _default_secrets_env() -> Optional[str]:
    cand = os.environ.get("OT_AGENT_SECRETS_ENV") or os.path.expanduser("~/Documents/secrets.env")
    return cand if os.path.isfile(cand) else None


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch a MarinSkyRL RL training job on the Iris CoreWeave H100 cluster.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- RL job args (mirror launch_rl_cloud.py) ---
    parser.add_argument(
        "--rl_config",
        required=True,
        help="Path to SkyRL/MarinSkyRL config YAML (repo-relative or absolute).",
    )
    parser.add_argument("--rl-config", dest="rl_config", help=argparse.SUPPRESS)

    parser.add_argument(
        "--model_path",
        required=True,
        help="Model path or HuggingFace ID (e.g., Qwen/Qwen3-8B).",
    )
    parser.add_argument("--model-path", dest="model_path", help=argparse.SUPPRESS)

    parser.add_argument(
        "--model-warm-source",
        "--model_warm_source",
        dest="model_warm_source",
        default=None,
        help="In-region CW-object-store prefix seeded (once, via scripts/iris/"
        "mirror_hf_to_s3.py) with the model weights, so the controller SYNCS them "
        "from there into each node's HF cache instead of cold-pulling ~160 GB per "
        "node from HF Hub (the flaky path behind the 80B r4a/r4b bring-up failures). "
        "Default: AUTO-DERIVE s3://marin-us-east-02a/models/<org>--<name> from the "
        "repo id (a missing/empty source is a clean no-op -> HF prestage fallback, "
        "byte-identical to today). Pass 'none'/'off' to DISABLE the warm path (pure "
        "HF prestage). Only used when the config runs HF_HUB_OFFLINE=1 with a "
        "repo-id model_path (same gate as --prestage-model).",
    )

    parser.add_argument(
        "--train_data",
        default="[]",
        help="Training data paths as a JSON list (e.g., '[\"org/dataset\"]').",
    )
    parser.add_argument("--train-data", dest="train_data", help=argparse.SUPPRESS)

    parser.add_argument(
        "--val_data",
        default="[]",
        help="Validation data paths as a JSON list.",
    )
    parser.add_argument("--val-data", dest="val_data", help=argparse.SUPPRESS)

    parser.add_argument(
        "--skyrl_override",
        action="append",
        default=[],
        help="SkyRL Hydra override (repeatable).",
    )
    parser.add_argument("--skyrl-override", dest="skyrl_override", action="append", help=argparse.SUPPRESS)

    parser.add_argument(
        "--experiments_dir",
        default="/app/experiments",
        help="In-container experiments output dir (on the synced /app workspace).",
    )
    parser.add_argument("--experiments-dir", dest="experiments_dir", help=argparse.SUPPRESS)

    # --- Resource / topology args (GPU multi-node) ---
    parser.add_argument(
        "--num-nodes",
        "--num_nodes",
        dest="num_nodes",
        type=int,
        default=1,
        help="Number of WHOLE H100 nodes to request EXCLUSIVELY, gang/co-scheduled "
        "(one iris task per node, all 8 GPUs each, coscheduled by leafgroup/IB).",
    )
    parser.add_argument(
        "--gpus-per-node",
        "--gpus_per_node",
        dest="gpus_per_node",
        type=int,
        default=DEFAULT_GPUS_PER_NODE,
        help="GPUs per node (CoreWeave nodes are 8x H100).",
    )
    parser.add_argument(
        "--gpu-variant",
        "--gpu_variant",
        dest="gpu_variant",
        default=DEFAULT_GPU_VARIANT,
        help="GPU variant (default H100).",
    )
    parser.add_argument(
        "--cpu",
        type=float,
        default=None,
        help="CPU cores per node. Default: derive from the selected GPU cluster's scale group "
        f"with a scheduling-safe cap of {MAX_DEFAULT_CPU_PER_NODE:g}.",
    )
    parser.add_argument(
        "--memory",
        default=DEFAULT_MEMORY_PER_NODE,
        help="Memory per node.",
    )
    parser.add_argument(
        "--disk",
        default=DEFAULT_DISK_PER_NODE,
        help=f"Ephemeral disk per node. Default 'auto' = {int(DISK_FRACTION * 100)}%% of the GPU "
        "node's live allocatable ephemeral-storage (whole-node-exclusive gangs have no "
        "co-tenants, so claim most of the node NVMe — keeps Ray object-spill / checkpoints "
        "clear of the ephemeral-storage eviction). Pass an explicit value (e.g. 4000GB) to override.",
    )
    parser.add_argument(
        "--ray-port",
        "--ray_port",
        dest="ray_port",
        type=int,
        default=6379,
        help="Port the cross-node Ray head binds.",
    )
    parser.add_argument(
        "--rendezvous-dir",
        "--rendezvous_dir",
        dest="rendezvous_dir",
        default=None,
        help="Shared object-store/path (gs://, s3://, or shared dir) for the multi-node "
        "Ray head/worker rendezvous. Required for --num-nodes>1. On cw-us-east-02a "
        "use an s3:// URI under the cluster's default bucket, e.g. "
        "s3://marin-us-east-02a/iris/rl-rdv/<job>; the cluster injects working creds "
        "+ AWS_ENDPOINT_URL into every task pod (iris-task-env Secret), so no external "
        "creds are needed. NOTE: the default object store moved R2 (s3://marin-na) -> "
        "CW (s3://marin-us-east-02a) on 2026-07-05 (marin c7caecc95a); pods now inject "
        "CW creds+endpoint and can NO LONGER reach s3://marin-na (R2).",
    )
    parser.add_argument(
        "--rendezvous-timeout",
        "--rendezvous_timeout",
        dest="rendezvous_timeout",
        type=int,
        default=None,
        help="Seconds the worker ranks poll for rank-0's Ray-head rendezvous file "
        "(forwarded to start_rl_iris_controller.py --rendezvous-timeout). Unset = the "
        "controller default (1800s). RAISE it (e.g. 3600) for a big model whose rank-0 "
        "pre-stage/snapshot_download can legitimately take >30 min, so a SLOW-but-not-hung "
        "head prestage completes inside the window instead of the workers timing out and "
        "killing the gang (the 80B rank-spread bring-up flake, 2026-07-11).",
    )
    parser.add_argument(
        "--trials-dir",
        "--trials_dir",
        dest="trials_dir",
        default="auto",
        help="Where Harbor writes per-trial agentic-RL rollout artifacts "
        "(terminal_bench_config.trials_dir). 'auto' (default) = the durable shared store at "
        "s3://marin-us-east-02a/iris/<job_name>/trace_jobs (pods reach it via auto-injected "
        "creds; inspectable post-hoc). 'local'/'off' = keep the config default (node-local "
        "/app/experiments/<run>/trace_jobs). Or pass an explicit s3://, gs://, or path URI. "
        "Ignored if you already set terminal_bench_config.trials_dir via --skyrl_override.",
    )

    # --- Iris submission args (mirror launch_eval_iris.py / IrisLauncher) ---
    parser.add_argument(
        "--cluster",
        default=DEFAULT_CLUSTER,
        help="Iris cluster name (default cw-us-east-02a).",
    )
    parser.add_argument(
        "--cluster-config",
        "--cluster_config",
        dest="cluster_config",
        default=None,
        help="Path to the iris cluster YAML. Default: auto-resolve lib/iris/config/"
        "<--cluster>.yaml in the marin repo (so --cluster cw-rno2a targets the 512xH100 "
        "RNO2A cluster without a manual --cluster-config).",
    )
    parser.add_argument(
        "--task-image",
        "--task_image",
        "--docker_image",
        "--docker-image",
        dest="task_image",
        default=None,
        help="Container image. Default: select an immutable digest from the execution cluster "
        "and the config's trainer.strategy.",
    )
    parser.add_argument(
        "--job-name",
        "--job_name",
        dest="job_name",
        default=None,
        help="Job name (auto-derived if not set).",
    )
    parser.add_argument(
        "--priority",
        default=DEFAULT_PRIORITY,
        choices=PRIORITY_NAMES,
        help="Iris priority band.",
    )
    parser.add_argument(
        "--max-retries",
        "--max_retries",
        dest="max_retries",
        type=int,
        default=6,
        help="Max retries on failure (iris auto-retries preemptions separately).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=0,
        help="Job timeout in seconds (0 = no timeout).",
    )
    parser.add_argument(
        "--no-wait",
        dest="no_wait",
        action="store_true",
        default=False,
        help="Submit and detach instead of streaming logs.",
    )
    parser.add_argument(
        "--preemptible",
        dest="preemptible",
        action="store_true",
        default=None,
        help="Force scheduling on preemptible workers.",
    )
    parser.add_argument(
        "--no-preemptible",
        dest="preemptible",
        action="store_false",
        help="Force scheduling on non-preemptible workers.",
    )
    # ----------------------------------------------------------------------- #
    # Cross-cluster ingress / federated submission (Exp2 opencode-RL fix #1).   #
    #                                                                           #
    # The default (direct) path is UNCHANGED: submit straight to --cluster's    #
    # own controller (byte-identical to before). The federated path is opt-in   #
    # via --target-cluster: submit through the marin meta-scheduler             #
    # (iris.oa.dev) with a `cluster EQ <peer>` constraint so marin DELEGATES    #
    # the whole job to the peer child and can then federation-proxy /proxy      #
    # requests to the peer's endpoint. This is the ONLY topology in which a     #
    # Daytona sandbox can reach a co-located CoreWeave vLLM through a single    #
    # public host (iris.oa.dev): the peer controller's own host is IP-locked    #
    # with no off-cluster surface, and marin only federates /proxy for a job it #
    # delegated (controller has_received_job_from_peer). See                    #
    # validate_controller_ingress_reachability() + .claude/ops/iris/iris_ingress.md. #
    # ----------------------------------------------------------------------- #
    parser.add_argument(
        "--ingress-mode",
        "--ingress_mode",
        dest="ingress_mode",
        default="auto",
        choices=["auto", "direct", "controller"],
        help="How the co-located served model (RecordProxy/vLLM) is exposed to a "
        "Daytona sandbox. 'auto' (default) = derive per harness (opencode->controller on "
        "CoreWeave, everything else->direct); an EXPLICIT 'direct'/'controller' always wins. "
        "'direct' = legacy path, no controller-ingress "
        "wiring (byte-identical). 'controller' = register the endpoint with the iris "
        "controller and serve it through the /proxy/t/<token>/... capability URL; on a "
        "CoreWeave cluster this REQUIRES --target-cluster (federated submission) so the "
        "capability URL is reachable — see validate_controller_ingress_reachability().",
    )
    parser.add_argument(
        "--ingress-host",
        "--ingress_host",
        dest="ingress_host",
        default=None,
        help="Public controller-ingress host the sandbox-facing capability URL is built "
        "against (only used with --ingress-mode controller). For the federated CoreWeave "
        "path this MUST be the marin meta-scheduler host 'iris.oa.dev' (the parent that "
        "owns the mirrored endpoint + signs the token), NOT the peer's own host.",
    )
    parser.add_argument(
        "--target-cluster",
        "--target_cluster",
        dest="target_cluster",
        default=None,
        help="Federate the whole job to this peer cluster via the marin meta-scheduler "
        "instead of submitting directly to --cluster's controller. Appends a "
        "`cluster EQ <peer>` constraint and submits through the marin controller "
        "(iris.oa.dev, IAP-gated — needs `iris login`), so marin delegates the job to "
        "the peer child and can federation-proxy /proxy to the peer's endpoint. Required "
        "to make --ingress-mode controller reachable from Daytona on CoreWeave. Leave "
        "unset for the default direct submission.",
    )
    parser.add_argument(
        "--parent-cluster-config",
        "--parent_cluster_config",
        dest="parent_cluster_config",
        default=None,
        help="Path to the PARENT (marin) iris cluster YAML used for federated submission "
        "when --target-cluster is set. Defaults to the marin.yaml sibling of "
        "--cluster-config. The direct path never reads this.",
    )
    parser.add_argument(
        "--record-literal",
        "--record_literal",
        dest="record_literal",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Co-locate Harbor's RecordProxy in front of vLLM to capture literal.jsonl. "
        "Default: enabled for every harness except terminus-2. Pass --record-literal to force "
        "it on or --no-record-literal to opt out. It is forwarded when controller ingress is used.",
    )
    parser.add_argument(
        "--parent-controller-config-in-pod",
        "--parent_controller_config_in_pod",
        dest="parent_controller_config_in_pod",
        default=None,
        help="In-pod path to the parent (marin) cluster YAML the in-pod worker mints "
        "against, if it differs from the launch-host --parent-cluster-config path. "
        "Defaults to the launch-host resolved marin.yaml path (must be materialized "
        "in-pod — see the OTAGENT_PARENT_CONTROLLER_CONFIG forwarding NOTE).",
    )
    parser.add_argument(
        "--secrets-env",
        "--secrets_env",
        dest="secrets_env",
        default=_default_secrets_env(),
        help="KEY=VALUE env file injected into the task (HF_TOKEN, WANDB_API_KEY, etc.). "
        "Defaults to $OT_AGENT_SECRETS_ENV, else ~/Documents/secrets.env.",
    )
    parser.add_argument(
        "--iris-ref",
        "--iris_ref",
        dest="iris_ref",
        default=DEFAULT_IRIS_VERSION,
        help="marin-iris pip spec installed into the RL venv at pod bootstrap for the "
        "controller-ingress registration/mint path (GAP D: iris is NOT baked into the "
        "gpu-rl image). Only installed under --ingress-mode controller (direct mode is "
        f"byte-identical, no install). Default: {DEFAULT_IRIS_VERSION}.",
    )
    # ----------------------------------------------------------------------- #
    # MarinSkyRL runtime-knob flags (deslop stage 3). Each promotes a live      #
    # SKYRL_* runtime env var to a first-class CLI flag. ALL default to None    #
    # ("unspecified") so an all-defaults launch injects NOTHING and the pod env #
    # is byte-identical to today (the SkyRL code's own default applies). A       #
    # config's `extra_env:` block is overlaid on TOP of these, so extra_env      #
    # still wins: precedence is  env/extra_env > flag > code-default.            #
    # ----------------------------------------------------------------------- #
    g = parser.add_argument_group("MarinSkyRL runtime knobs (SKYRL_* -> flags)")
    g.add_argument(
        "--r3-transport",
        "--r3_transport",
        dest="r3_transport",
        choices=["by_value", "resident", "decentral"],
        default=None,
        help="R3 (rollout routed-experts) transport for MoE async RL. 'decentral' "
        "(code default) routes the captured routed-experts generation-worker -> "
        "node-resident consumer (head holds ~0 R3); 'resident' de-dups to 1 "
        "copy/dp-group on the driver head plasma; 'by_value' is the old per-actor "
        "by-value dispatch. Folds SKYRL_R3_RESIDENT + SKYRL_R3_DECENTRAL. "
        "Default: unset = code default (decentral).",
    )
    g.add_argument(
        "--r3-put-timeout-s",
        "--r3_put_timeout_s",
        dest="r3_put_timeout_s",
        type=int,
        default=None,
        help="Bounded ray.put() timeout (s) for an R3 dp-chunk dispatch "
        "(SKYRL_DISPATCH_PUT_TIMEOUT_S). Default: unset = 600.",
    )
    g.add_argument(
        "--nccl-timeout-s",
        "--nccl_timeout_s",
        dest="nccl_timeout_s",
        type=int,
        default=None,
        help="Worker NCCL-collective timeout in seconds (SKYRL_WORKER_NCCL_TIMEOUT_IN_S). Default: unset = 1800.",
    )
    g.add_argument(
        "--host-ram-monitor",
        dest="host_ram_monitor",
        choices=["on", "off"],
        default=None,
        help="Policy-worker host-RAM/cgroup-mem monitor thread (SKYRL_POLICY_HOST_RAM_MONITOR). Default: unset = on.",
    )
    g.add_argument(
        "--host-ram-monitor-interval-s",
        dest="host_ram_monitor_interval_s",
        type=int,
        default=None,
        help="Host-RAM monitor sample interval, s (SKYRL_POLICY_HOST_RAM_MONITOR_INTERVAL). Default: unset = 60.",
    )
    g.add_argument(
        "--tis-splice",
        dest="tis_splice",
        choices=["on", "off"],
        default=None,
        help="TIS served-id splice policy (SKYRL_TIS_SPLICE) — use vLLM's raw served "
        "token ids as the generated region for exact-by-id TIS alignment. "
        "Default: unset = on (no-op on non-thinking turns).",
    )
    g.add_argument(
        "--gdn-mask-fla",
        dest="gdn_mask_fla",
        choices=["auto", "on", "off"],
        default=None,
        help="Force the pure-torch GatedDeltaNet path / mask the broken fla wheel "
        "(SKYRL_GDN_MASK_FLA). 'auto' (and unset) derive it from the model arch "
        "(on for Qwen3-Next/GDN, off for dense). Default: unset = auto.",
    )
    g.add_argument(
        "--gdn-flashqla",
        dest="gdn_flashqla",
        choices=["on", "off"],
        default=None,
        help="Opt-in FlashQLA fused GDN tilelang kernel (SKYRL_GDN_FLASHQLA); needs the "
        "fla_tilelang overlay. Default: unset = off.",
    )
    g.add_argument(
        "--forward-dispatch-fix",
        dest="forward_dispatch_fix",
        choices=["on", "off"],
        default=None,
        help="MoE async-dispatch forward fix (SKYRL_FORWARD_DISPATCH_FIX), a correctness "
        "knob. Default: unset = on. Pass off only for an A/B.",
    )
    g.add_argument(
        "--weightsync-drain-barrier",
        dest="weightsync_drain_barrier",
        choices=["on", "off"],
        default=None,
        help="Post-weight-sync async drain barrier (SKYRL_WEIGHTSYNC_DRAIN_BARRIER), a "
        "correctness knob. Default: unset = on.",
    )
    g.add_argument(
        "--cp-require-right-align",
        dest="cp_require_right_align",
        choices=["on", "off"],
        default=None,
        help="Require right-aligned attention mask under context-parallel "
        "(SKYRL_CP_REQUIRE_RIGHT_ALIGN), a correctness knob. Default: unset = on.",
    )
    g.add_argument(
        "--w13-reload-bracket",
        dest="w13_reload_bracket",
        choices=["on", "off"],
        default=None,
        help="Bracket the MoE weight-sync with layerwise-reload init/finalize so FusedMoE "
        "w13 is re-swapped exactly once (SKYRL_W13_RELOAD_BRACKET), a correctness "
        "knob. Default: unset = on.",
    )
    g.add_argument(
        "--ep-loader-chunk-rows",
        dest="ep_loader_chunk_rows",
        type=int,
        default=None,
        help="Per-broadcast dim-0 row budget for the streamed EP full-state-dict loader "
        "(SKYRL_EP_LOADER_CHUNK_ROWS). Default: unset = 8.",
    )
    g.add_argument(
        "--collective-count-diag",
        dest="collective_count_diag",
        choices=["on", "off"],
        default=None,
        help="GC-proof per-rank default-PG collective-count instrumentation "
        "(SKYRL_COLLECTIVE_COUNT_DIAG), a DIAGNOSTIC knob for the 80B gs1 NCCL "
        "desync. Logs each policy rank's default-PG collective count at forward "
        "phase boundaries (forward/_forward_impl enter+exit + the first MoE-EP "
        "all-to-all per forward) to the finelog, which survives pod GC — diffing "
        "the counts across ranks at the wedge localizes the divergent EP group. "
        "O(phases), reads torch's own PG seq counter (no perturbation). "
        "Default: unset = off.",
    )

    parser.add_argument(
        "--dry-run",
        "--dry_run",
        dest="dry_run",
        action="store_true",
        default=False,
        help="Print the resolved config + in-container command without submitting.",
    )

    return parser


def priority_band_enum_name(priority: str) -> str:
    """Return the Iris protobuf enum name for a supported CLI priority.

    Raises:
        KeyError: If ``priority`` is not one of ``PRIORITY_NAMES``.
    """
    if priority not in PRIORITY_NAMES:
        raise KeyError(priority)
    return f"PRIORITY_BAND_{priority.upper()}"


def build_skyrl_flag_env(args: argparse.Namespace) -> dict[str, str]:
    """Translate the MarinSkyRL runtime-knob CLI flags into SKYRL_* env vars for the
    pod. Only flags that were explicitly set (non-None) emit an entry, so an
    all-defaults invocation returns {} and the pod env stays byte-identical to today.
    The caller overlays the config's ``extra_env:`` on top of this, so a config's
    explicit value still wins (precedence: env/extra_env > flag > code default)."""
    env: dict[str, str] = {}

    def _onoff(name: str, value) -> None:
        if value is not None:
            env[name] = "1" if value == "on" else "0"

    # R3 transport: fold the nested resident && decentral gating into one choice.
    if args.r3_transport == "by_value":
        env["SKYRL_R3_RESIDENT"] = "0"
    elif args.r3_transport == "resident":
        env["SKYRL_R3_RESIDENT"] = "1"
        env["SKYRL_R3_DECENTRAL"] = "0"
    elif args.r3_transport == "decentral":
        env["SKYRL_R3_RESIDENT"] = "1"
        env["SKYRL_R3_DECENTRAL"] = "1"
    if args.r3_put_timeout_s is not None:
        env["SKYRL_DISPATCH_PUT_TIMEOUT_S"] = str(args.r3_put_timeout_s)
    if args.nccl_timeout_s is not None:
        env["SKYRL_WORKER_NCCL_TIMEOUT_IN_S"] = str(args.nccl_timeout_s)
    _onoff("SKYRL_POLICY_HOST_RAM_MONITOR", args.host_ram_monitor)
    if args.host_ram_monitor_interval_s is not None:
        env["SKYRL_POLICY_HOST_RAM_MONITOR_INTERVAL"] = str(args.host_ram_monitor_interval_s)
    _onoff("SKYRL_TIS_SPLICE", args.tis_splice)
    # GDN mask: 'auto' (like unset) leaves the env unset so the code auto-derives.
    if args.gdn_mask_fla in ("on", "off"):
        env["SKYRL_GDN_MASK_FLA"] = "1" if args.gdn_mask_fla == "on" else "0"
    _onoff("SKYRL_GDN_FLASHQLA", args.gdn_flashqla)
    _onoff("SKYRL_FORWARD_DISPATCH_FIX", args.forward_dispatch_fix)
    _onoff("SKYRL_WEIGHTSYNC_DRAIN_BARRIER", args.weightsync_drain_barrier)
    _onoff("SKYRL_CP_REQUIRE_RIGHT_ALIGN", args.cp_require_right_align)
    _onoff("SKYRL_W13_RELOAD_BRACKET", args.w13_reload_bracket)
    if args.ep_loader_chunk_rows is not None:
        env["SKYRL_EP_LOADER_CHUNK_ROWS"] = str(args.ep_loader_chunk_rows)
    _onoff("SKYRL_COLLECTIVE_COUNT_DIAG", args.collective_count_diag)
    return env


def _load_rl_config_yaml(rl_config_path: str) -> dict:
    """Resolve an RL config path (repo-relative, else as given) and parse its YAML to a dict.

    Raises on an unreadable/invalid file; callers that want a soft default wrap this."""
    full = PROJECT_ROOT / rl_config_path
    path = full if full.exists() else Path(rl_config_path)
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_config_extra_env(rl_config_path: str) -> dict[str, str]:
    """Read a top-level ``extra_env:`` mapping from the RL config YAML.

    The Iris path has NO ``container:`` block (the gpu-rl Docker image is the
    runtime), so any SLURM-style ``container.extra_env`` shell-export plumbing never
    runs — without this, env declared in the YAML is silently
    dropped and only the launcher's hardcoded passthrough (HF/WANDB/DAYTONA) reaches
    the pod. This forwards a top-level ``extra_env:`` block (and, defensively,
    ``container.extra_env`` if a ported config still carries one) into the iris
    EnvironmentSpec so e.g. EPDIAG probe arms + R3/DCP guard env take effect.

    Values are coerced to str (YAML may parse "1"/true as int/bool). Returns {} if
    the file is unreadable or declares no extra_env (byte-identical behavior for the
    existing extra_env-less iris configs).
    """
    try:
        raw = _load_rl_config_yaml(rl_config_path)
    except Exception as exc:  # noqa: BLE001
        print(f"[rl-iris] WARNING: could not read extra_env from {rl_config_path}: {exc}", file=sys.stderr)
        return {}
    extra = dict(raw.get("extra_env") or {})
    container_env = (raw.get("container") or {}).get("extra_env") or {}
    for k, v in container_env.items():
        extra.setdefault(k, v)
    out: dict[str, str] = {}
    for k, v in extra.items():
        if v is None:
            continue
        if isinstance(v, bool):
            v = int(v)
        out[str(k)] = str(v)
    return out


def load_config_policy_chat_template(rl_config_path: str) -> Optional[str]:
    """The config's top-level ``policy_chat_template`` (repo-relative jinja path), or None.

    None when the key is unset (existing configs have no such key). Set only by single-turn
    RLVR configs that must force a chat template onto the policy tokenizer cache because the
    SFT repo ships none. Read errors propagate: this drives fail-loud template machinery, so
    an unreadable config must abort rather than silently skip the override."""
    value = _load_rl_config_yaml(rl_config_path).get("policy_chat_template")
    return str(value) if value else None


def load_config_trainer_ckpt_path(rl_config_path: str) -> Optional[str]:
    """Return an EXPLICIT ``trainer.ckpt_path`` from the RL config YAML, else None.

    The iris configs set ``ckpt_path: null`` (auto-derived downstream in
    rl_config_translation). A config that sets it explicitly (non-null, non-empty) should
    WIN over the launcher's durable-s3 default, so build_task_command consults this
    before injecting its override. Returns None when the file is unreadable, has no
    ``trainer.ckpt_path``, or the value is null/empty (byte-identical to today for
    every existing iris config, which all leave it null)."""
    try:
        raw = _load_rl_config_yaml(rl_config_path)
    except Exception as exc:  # noqa: BLE001
        print(f"[rl-iris] WARNING: could not read ckpt_path from {rl_config_path}: {exc}", file=sys.stderr)
        return None
    val = (raw.get("trainer") or {}).get("ckpt_path")
    if val is None or (isinstance(val, str) and not val.strip()):
        return None
    return str(val)


def load_config_trainer_export_path(rl_config_path: str) -> Optional[str]:
    """Return an EXPLICIT ``trainer.export_path`` from the RL config YAML, else None.

    Same contract as ``load_config_trainer_ckpt_path``: an explicitly set value wins over
    the launcher's durable default, and ``null``/empty means "derive one"."""
    try:
        raw = _load_rl_config_yaml(rl_config_path)
    except Exception as exc:  # noqa: BLE001
        print(f"[rl-iris] WARNING: could not read export_path from {rl_config_path}: {exc}", file=sys.stderr)
        return None
    val = (raw.get("trainer") or {}).get("export_path")
    if val is None or (isinstance(val, str) and not val.strip()):
        return None
    return str(val)


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


def normalize(args: argparse.Namespace) -> None:
    """Validate + normalize. Keep rl_config repo-relative so it resolves on /app."""
    # Resolve rl_config to a repo-relative path (it must exist on the synced
    # /app workspace, NOT be an absolute host path).
    rl_cfg = Path(args.rl_config)
    if rl_cfg.is_absolute():
        try:
            args.rl_config = str(rl_cfg.resolve().relative_to(PROJECT_ROOT))
        except ValueError:
            raise SystemExit(
                f"--rl_config {args.rl_config!r} is absolute and not under the repo "
                f"({PROJECT_ROOT}); pass a repo-relative path so it resolves on /app."
            )
    # Verify it exists locally (so we fail fast before submitting).
    if not (PROJECT_ROOT / args.rl_config).exists():
        # Fall back to cloud/iris/configs/<name>[.yaml].
        yaml_dir = Path("cloud/iris/configs")
        for cand in (yaml_dir / args.rl_config, yaml_dir / f"{args.rl_config}.yaml"):
            if (PROJECT_ROOT / cand).exists():
                args.rl_config = str(cand)
                break
        else:
            print(
                f"[rl-iris] WARNING: --rl_config {args.rl_config!r} not found under "
                f"{PROJECT_ROOT}; the worker will error if it isn't on /app.",
                file=sys.stderr,
            )

    if args.num_nodes < 1:
        raise SystemExit("--num-nodes must be >= 1.")
    if args.gpus_per_node < 1:
        raise SystemExit("--gpus-per-node must be >= 1.")


def build_task_command(args: argparse.Namespace) -> List[str]:
    """Build the in-container command, multi-node-aware.

    The full pipeline that runs inside each task container:
      cd /app
      && export SKYRL_HOME + PYTHONPATH (live /app + skyrl-train win)
      && <RL_PYTHON> cloud/iris/start_rl_iris_controller.py
            --ray-port ... --rendezvous-dir ...
            -- <RL_PYTHON> -m cloud.iris.run_rl --rl_config ... --num_nodes N ...

    Rank 0 (IRIS_TASK_ID==0) starts the Ray head and runs run_rl.py (which, with
    RAY_ADDRESS set + --num_nodes>1, attaches to the cluster instead of starting a
    local one). Workers join Ray and park. We invoke the gpu-rl venv python by
    absolute path so it is used regardless of whichever venv iris's setup phase
    activates.
    """
    total_gpus = args.num_nodes * args.gpus_per_node

    # The MarinSkyRL training command rank 0 runs (run_rl.py owns config parse,
    # hydra-arg build, HF data resolution, and the SkyRL entrypoint launch).
    train_cmd: List[str] = [
        RL_PYTHON,
        "-m",
        "cloud.iris.run_rl",
        "--rl_config",
        args.rl_config,
        "--model_path",
        args.model_path,
        "--job_name",
        args.job_name,
        "--gpus",
        str(total_gpus),
        "--num_nodes",
        str(args.num_nodes),
        "--gpus_per_node",
        str(args.gpus_per_node),
        "--experiments_dir",
        args.experiments_dir,
        "--ray_port",
        str(args.ray_port),
    ]
    if args.train_data and args.train_data != "[]":
        train_cmd.extend(["--train_data", args.train_data])
    if args.val_data and args.val_data != "[]":
        train_cmd.extend(["--val_data", args.val_data])
    for override in args.skyrl_override or []:
        train_cmd.extend(["--skyrl_override", override])

    # Cross-cluster ingress (opencode-RL literal capture): forward the ingress flags to
    # the in-pod runner (cloud.iris.run_rl), which stands up the RecordProxy + registers
    # + mints the capability URL. Only emitted under --ingress-mode controller; the
    # default (direct) path adds nothing (byte-identical run_rl invocation).
    if getattr(args, "ingress_mode", "direct") == "controller":
        train_cmd.extend(["--ingress_mode", "controller"])
        if getattr(args, "ingress_host", None):
            train_cmd.extend(["--ingress_host", args.ingress_host])
        if getattr(args, "target_cluster", None):
            train_cmd.extend(["--target_cluster", args.target_cluster])
            # Parent (marin) config the in-pod worker mints against. Prefer an explicit
            # in-pod path; else pass the resolved marin.yaml path (must be materialized
            # in-pod — see the OTAGENT_PARENT_CONTROLLER_CONFIG env forwarding + NOTE).
            parent_cfg_in_pod = getattr(args, "parent_controller_config_in_pod", None) or (
                args.parent_cluster_config or _resolve_parent_cluster_config(args.cluster_config)
            )
            if parent_cfg_in_pod:
                train_cmd.extend(["--parent_controller_config", parent_cfg_in_pod])
    if getattr(args, "record_literal", False):
        train_cmd.append("--record_literal")

    # Durable Harbor rollout artifacts. The config default (trials_dir: null) resolves to a
    # node-local path on the rank-0 pod (/app/experiments/<run>/trace_jobs); point
    # terminal_bench_config.trials_dir at the durable shared store (s3://, creds auto-injected)
    # so rollouts persist + are inspectable post-hoc. Skip if the user opted out
    # (--trials-dir local) or already set it explicitly.
    trials_dir = (args.trials_dir or "auto").strip()
    user_set_trials = any("terminal_bench_config.trials_dir=" in o for o in (args.skyrl_override or []))
    if trials_dir.lower() not in ("local", "off", "none", "") and not user_set_trials:
        if trials_dir.lower() == "auto":
            trials_dir = f"s3://marin-us-east-02a/iris/{args.job_name}/trace_jobs"
        train_cmd.extend(["--skyrl_override", f"++terminal_bench_config.trials_dir={trials_dir}"])

    # Durable RESUMABLE checkpoint (preempt-safe -> makes `--priority batch` safe for
    # long runs). Without this, trainer.ckpt_path auto-derives (rl_config_translation) to
    # {experiments_dir}/{job_name}/checkpoints, and on iris experiments_dir defaults to
    # the in-container /app/experiments — EPHEMERAL pod-local disk. A batch preempt +
    # re-admit wipes it, so the trainer can't find latest_ckpt_global_step.txt and
    # restarts from step 0 despite resume_mode: latest. Redirect ckpt_path to a STABLE
    # per-job path on the durable CW object store — SAME bucket + auto-injected creds
    # path as trials_dir above, so ckpt co-locates with rollouts and follows any store
    # migration identically. It MUST be keyed on job_name ONLY (NOT a fresh-per-attempt
    # sub-path) so a re-admitted SAME --job-name job finds latest_ckpt_global_step.txt
    # (read path == write path, MarinSkyRL skyrl-train utils/io/io.py is fsspec-s3) and
    # auto-resumes from the banked step. iris-ONLY: SLURM uses a different launcher where
    # experiments_dir is durable $WORK, so this path never runs there. Respect an
    # explicit ckpt_path from the YAML or a --skyrl_override (either wins).
    user_set_ckpt = any("trainer.ckpt_path=" in o for o in (args.skyrl_override or []))
    yaml_ckpt = load_config_trainer_ckpt_path(args.rl_config)
    if not user_set_ckpt and not yaml_ckpt:
        ckpt_path = f"s3://marin-us-east-02a/iris/{args.job_name}/checkpoints"
        train_cmd.extend(["--skyrl_override", f"++trainer.ckpt_path={ckpt_path}"])
        print(f"[rl-iris] Durable resumable ckpt_path: {ckpt_path}")

    # export_path needs the SAME durable treatment as ckpt_path, and for a sharper reason.
    # Left unset it is auto-derived to a node-local <experiments_dir>/<job>/exports
    # (rl_config_translation.py). save_hf_model then writes the HF export on a POLICY WORKER
    # while HFHubUploadCallback runs on the DRIVER, so the callback finds nothing, every
    # upload fails, and the run ends with an empty repo — after which both nodes are reclaimed
    # and the export is gone for good. Every completed run in the 2026-07 sweep lost its
    # published model exactly this way; the training checkpoints survived only because
    # ckpt_path was already durable. The callback reads through skyrl's fsspec io layer, so an
    # s3:// export_path is visible from any node. Respect an explicit value from the YAML or a
    # --skyrl_override (either wins).
    user_set_export = any("trainer.export_path=" in o for o in (args.skyrl_override or []))
    yaml_export = load_config_trainer_export_path(args.rl_config)
    if not user_set_export and not yaml_export:
        export_path = f"s3://marin-us-east-02a/iris/{args.job_name}/exports"
        train_cmd.extend(["--skyrl_override", f"++trainer.export_path={export_path}"])
        print(f"[rl-iris] Durable export_path: {export_path}")

    # The controller wraps the training command for the multi-node Ray bootstrap.
    controller_cmd: List[str] = [
        RL_PYTHON,
        "cloud/iris/start_rl_iris_controller.py",
        "--ray-port",
        str(args.ray_port),
    ]
    if args.rendezvous_dir:
        controller_cmd.extend(["--rendezvous-dir", args.rendezvous_dir])
    # Worker rendezvous poll deadline. Unset = controller default (1800s). Raise it when
    # rank-0's per-node pre-stage of a large model can legitimately exceed 30 min, so a
    # slow-but-not-hung head prestage completes before the workers give up + kill the gang.
    if args.rendezvous_timeout is not None:
        controller_cmd.extend(["--rendezvous-timeout", str(args.rendezvous_timeout)])
    # Per-NODE task-dataset staging. run_rl.py's resolve_rl_train_data() extracts the
    # HF task dataset to the node-local $DCFT=/opt/openthoughts/tasks/ (gpu-rl image),
    # but it runs ONLY on rank 0 (the head), so the Ray-scheduled rollout workers on
    # ranks 1..N-1 find an empty tasks dir and every rollout dies with
    # FileNotFoundError: .../task.toml -> reward always 0 (data-starved, doomed run).
    # Fix: forward --train-data to the controller so it can run the SAME extraction
    # on EVERY node before Ray starts, populating the identical node-local path on
    # all pods. Idempotent (on_exist=skip) — rank-0's later run_rl re-resolve is a
    # cheap no-op.
    if args.train_data and args.train_data != "[]":
        controller_cmd.extend(["--train-data", args.train_data])
    # Per-NODE model pre-staging, coupled to HF_HUB_OFFLINE. A config that runs the
    # FSDP ranks offline (extra_env HF_HUB_OFFLINE=1) has NO warm cache unless the
    # weights are pulled first; without pre-staging each of the N*8 ranks would race
    # HF Hub online inside init_model and a slow straggler blows the 20-min c10d store
    # barrier (the 80B init-straggle kill, 2026-07-10). When the config is offline,
    # forward the model repo-id so the controller pre-downloads it ONCE PER NODE into
    # the node-local HF cache before Ray — off the collective critical path. Online
    # configs are byte-identical (no flag forwarded).
    _cfg_env = load_config_extra_env(args.rl_config)
    _policy_chat_template = load_config_policy_chat_template(args.rl_config)
    _offline = str(_cfg_env.get("HF_HUB_OFFLINE", "")).strip().lower() in ("1", "true", "yes", "on")
    # A policy_chat_template override rewrites the node-local tokenizer cache, so it REQUIRES
    # a prestage even when the config is not offline (nothing to rewrite otherwise).
    if _offline or _policy_chat_template:
        if args.model_path and not args.model_path.startswith(("s3://", "gs://", "gcs://")):
            controller_cmd.extend(["--prestage-model", args.model_path])
            # In-region warm source. Default = auto-derive the CW-S3 convention path from
            # the repo id; a seed job (mirror_hf_to_s3.py) populates it once and every node
            # then S3-syncs from there instead of cold-pulling from HF Hub. When the source
            # is un-seeded the controller falls back to the HF prestage (byte-identical to
            # pre-warm-path). 'none'/'off' disables the warm path entirely (pure HF prestage).
            warm = args.model_warm_source
            if warm is None:
                warm = f"s3://marin-us-east-02a/models/{args.model_path.replace('/', '--')}"
            elif warm.strip().lower() in ("none", "off", ""):
                warm = None
            if warm:
                controller_cmd.extend(["--model-warm-source", warm])
    # Force the delphi chat template onto every node's tokenizer cache (single-turn RLVR).
    # Repo-relative path (resolved in-pod against /app by the controller). No-op for configs
    # without policy_chat_template.
    if _policy_chat_template:
        controller_cmd.extend(["--policy-chat-template", _policy_chat_template])
    controller_cmd.append("--")
    controller_cmd.extend(train_cmd)

    # Wrap in a bash bootstrap: cd to the synced workspace and set PYTHONPATH so
    # live /app + skyrl-train win over the image's baked copies. Use the absolute
    # RL venv python (set above) — independent of iris's activated venv.
    #
    # ALL MarinSkyRL code resolves from the SYNCED /app (the launcher syncs
    # PROJECT_ROOT -> /app), retiring the /app-vs-/opt/skyrl shadow split:
    #   - /app            : cloud.* (cloud/iris/*)  -> synced source
    #   - /app/skyrl-train: skyrl_train.* + examples.terminal_bench.* -> synced source
    #   - /opt/skyrl/skyrl-train: baked fallback (kept LAST so a partial /app can't
    #     ModuleNotFoundError, but /app wins for anything it carries)
    # The venv/deps still come from the /opt/skyrl editable install; only the SOURCE
    # that `import skyrl_train`/`examples.*` resolves is moved to /app. This makes the
    # /app sync the single deploy vector — everything now rides the /app sync off `main`;
    # /opt/skyrl stays at its baked HEAD purely as a last-resort import fallback.
    pythonpath = f"{APP_DIR}:{APP_DIR}/skyrl-train:{SKYRL_HOME}/skyrl-train"
    # GAP D fix: install marin-iris into the RL venv at bootstrap for the controller-
    # ingress registration/mint path. cloud.iris.ingress_utils hard-imports
    # iris.cluster.client.* / iris.rpc.*, but the gpu-rl image bakes ONLY MarinSkyRL +
    # harbor (never iris, a marin-monorepo pkg) -> `ModuleNotFoundError: No module named
    # 'iris'` in driver init. This is a lightweight live install (no ~40-min
    # kaniko rebuild): marin-iris is a pure-python wheel, installed live. Only needed in
    # controller mode (direct mode never imports iris), so gate on ingress_mode ==
    # controller -> the default direct path is byte-identical (no install, no env change).
    #   - NO [controller] extra: the CLIENT registration path (EndpointClient + rpc stubs)
    #     needs neither kubernetes<36 nor Secret-Manager (it loads grpc + connectrpc +
    #     rigging + finelog only), so skipping it avoids the biggest dep-conflict source.
    #   - No --constraint file: the lock IS the constraint (Dockerfile.gpu-rl builds the
    #     RL venv via `uv sync --frozen` off skyrl-train/uv.lock; there is no baked
    #     rl_env_constraints.txt anymore — pointing at it File-not-found'd on newer images).
    #     `uv pip install` does not upgrade already-satisfying installed deps, and iris's
    #     deps are all present with satisfied >= bounds, so uv only ADDS pure-python leaves;
    #     torch/vllm/flash_attn aren't in iris's tree. The one real downgrade vector (boto,
    #     an upper-bound pin) is separately snapshot+restored below.
    #   - GAP E#2 boto guard: the marin-iris solve DOWNGRADES the (deliberately-unpinned)
    #     botocore cluster (1.43.46 -> 1.43.0), breaking `from botocore.docs.utils import
    #     DocumentModifiedShape` (accelerate imports it transitively -> a MASKED
    #     "accelerate circular import" that killed every prior controller-ingress smoke).
    #     Snapshot the baked boto pins with `uv pip freeze` (the venv is uv-managed, NO
    #     pip module) BEFORE the install and force-restore them (--no-deps) AFTER.
    #   - Under `set -e` + a hard import of the exact registration path AND torch/vllm/
    #     flash_attn + DocumentModifiedShape asserts: a clobbered pin KILLS the job loud
    #     at bootstrap rather than dying deep in driver init.
    iris_refresh = ""
    if getattr(args, "ingress_mode", "direct") == "controller":
        ispec = getattr(args, "iris_ref", None) or DEFAULT_IRIS_VERSION
        iris_refresh = (
            f"_BOTO_BAKED=$(uv pip freeze --python {shlex.quote(RL_PYTHON)} 2>/dev/null | "
            f"grep -iE '^(botocore|boto3|s3transfer|awscli)==' | tr '\\n' ' ' || true); "
            f'echo "[rl-iris] boto baked pins: $_BOTO_BAKED"; '
            f"uv pip install --python {shlex.quote(RL_PYTHON)} {shlex.quote(ispec)}; "
            f'if [ -n "$_BOTO_BAKED" ]; then uv pip install --python '
            f"{shlex.quote(RL_PYTHON)} --no-deps $_BOTO_BAKED; fi; "
            f'{RL_PYTHON} -c "import importlib.metadata as m; '
            f"import iris.cluster.client.endpoint_client, iris.cluster.client.job_info, "
            f"iris.rpc.controller_connect, iris.cluster.types; "
            f"print('[rl-iris] marin-iris now', m.version('marin-iris'), "
            f"'(controller-ingress import OK)')\"; "
            f'{RL_PYTHON} -c "import botocore; from botocore.docs.utils import '
            f"DocumentModifiedShape; print('[rl-iris] boto cluster intact: botocore', "
            f'botocore.__version__)"; '
            f'{RL_PYTHON} -c "import torch, vllm, flash_attn, flash_attn_2_cuda; '
            f"print('[rl-iris] post-iris pins intact: torch', torch.__version__, "
            f"'vllm', vllm.__version__)\"; "
        )
    ctrl = shlex.join(controller_cmd)
    # TileLang JIT-cache warm-start shim (Fix A) — GDN/FlashQLA runs only.
    # SKYRL_GDN_FLASHQLA=1 lazily JIT-compiles the FlashQLA GatedDeltaNet TileLang
    # kernels on the first GPU forward into the node-local, ephemeral TileLang cache
    # (~71 min cold on the first r4f run, x every one of the N gang pods — kaniko is
    # CPU-only so they can't be baked into the image). This brackets the train command
    # with a per-pod, per-NODE cache sync (the bash runs once per task pod / node, and
    # TileLang's cache is node-local, so one --down warms all 8 local GPU workers):
    #   --down BEFORE the controller -> pulls the keyed warm cache (seed cache.tgz +
    #          incremental per-hash-dir objects) into TILELANG_CACHE_DIR so TileLang
    #          hash-matches and skips the cold compile. A miss is a warn+continue no-op.
    #   --up   at EXIT (bash EXIT trap; fires on normal completion AND a `set -e`/crash
    #          exit) -> uploads NEWLY-compiled hash-dirs as per-hash objects (race-free
    #          across the ~16 writers — content-addressed, no cache.tgz overwrite).
    # The shim self-gates on SKYRL_GDN_FLASHQLA and NEVER fails the job (best-effort;
    # exits 0 even on S3 error). We ALSO branch here on SKYRL_GDN_FLASHQLA so a non-GDN
    # run (e.g. 30B-coder) keeps the BYTE-IDENTICAL `exec <controller>` fast path.
    # TILELANG_CACHE_DIR is exported (defaulting to TileLang's own default) so the shim
    # and the trainer's TileLang agree on the location; a config-set value wins.
    # TILELANG_CACHE_MODEL_PATH lets the shim derive the model component of the key.
    sync_py = "cloud/iris/tilelang_cache_sync.py"
    tl_down = f"{RL_PYTHON} {sync_py} --down || true"
    tl_up = f"{RL_PYTHON} {sync_py} --up || true"
    # The controller is run as a BACKGROUND child + `wait` (not `exec`) so we can
    # (a) run --up at exit via the bash EXIT trap and (b) FORWARD SIGTERM/SIGINT to
    # the controller — preserving the old `exec` graceful-shutdown path (rank-0's Ray
    # teardown + done-marker on preemption) that a plain child would lose. `wait` is
    # interrupted by the trapped signal (rc>128); we re-`wait` to reap the child's
    # real exit code after its forwarded-TERM shutdown.
    gdn_branch = (
        f'if [ "${{SKYRL_GDN_FLASHQLA:-0}}" = "1" ] || '
        f'[ "${{SKYRL_GDN_FLASHQLA:-}}" = "true" ] || '
        f'[ "${{SKYRL_GDN_FLASHQLA:-}}" = "on" ]; then '
        f'export TILELANG_CACHE_DIR="${{TILELANG_CACHE_DIR:-/root/.tilelang/cache}}"; '
        f"export TILELANG_CACHE_MODEL_PATH={shlex.quote(args.model_path)}; "
        f"{tl_down}; "
        f"trap {shlex.quote(tl_up)} EXIT; "
        f'trap \'[ -n "$_child" ] && kill -TERM "$_child" 2>/dev/null\' TERM INT; '
        f"set +e; {ctrl} & _child=$!; "
        f'wait "$_child"; _rc=$?; '
        f'if [ $_rc -gt 128 ]; then wait "$_child" 2>/dev/null; _rc=$?; fi; '
        f"exit $_rc; "
        f"else exec {ctrl}; fi"
    )
    bash = (
        f"set -e; cd {APP_DIR}; "
        f"{iris_refresh}"
        f"export SKYRL_HOME={shlex.quote(SKYRL_HOME)}; "
        f"export PYTHONPATH={shlex.quote(pythonpath)}:${{PYTHONPATH:-}}; "
        f"export VLLM_USE_V1=1; "
        f"{gdn_branch}"
    )
    return ["bash", "-c", bash]


def main() -> int:
    parser = create_parser()
    args = parser.parse_args()

    # Resolve the cluster YAML from --cluster when not explicitly given, so
    # `--cluster cw-rno2a` targets the 512xH100 RNO2A cluster (the delphi pilot's
    # 512-node target) without a manual --cluster-config.
    if not args.cluster_config:
        args.cluster_config = _resolve_cluster_config_default(args.cluster)
    normalize(args)
    resolve_launch_defaults(args)

    # Derive the controller-ingress config from the target cluster so an agentic
    # CoreWeave launch works from --target-cluster alone (no manual --ingress-mode/
    # --ingress-host). Runs BEFORE the reachability guard, which then only sees the
    # single correct cluster-determined config.
    autoconfigure_ingress(args)

    # Fail loud (before any submit / GPU allocation) when controller-ingress would
    # produce a capability URL the Daytona sandbox cannot reach — the Exp2 blocker
    # (opencode never reaches vLLM on CoreWeave via a directly-submitted job). The
    # default direct path returns immediately (byte-identical).
    validate_controller_ingress_reachability(args)
    parent_credentials_json = prepare_federated_parent_credentials(args)

    if not args.job_name:
        args.job_name = f"rl-iris-{time.strftime('%Y%m%d-%H%M%S')}"

    # Load --secrets-env into os.environ on the launch host (so launch-host
    # hooks see it) AND collect them for injection into the task. Reuse the
    # (file overrides shell; same semantics as the OT-Agent iris launchers).
    load_secrets_env_into_os_environ(args.secrets_env)

    if _rl_config_is_agentic(args.rl_config):
        daytona_api_key = _resolve_daytona_rl_api_key()
        os.environ["DAYTONA_API_KEY"] = daytona_api_key
        print(
            "[rl-iris] Daytona: agentic run uses canonical Google Secret Manager "
            f"{DAYTONA_RL_SECRET_NAME} version {DAYTONA_RL_SECRET_VERSION}.",
            flush=True,
        )
        _purge_stale_daytona_snapshots(daytona_api_key)

    command = build_task_command(args)

    # Per-task resources: a WHOLE node (8 H100 + IB), one task per node.
    gpu_spec = f"{args.gpu_variant}x{args.gpus_per_node}"

    # Resolve the "auto" disk default to ~80% of the node's live allocatable ephemeral-storage.
    # (Whole-node-exclusive gangs have no co-tenants → reserving disk is wasted; a too-low fixed
    # default evicted long MoE steps once Ray spilled to the metered /tmp. See _resolve_default_disk.)
    if str(args.disk).strip().lower() == "auto":
        args.disk = _resolve_default_disk()

    user = os.environ.get("USER") or os.environ.get("USERNAME") or "user"
    print(f"[rl-iris] Job:        /{user}/{args.job_name}", flush=True)
    print(f"[rl-iris] Cluster:    {args.cluster}  ({args.cluster_config})", flush=True)
    print(f"[rl-iris] Image:      {args.task_image}", flush=True)
    print(
        f"[rl-iris] Topology:   {args.num_nodes} node(s) x {gpu_spec}  "
        f"(= {args.num_nodes * args.gpus_per_node} GPUs, exclusive, gang/leafgroup)",
        flush=True,
    )
    print(f"[rl-iris] Per node:   cpu={args.cpu} memory={args.memory} disk={args.disk}", flush=True)
    print(f"[rl-iris] Priority:   {args.priority}", flush=True)
    print(f"[rl-iris] RL config:  {args.rl_config}  model={args.model_path}", flush=True)
    # Surface the resolved SKYRL_* runtime-knob flag env here (before the --dry-run
    # return) so a dry-run confirms e.g. --collective-count-diag actually resolves.
    # This is display-only; main() re-derives it (idempotent, pure fn of args) below.
    _flag_env_preview = build_skyrl_flag_env(args)
    if _flag_env_preview:
        print(
            f"[rl-iris] SKYRL flag env: {', '.join(f'{k}={v}' for k, v in sorted(_flag_env_preview.items()))}",
            flush=True,
        )
    if args.num_nodes > 1:
        print(f"[rl-iris] Rendezvous: {args.rendezvous_dir}", flush=True)
    print(f"[rl-iris] Command:    {shlex.join(command)}", flush=True)

    if args.dry_run:
        print("[rl-iris] --dry-run: not submitting", flush=True)
        return 0

    # Defer heavy iris imports so --dry-run / --help stay snappy.
    #
    # NOTE: post iris PR #6652 (pydantic config parsing) + #6730 (multi-backend
    # controller) the old submit API moved. The config is now a pydantic
    # ``IrisClusterConfig`` loaded via the MODULE-LEVEL ``load_config(path)``
    # (the ``IrisConfig`` class + its ``.load()`` / ``.provider_bundle()`` /
    # ``.proto`` are gone). The provider bundle is now built by the module-level
    # ``iris.cluster.composer.provider_bundle(config)``, and ``LocalCluster``
    # moved to ``iris.cluster.local_cluster``. The job-build helpers
    # (build_resources / build_job_constraints / resolve_multinode_defaults /
    # EnvironmentSpec / Entrypoint / job_pb2) and the ``IrisClient.remote(...)`` /
    # ``client.submit(...)`` surface are UNCHANGED — see how the marin CLI itself
    # now submits in iris/cli/job.py + iris/cli/connect.py, which this mirrors.
    from iris.client import IrisClient
    from iris.cluster.config import load_config
    from iris.cluster.composer import provider_bundle
    from iris.cluster.local_cluster import LocalCluster
    from iris.cluster.types import EnvironmentSpec, Entrypoint
    from iris.cli.job import build_resources, build_job_constraints, resolve_multinode_defaults
    from iris.rpc import job_pb2

    # Per-task resources: whole node, all GPUs (no co-tenant → exclusive).
    resources = build_resources(None, gpu_spec, cpu=args.cpu, memory=args.memory, disk=args.disk)

    # Multi-node gang: replicas=num_nodes; for GPUs with replicas>1 this returns
    # CoschedulingConfig(group_by="leafgroup") — co-schedule all nodes on one IB
    # leaf fabric, atomically (Kueue gang admission on cw-us-east-02a).
    replicas, coscheduling = resolve_multinode_defaults(None, args.gpu_variant, args.num_nodes)

    resources_proto = resources.to_proto()
    # --target-cluster (federated submission) appends a `cluster EQ <peer>` constraint
    # so the marin meta-scheduler DELEGATES the whole job to the peer child (see the
    # submission block below). None on the default direct path (byte-identical).
    constraints = build_job_constraints(
        resources_proto=resources_proto,
        tpu_variants=[],
        replicas=replicas,
        regions=None,
        zone=None,
        preemptible=args.preemptible,
        target_cluster=args.target_cluster,
    )

    priority_band = job_pb2.PriorityBand.Value(priority_band_enum_name(args.priority))

    # Env: secrets file values + the standard RL/iris-serve signals. iris injects
    # IRIS_TASK_ID / IRIS_NUM_TASKS / IRIS_ADVERTISE_HOST per task automatically.
    env_vars: dict[str, str] = {}
    # MarinSkyRL runtime-knob flags (deslop stage 3) -> SKYRL_* env vars. Seeded
    # FIRST (below the config extra_env) so a config's explicit extra_env value still
    # OVERRIDES a flag; an all-defaults launch contributes {} (byte-identical).
    flag_env = build_skyrl_flag_env(args)
    if flag_env:
        env_vars.update(flag_env)
        print(f"[rl-iris] SKYRL flag env: {', '.join(f'{k}={v}' for k, v in sorted(flag_env.items()))}", flush=True)
    # Forward the RL config YAML's top-level `extra_env:` block (the Iris analog of
    # the SLURM container.extra_env exports — see load_config_extra_env). Overlaid
    # ON TOP of the flag env so an explicit config value wins; the launcher's own
    # signals (rendezvous/secrets, below) then win over both on any collision.
    config_extra_env = load_config_extra_env(args.rl_config)
    if config_extra_env:
        env_vars.update(config_extra_env)
        print(f"[rl-iris] Config extra_env: {', '.join(sorted(config_extra_env))}", flush=True)
    # ── Per-cluster infra-env DEFAULTS (fill-gap belt for cluster-specific footguns) ──────────
    # Some clusters need a specific network/NCCL interface that a cluster-AGNOSTIC RL config
    # won't (and shouldn't) carry. Fill it in here, keyed on --target-cluster, ONLY if neither
    # the flag env nor the config's extra_env already set it (lowest precedence — an explicit
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
    # fsspec rendezvous in start_rl_iris_controller.py uses default credential
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
    # WANDB routing default: the iris RL configs log to wandb (trainer.logger: wandb;
    # CoreWeave has egress). SkyRL's wandb.init passes project= but NOT entity=
    # (MarinSkyRL tracking.py), so without WANDB_ENTITY the run silently lands in the
    # API key's DEFAULT entity (e.g. nyu-dice-lab), not the team org. Default both to
    # the OT-Agent team here so every run lands in
    # dogml/OpenThoughts-Agent; an explicitly-set launch-host WANDB_ENTITY/PROJECT wins.
    os.environ.setdefault("WANDB_ENTITY", "dogml")
    os.environ.setdefault("WANDB_PROJECT", "OpenThoughts-Agent")
    for k in (
        "HF_TOKEN",
        "WANDB_API_KEY",
        "WANDB_ENTITY",
        "WANDB_PROJECT",
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

    # Federated controller-ingress pod plumbing (opencode-RL literal capture): the in-pod
    # worker mints the capability token at the PARENT (marin/iris.oa.dev) for the mirrored
    # endpoint, which needs (a) the parent cluster config path and (b) IAP credentials to
    # authenticate to iris.oa.dev. We forward the config path + any launch-host IAP cred
    # env so the in-pod _ParentControllerClient can re-mint the IAP OIDC token.
    #
    # The parent config file is not baked into the gpu-rl image, so forward its contents
    # for in-pod materialization. Federated controller ingress always forwards the cached
    # Marin login record after prepare_federated_parent_credentials() has minted a token
    # from it locally. Direct submission (no --target-cluster) forwards none of this.
    if getattr(args, "target_cluster", None) and getattr(args, "ingress_mode", "direct") == "controller":
        from cloud.iris.ingress_utils import (
            PARENT_CONTROLLER_CONFIG_ENV,
            PARENT_CONTROLLER_CONFIG_YAML_ENV,
            PARENT_CREDENTIALS_JSON_ENV,
        )

        parent_cfg = (
            getattr(args, "parent_controller_config_in_pod", None)
            or args.parent_cluster_config
            or _resolve_parent_cluster_config(args.cluster_config)
        )
        if parent_cfg:
            env_vars[PARENT_CONTROLLER_CONFIG_ENV] = parent_cfg
            # marin.yaml is not baked into the gpu-rl image and is not part of the
            # synced workspace, so the path above won't resolve in-pod. Forward the
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
        # A CoreWeave pod has no cached `iris login` and no Marin-allowlisted ambient
        # service account. Forwarding this validated record is therefore mandatory for
        # the in-pod parent mint; it remains a secret in the submitted job environment.
        if parent_credentials_json is None:
            raise AssertionError("federated controller ingress must have validated parent credentials")
        env_vars[PARENT_CREDENTIALS_JSON_ENV] = parent_credentials_json

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
            yield IrisClient.remote(controller_url, workspace=PROJECT_ROOT)

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

        client_cm = open_iris_client(config_file=Path(submit_cluster_config), workspace=PROJECT_ROOT)
    else:
        client_cm = _direct_client()

    with client_cm as client:
        entrypoint = Entrypoint.from_command(*command)
        job = client.submit(
            entrypoint=entrypoint,
            name=args.job_name,
            resources=resources,
            # setup_scripts=[] means "no setup at all; the image is used as-is"
            # (iris EnvironmentSpec docstring). Leaving it unset makes iris build its
            # default script, which runs `uv sync` over the synced workspace and creates
            # a venv at $IRIS_WORKDIR/.venv. That venv is a plain PyPI resolve -- no CUDA
            # stack, no ray, no boto3 -- and iris puts it FIRST on $PATH, so a bare
            # `python` or `ray` in a task pod resolves to it instead of the image's real
            # env at /opt/openthoughts/envs/rl. Everything here already uses absolute
            # interpreter paths to route around that; the sync itself is pure cost on a
            # fully baked image, and the shadow venv has repeatedly cost debugging time.
            #
            # REQUIRES an image that bakes cloudpickle/py-spy/memray, since [] also skips
            # iris's own runtime setup that would otherwise install them. Landed in the
            # Dockerfile before this flip; py-spy is what a wedge diagnosis attaches with.
            environment=EnvironmentSpec(env_vars=env_vars, extras=[], setup_scripts=[]),
            constraints=constraints or None,
            coscheduling=coscheduling,
            replicas=replicas,
            max_retries_failure=args.max_retries,
            task_image=args.task_image,
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
            return 0
        print(
            f"[rl-iris] Now streaming logs for {full_job_id}. This process runs until the job ends.\n"
            "[rl-iris] Ctrl-C or SIGINT TERMINATES the job. It does not detach from it.\n"
            "[rl-iris] Use --no-wait to submit and return instead.\n"
            "[rl-iris] To stop a backgrounded launcher and keep the job alive, use kill or kill -9. "
            "Never use kill -2.",
            file=sys.stderr,
            flush=True,
        )
        try:
            status = job.wait(stream_logs=True, timeout=float("inf"))
            exit_code = 0 if status.state == job_pb2.JOB_STATE_SUCCEEDED else 1
        except KeyboardInterrupt:
            print(f"[rl-iris] Terminating job {full_job_id}...", file=sys.stderr, flush=True)
            client.terminate_job(job.job_id)
            exit_code = 130
        print(f"[rl-iris] Job exit: {exit_code}", flush=True)
        return exit_code


def _seconds_to_duration(secs: int):
    from rigging.timing import Duration

    return Duration.from_seconds(secs)


if __name__ == "__main__":
    sys.exit(main())
