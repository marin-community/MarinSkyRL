#!/usr/bin/env python3
"""Sync and summarize CoreWeave Iris and explicitly selected Jupiter RL jobs.

The monitor is deliberately read-only.  Each Iris job gets one stable local
directory, keyed by its full Iris job id, so repeated sweeps and Iris task
retries refresh the same artifacts instead of making timestamped copies.  It
captures the complete finelog plus complete pod/Ray/vLLM logs, then mirrors
the 500 most recently modified Harbor ``trace_jobs`` across the active RL fleet
by default. The recent trace selection is based on object-store ``LastModified``
metadata, never trace names. Use ``--trace-sync-limit 0`` for a deliberately
full trace sync. The sync still skips non-log objects larger than the configured
size bound; that rule avoids repeatedly downloading giant rollout payloads while
preserving any diagnostic log regardless of size.

By default the scope is the current lab user's active RL jobs on every
configured CoreWeave GPU cluster. Jupiter runs are opt-in because their
artifacts live on GPFS: pass the canonical launcher experiment path with
``--jupiter-run SLURM_JOB_ID=/absolute/experiment/path``. The Jupiter sync only
touches known subtrees below that path and performs one shallow trace listing.

For a Jupiter-only status sweep:

``python scripts/iris/watch_coreweave_rl.py --jupiter-only \
  --jupiter-run 1234567=/absolute/path/to/experiments/my-run --status-only``
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

from botocore.exceptions import ClientError


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from infra.artifact_files import LOG_SUFFIXES  # noqa: E402
from infra.rl_metrics import (  # noqa: E402
    ENTROPY_KEYS,
    GRAD_NORM_KEYS,
    POLICY_LOG_RATIO_ABS_MAX_KEYS,
    POLICY_LOG_RATIO_ABS_MEAN_KEYS,
    POLICY_LOG_RATIO_ABS_P99_KEYS,
    POLICY_LOSS_KEYS,
    REWARD_KEYS,
    TIS_EXACT_MATCH_KEYS,
    TIS_IMPORTANCE_RATIO_MEAN_KEYS,
    TIS_LOG_RATIO_ABS_MEAN_KEYS,
    metric_value,
    parse_training_metrics_result,
    parse_tis_enabled,
    training_metrics_parse_error,
)
from scripts.iris.jupiter_rl_artifacts import (  # noqa: E402
    JupiterJobStatus,
    JupiterRunSpec,
    parse_jupiter_run_spec,
    query_jupiter_job_status,
    sync_jupiter_artifacts,
    validate_jupiter_host,
)
from scripts.iris.coreweave_clusters import CLUSTERS as COREWEAVE_CLUSTERS  # noqa: E402
from scripts.iris.coreweave_ops import (  # noqa: E402
    NAMESPACE,
    iter_objects,
    kubectl_base,
    object_store_client,
    pod_matches_job,
    ray_log_inventory,
    resolve_container_python,
    safe_relative_path,
    save_ray_logs,
    split_s3_uri,
)
from scripts.iris.iris_ops import (  # noqa: E402
    DEFAULT_BUNDLE_ROOT,
    ERROR_DETAIL_CHARS,
    MonitorError,
    StyledCell,
    TERMINAL_STATES,
    box_table,
    filter_records,
    format_duration,
    job_bundle,
    parse_regex_filters,
    run_iris_command,
    write_bundle_manifest,
    write_error_report,
)


DEFAULT_USER = "benjaminfeuer"
DEFAULT_MAX_NON_LOG_BYTES = 100 * 1024 * 1024
DEFAULT_TRACE_SYNC_LIMIT = 500
DEFAULT_JUPITER_TRACE_SYNC_LIMIT = 20
DEFAULT_JUPITER_HOST = "Jupiter"
CURRENT_FINELOG_MAX_LINES = 600_000
COMPLETE_FINELOG_MAX_LINES = 10_000_000
FinelogScope = Literal["current", "complete"]
SyncScope = Literal["status", "terminal", "full"]
ACTIVE_STATES = {1: "pending", 2: "building", 3: "running"}
STATE_NAMES = {
    **ACTIVE_STATES,
    4: "succeeded",
    5: "failed",
    6: "killed",
    7: "worker_failed",
    8: "unschedulable",
}
# Display state for an active job whose task has not been placed on a node yet.
AWAITING_PLACEMENT = "awaiting placement"
# Iris active states that precede running; a root job or task in one of these
# has not been placed on a node yet.
PRE_RUNNING_STATES = frozenset({"pending", "building"})
RL_ENTRYPOINT_MARKERS = ("task_runtime.py", "cloud.iris.training_driver")
TRIALS_URI_PATTERN = re.compile(
    r"(?:terminal_bench_config\.trials_dir=|--trials[_-]dir(?:=|\s+))['\"]?"
    r"(?P<uri>s3://[^\s'\"\\]+)"
)
TRIALS_DIR_OPTIONS = frozenset({"--trials-dir", "--trials_dir"})
TRAIN_DATA_PATTERN = re.compile(
    r"--train[_-]data(?:=|\s+)(?:'(?P<single>\[[^']+\])'|\"(?P<double>\[[^\"]+\])\"|(?P<bare>\[[^\s]+\]))"
)
PROGRESS_PATTERN = re.compile(r"Training Step Progress:\s*(\d+)\s*/\s*(\d+)")
ERROR_PATTERNS = (
    re.compile(r"CUDA out of memory", re.IGNORECASE),
    re.compile(r"(?:RayTaskError|ActorDiedError|WorkerCrashedError)"),
    re.compile(r"Traceback \(most recent call last\)"),
    re.compile(r"Train loop failed", re.IGNORECASE),
)
IRIS_ATTEMPT_BOUNDARY_PATTERN = re.compile(r"task=\S+/0 \| \[iris setup\] step 1/2")
NON_ERROR_LOG_LEVEL_PATTERN = re.compile(r"\b(?:TRACE|DEBUG|INFO|WARNING)\b")
MISSING_OBJECT_ERROR_CODES = {"404", "NoSuchKey", "NoSuchObject", "NotFound"}


@dataclass(frozen=True)
class Cluster:
    name: str
    kubeconfig: Path
    context: str | None


@dataclass(frozen=True)
class MonitorSettings:
    bundle_root: Path
    status_only: bool
    max_non_log_bytes: int
    iris_trace_sync_limit: int
    jupiter_host: str
    jupiter_trace_sync_limit: int


@dataclass(frozen=True)
class RlJob:
    cluster: Cluster
    job_id: str
    state: str
    submitted_at_ms: int
    entrypoint: str
    dataset: str = "—"
    finished_at_ms: int | None = None
    # Highest active task state from the Iris tasks table (pending/building/
    # running); None when no task is active, e.g. a terminal job.
    task_state: str | None = None

    @property
    def short_name(self) -> str:
        return self.job_id.rstrip("/").rsplit("/", 1)[-1]

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    @property
    def bundle_job_id(self) -> str:
        return self.job_id

    @property
    def trials_uri(self) -> str | None:
        raise NotImplementedError

    @property
    def uses_iris_trace_budget(self) -> bool:
        raise NotImplementedError

    def trace_sync_limit(self, settings: MonitorSettings) -> int:
        raise NotImplementedError

    def sync_artifacts(
        self,
        settings: MonitorSettings,
        progress: ProgressReporter,
        *,
        scope: SyncScope,
    ) -> tuple[ArtifactResult, Path]:
        raise NotImplementedError


@dataclass(frozen=True)
class IrisRlJob(RlJob):
    """An Iris-backed RL job whose artifacts live in Kubernetes and object storage."""

    @property
    def trials_uri(self) -> str | None:
        return iris_trials_uri(self)

    @property
    def uses_iris_trace_budget(self) -> bool:
        return not self.is_terminal

    def trace_sync_limit(self, settings: MonitorSettings) -> int:
        return settings.iris_trace_sync_limit

    def sync_artifacts(
        self,
        settings: MonitorSettings,
        progress: ProgressReporter,
        *,
        scope: SyncScope,
    ) -> tuple[ArtifactResult, Path]:
        return sync_iris_job(self, settings.bundle_root, scope=scope, progress=progress)


@dataclass(frozen=True, kw_only=True)
class JupiterRlJob(RlJob):
    """An explicit Jupiter Slurm job and its canonical GPFS experiment root."""

    experiment_dir: str
    launcher_job_name: str

    @property
    def short_name(self) -> str:
        return self.launcher_job_name

    @property
    def bundle_job_id(self) -> str:
        return f"/slurm/{self.job_id}"

    @property
    def trials_uri(self) -> str | None:
        return None

    @property
    def uses_iris_trace_budget(self) -> bool:
        return False

    def trace_sync_limit(self, settings: MonitorSettings) -> int:
        return settings.jupiter_trace_sync_limit

    def sync_artifacts(
        self,
        settings: MonitorSettings,
        progress: ProgressReporter,
        *,
        scope: SyncScope,
    ) -> tuple[ArtifactResult, Path]:
        return sync_jupiter_job(
            self,
            settings.bundle_root,
            scope=scope,
            jupiter_host=settings.jupiter_host,
            jupiter_trace_sync_limit=settings.jupiter_trace_sync_limit,
            max_non_log_bytes=settings.max_non_log_bytes,
            progress=progress,
        )


MonitoredJob = IrisRlJob | JupiterRlJob


@dataclass(frozen=True)
class ArtifactResult:
    finelog: str
    pod_logs: str
    ray_logs: str
    traces: str
    trace_started: int | None
    trace_completed: int | None
    errors: tuple[str, ...]
    slurm_logs: str = "not applicable"
    trace_selected: int | None = None


@dataclass(frozen=True)
class SyncedJob:
    job: MonitoredJob
    artifacts: ArtifactResult
    directory: Path


@dataclass(frozen=True)
class JobReportValue:
    cluster: str
    directory: str
    artifacts: ArtifactResult


@dataclass(frozen=True)
class JobReportEntry:
    key: str
    value: JobReportValue
    row: list[object]
    errors: tuple[MonitorError, ...]


@dataclass(frozen=True)
class ReportData:
    rows: list[list[object]]
    jobs: dict[str, JobReportValue]
    errors: list[MonitorError]


@dataclass(frozen=True)
class TraceJobObjects:
    """One remote trace directory and its latest object modification time."""

    name: str
    last_modified: datetime
    objects: tuple[dict[str, Any], ...]
    completed: bool


@dataclass(frozen=True)
class TraceInventory:
    """Remote trace objects for one RL job, collected before fleet selection."""

    job: IrisRlJob
    bucket: str
    root_prefix: str
    client: Any
    traces: tuple[TraceJobObjects, ...]
    available: int
    completed: int


@dataclass
class ProgressReporter:
    """Emit phase-level timing to stderr without contaminating the status table."""

    enabled: bool = True
    started_at: float = field(default_factory=time.monotonic)

    def phase(self, message: str) -> None:
        if not self.enabled:
            return
        elapsed_seconds = int(time.monotonic() - self.started_at)
        elapsed = f"{elapsed_seconds // 60:02d}:{elapsed_seconds % 60:02d}"
        print(f"[rl-watch +{elapsed}] {message}", file=sys.stderr, flush=True)


CLUSTERS = tuple(Cluster(name, config.kubeconfig, config.context) for name, config in COREWEAVE_CLUSTERS.items())
JUPITER_CLUSTER = Cluster("jsc-jupiter", Path(), None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle-root",
        type=Path,
        default=DEFAULT_BUNDLE_ROOT,
        help="Root for canonical local RL evidence bundles and reports.",
    )
    parser.add_argument(
        "--hours",
        type=float,
        default=24.0,
        help="Only include jobs submitted within this many hours; 0 means all history (default: 24).",
    )
    parser.add_argument(
        "--user",
        default=DEFAULT_USER,
        help=f"Iris user to monitor and whose per-cluster budget is reported (default: {DEFAULT_USER}).",
    )
    parser.add_argument("--all-users", action="store_true", help="Discover active RL jobs for every user.")
    parser.add_argument(
        "--jupiter-run",
        action="append",
        default=[],
        type=parse_jupiter_run_spec,
        metavar="SLURM_JOB_ID=/ABSOLUTE/EXPERIMENT/PATH",
        help="Also monitor one explicit Jupiter RL run. Repeat for multiple runs.",
    )
    parser.add_argument(
        "--jupiter-host",
        default=DEFAULT_JUPITER_HOST,
        help=f"SSH host or configured alias used for Jupiter access (default: {DEFAULT_JUPITER_HOST}).",
    )
    parser.add_argument(
        "--jupiter-only",
        action="store_true",
        help="Skip CoreWeave Iris discovery and monitor only the explicit --jupiter-run entries.",
    )
    parser.add_argument(
        "--max-non-log-bytes",
        type=int,
        default=DEFAULT_MAX_NON_LOG_BYTES,
        help="Skip non-log trace objects larger than this many bytes (default: 100 MiB; 0 disables).",
    )
    parser.add_argument(
        "--trace-sync-limit",
        type=int,
        default=DEFAULT_TRACE_SYNC_LIMIT,
        help=(
            "Sync only this many most recently modified trace directories across all discovered RL jobs "
            "(default: 500; 0 syncs every remote trace)."
        ),
    )
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="Fetch only the current finelog tail for status; skip pod, Ray, and trace artifacts.",
    )
    parser.add_argument(
        "--jupiter-trace-sync-limit",
        type=int,
        default=DEFAULT_JUPITER_TRACE_SYNC_LIMIT,
        help=(
            "Sync this many newest trace directories per explicit Jupiter run "
            "(default: 20; 0 deliberately syncs all traces in that run)."
        ),
    )
    parser.add_argument(
        "--filter",
        action="append",
        default=[],
        metavar="KEY=REGEX",
        help=(
            "Keep jobs matching every case-insensitive regex filter. Available keys: "
            "cluster, job, name, dataset, type, state, submitted, duration."
        ),
    )
    parser.add_argument(
        "--quiet-progress",
        action="store_true",
        help="Suppress stderr phase/timing updates while artifacts are collected.",
    )
    return parser.parse_args()


def run_iris(cluster: Cluster, arguments: list[str], *, timeout: int = 300) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["KUBECONFIG"] = str(cluster.kubeconfig)
    return run_iris_command(
        arguments,
        cluster=cluster.name,
        environment=environment,
        timeout=timeout,
    )


def command_error_message(result: subprocess.CompletedProcess[str], *, tail: int = ERROR_DETAIL_CHARS) -> str:
    """Flatten a failed command's output into one tail-trimmed line."""
    return (result.stderr or result.stdout).strip().replace("\n", " ")[-tail:]


def entrypoint_text(raw: str) -> str:
    try:
        return json.dumps(json.loads(raw))
    except json.JSONDecodeError:
        return raw


def command_strings(entrypoint: str) -> list[str]:
    """Return all string leaves from an Iris entrypoint JSON payload."""
    try:
        value = json.loads(entrypoint)
    except json.JSONDecodeError:
        return [entrypoint]

    strings: list[str] = []

    def visit(item: Any) -> None:
        if isinstance(item, str):
            strings.append(item)
        elif isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return strings


def dataset_from_entrypoint(entrypoint: str) -> str:
    """Extract and deduplicate the submitted dataset list without guessing from a config."""
    datasets: list[str] = []
    for command in command_strings(entrypoint):
        for match in TRAIN_DATA_PATTERN.finditer(command):
            raw_dataset_list = next(value for value in match.groupdict().values() if value is not None)
            try:
                values = json.loads(raw_dataset_list)
            except json.JSONDecodeError:
                continue
            if isinstance(values, list):
                datasets.extend(str(value) for value in values)
    return ", ".join(dict.fromkeys(datasets)) or "—"


def csv_rows(output: str) -> list[dict[str, str]]:
    """Parse Iris CSV after its informational controller/tunnel preamble."""
    lines = output.splitlines()
    header_index = next((index for index, line in enumerate(lines) if line.startswith("job_id,")), None)
    if header_index is None:
        raise ValueError("Iris query returned no CSV job_id header")
    return list(csv.DictReader(lines[header_index:]))


def discover_rl_jobs(
    cluster: Cluster,
    user: str | None,
    *,
    submitted_since_ms: int | None = None,
) -> tuple[list[IrisRlJob], list[str]]:
    where_user = "" if user is None else f" AND j.job_id LIKE '/{user}/%'"
    where_submission = "" if submitted_since_ms is None else f" AND j.submitted_at_ms >= {submitted_since_ms}"
    # Iris reports the root RL job as running (state 3) as soon as it is
    # accepted, which can be long before any workload task is placed on a
    # node. Resolve the highest active task state from ACTIVE_STATES so the
    # codes stay in one place; the CASE checks them high-to-low.
    task_state_case = (
        "CASE "
        + " ".join(
            f"WHEN EXISTS (SELECT 1 FROM tasks t WHERE t.job_id=j.job_id AND t.state={code}) THEN {code} "
            for code in sorted(ACTIVE_STATES, reverse=True)
        )
        + "ELSE NULL END AS task_state "
    )
    sql = (
        "SELECT j.job_id, j.state, j.submitted_at_ms, j.finished_at_ms, jc.entrypoint_json, "
        f"{task_state_case}"
        "FROM jobs j JOIN job_config jc ON j.job_id=jc.job_id "
        "WHERE ("
        f"j.state IN ({','.join(str(state) for state in sorted(ACTIVE_STATES))}) "
        "OR j.state IN (4,5)"
        f"){where_user}{where_submission} "
        "ORDER BY j.submitted_at_ms DESC"
    )
    result = run_iris(cluster, ["query", sql, "-f", "csv"])
    if result.returncode:
        return [], [f"{cluster.name}: discovery failed: {command_error_message(result)}"]

    jobs: list[IrisRlJob] = []
    try:
        rows = csv_rows(result.stdout)
    except ValueError as error:
        return [], [f"{cluster.name}: discovery failed: {error}"]
    for row in rows:
        entrypoint = entrypoint_text(row.get("entrypoint_json", ""))
        if not any(marker in entrypoint for marker in RL_ENTRYPOINT_MARKERS):
            continue
        try:
            state_code = int(row["state"])
            submitted_at_ms = int(row["submitted_at_ms"])
        except (KeyError, ValueError):
            continue
        task_state_value = row.get("task_state")
        jobs.append(
            IrisRlJob(
                cluster=cluster,
                job_id=row["job_id"],
                state=STATE_NAMES.get(state_code, f"state-{state_code}"),
                submitted_at_ms=submitted_at_ms,
                entrypoint=entrypoint,
                dataset=dataset_from_entrypoint(entrypoint),
                finished_at_ms=int(row["finished_at_ms"]) if row.get("finished_at_ms") else None,
                task_state=(
                    STATE_NAMES.get(int(task_state_value), f"state-{task_state_value}") if task_state_value else None
                ),
            )
        )
    return jobs, []


@dataclass(frozen=True)
class UserBudget:
    """One user's Iris budget on one cluster: consumed (spent) vs allotted (limit)."""

    user: str
    cluster: str
    spent: int | None
    limit: int | None
    max_band: str | None
    note: str | None = None


def _budget_field(line: str, key: str) -> str | None:
    """Return the trimmed value after ``key:`` on a budget output line, else None."""
    stripped = line.strip()
    prefix = f"{key}:"
    return stripped.removeprefix(prefix).strip() if stripped.startswith(prefix) else None


def fetch_user_budget(cluster: Cluster, user: str) -> UserBudget:
    """Query one cluster controller for the user's current budget spend."""
    result = run_iris(cluster, ["user", "budget", "get", user])
    if result.returncode:
        message = command_error_message(result)
        note = "no budget set" if "No budget found" in message else f"query failed: {message}"
        return UserBudget(user, cluster.name, None, None, None, note)

    spent: int | None = None
    limit: int | None = None
    max_band: str | None = None
    for line in result.stdout.splitlines():
        if (value := _budget_field(line, "Spent")) is not None:
            spent = int(value) if value else None
        elif (value := _budget_field(line, "Limit")) is not None:
            limit = int(value) if value else None
        elif (value := _budget_field(line, "Max band")) is not None:
            max_band = value or None
    return UserBudget(user, cluster.name, spent, limit, max_band)


def fetch_user_budgets(user: str, progress: ProgressReporter) -> list[UserBudget]:
    """Fetch the user's budget across every configured CoreWeave cluster."""
    progress.phase(f"fetching Iris budget for {user} across {len(CLUSTERS)} cluster(s)")
    return [fetch_user_budget(cluster, user) for cluster in CLUSTERS]


def budget_line(budget: UserBudget) -> str:
    """Render one cluster's budget as a compact consumed-vs-allotted line."""
    if budget.note:
        return f"{budget.cluster}  {budget.note}"
    spent = f"{budget.spent:,}" if budget.spent is not None else "—"
    limit_value = f"{budget.limit:,}" if budget.limit is not None else "—"
    pct = f" ({budget.spent / budget.limit:.0%})" if budget.limit and budget.spent is not None else ""
    return f"{budget.cluster}  spent={spent} / limit={limit_value}{pct}  band={budget.max_band or '—'}"


def render_budget_section(user: str, budgets: list[UserBudget]) -> str:
    """Render the per-cluster Iris budget block for the status report."""
    lines = [f"## Iris budget — user={user}"]
    lines.extend(budget_line(budget) for budget in budgets)
    return "\n".join(lines)


def job_directory(bundle_root: Path, job: MonitoredJob) -> Path:
    """Return the shared canonical evidence directory for this RL job."""
    return job_bundle(bundle_root, job.cluster.name, job.bundle_job_id).directory


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def fetch_finelog(job: IrisRlJob, destination: Path, *, scope: FinelogScope = "complete") -> tuple[str, str | None]:
    log_args = (
        ["--max-lines", str(CURRENT_FINELOG_MAX_LINES), "--tail"]
        if scope == "current"
        else ["--max-lines", str(COMPLETE_FINELOG_MAX_LINES), "--no-tail"]
    )
    result = run_iris(
        job.cluster,
        ["job", "logs", job.job_id, *log_args],
        timeout=900,
    )
    stderr_path = destination / "finelog.stderr"
    stderr_path.write_text(result.stderr)
    if result.returncode:
        return "unavailable", f"finelog: {command_error_message(result, tail=180)}"
    (destination / "finelog.log").write_text(result.stdout)
    return f"{len(result.stdout.splitlines()):,} lines", None


def job_pods(job: IrisRlJob) -> list[tuple[str, str]]:
    base = kubectl_base(COREWEAVE_CLUSTERS[job.cluster.name], SimpleNamespace(kubeconfig=None, kube_context=None))
    result = subprocess.run(
        [*base, "-n", NAMESPACE, "get", "pods", "-o", "json"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode:
        raise RuntimeError(command_error_message(result))
    return sorted(
        (
            item["metadata"]["name"],
            item.get("status", {}).get("phase", "Unknown"),
        )
        for item in json.loads(result.stdout).get("items", [])
        if pod_matches_job(item, job.job_id)
    )


def fetch_complete_pod_log(base: list[str], pod: str, destination: Path) -> None:
    result = subprocess.run(
        [*base, "-n", NAMESPACE, "logs", pod, "-c", "task", "--tail=-1"],
        capture_output=True,
        text=True,
        timeout=900,
    )
    destination.write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(command_error_message(result))


def fetch_complete_ray_logs(base: list[str], pod: str, destination: Path) -> int:
    runtime_python = resolve_container_python(base, pod, "task")
    inventory = ray_log_inventory(base, pod, "task", patterns=None, python_executable=runtime_python)
    if not inventory:
        return 0
    try:
        saved, skipped = save_ray_logs(
            base,
            pod,
            "task",
            inventory,
            sys.maxsize,
            destination,
            incremental=True,
            python_executable=runtime_python,
        )
    except RuntimeError:
        # Ray rotates/removes worker logs while a live pod is writing. Rebuild
        # the inventory once so a stale path cannot abort the whole fleet scan.
        inventory = ray_log_inventory(base, pod, "task", patterns=None, python_executable=runtime_python)
        if not inventory:
            return 0
        saved, skipped = save_ray_logs(
            base,
            pod,
            "task",
            inventory,
            sys.maxsize,
            destination,
            incremental=True,
            python_executable=runtime_python,
        )
    if skipped:
        raise AssertionError("A maximum-size sync should not skip Ray/vLLM logs.")
    return len(saved)


def sync_pod_and_ray_logs(
    job: IrisRlJob,
    destination: Path,
    *,
    progress: ProgressReporter | None = None,
) -> tuple[str, str, list[str]]:
    """Capture all current pod stdout plus all Ray/vLLM logs without a size cap."""
    errors: list[str] = []
    try:
        pods = job_pods(job)
    except (RuntimeError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        return "unavailable", "unavailable", [f"pod discovery: {error}"]
    if not pods:
        return "no pod yet", "no pod yet", []
    running_pods = [pod for pod, phase in pods if phase == "Running"]
    if not running_pods:
        phases = ", ".join(sorted(phase for _, phase in pods))
        return f"{len(pods)} pod(s): {phases}", "awaiting host", []

    base = kubectl_base(COREWEAVE_CLUSTERS[job.cluster.name], SimpleNamespace(kubeconfig=None, kube_context=None))
    pod_dir = destination / "pod_logs"
    ray_dir = destination / "ray_vllm_logs"
    pod_dir.mkdir(exist_ok=True)
    ray_dir.mkdir(exist_ok=True)
    ray_files = 0
    for index, pod in enumerate(running_pods, start=1):
        try:
            if progress:
                progress.phase(f"pod stdout {index}/{len(running_pods)} {job.short_name}/{pod}")
            fetch_complete_pod_log(base, pod, pod_dir / f"{pod}.log")
        except (RuntimeError, subprocess.SubprocessError) as error:
            errors.append(f"{pod} stdout: {error}")
        try:
            if progress:
                progress.phase(f"Ray/vLLM logs {index}/{len(running_pods)} {job.short_name}/{pod}")
            pod_ray_dir = ray_dir / pod
            pod_ray_dir.mkdir(exist_ok=True)
            ray_files += fetch_complete_ray_logs(base, pod, pod_ray_dir)
        except (RuntimeError, subprocess.SubprocessError, json.JSONDecodeError) as error:
            errors.append(f"{pod} Ray/vLLM: {error}")
    return f"{len(pods)} pod(s), {len(running_pods)} Running", f"{ray_files:,} files", errors


def trials_uri_from_entrypoint(entrypoint: str) -> str | None:
    """Return the object-store trials directory submitted to Iris."""
    arguments = command_strings(entrypoint)
    for index, argument in enumerate(arguments):
        match = TRIALS_URI_PATTERN.search(argument)
        if match:
            return match.group("uri").rstrip("/")
        if argument in TRIALS_DIR_OPTIONS and index + 1 < len(arguments):
            candidate = arguments[index + 1].strip("'\"").rstrip("/")
            if candidate.startswith("s3://"):
                return candidate
    return None


def iris_trials_uri(job: IrisRlJob) -> str:
    submitted_uri = trials_uri_from_entrypoint(job.entrypoint)
    if submitted_uri is not None:
        return submitted_uri
    # Older jobs may omit the submitted value. Preserve their historical path
    # without applying it to jobs that record the lifecycle-managed URI.
    return f"s3://marin-us-east-02a/iris/{job.short_name}/trace_jobs"


def is_log_object(relative_path: str) -> bool:
    path = relative_path.lower()
    filename = path.rsplit("/", 1)[-1]
    return any(suffix in filename for suffix in LOG_SUFFIXES) or "/logs/" in path or path.startswith("logs/")


def is_missing_object_error(error: ClientError) -> bool:
    """Return whether a listed object disappeared before it could be downloaded."""
    error_details = error.response.get("Error", {})
    return str(error_details.get("Code")) in MISSING_OBJECT_ERROR_CODES


def recent_trace_jobs(
    objects: list[dict[str, Any]], root_prefix: str, trace_sync_limit: int
) -> tuple[list[TraceJobObjects], int, int]:
    """Return trace directories ordered by latest remote object modification.

    Object-store ``LastModified`` is the only ordering source. A trace is
    complete when any object in its directory is ``result.json``; those counts
    cover the complete remote prefix even when a recent subset is downloaded.
    """
    traces: dict[str, list[dict[str, Any]]] = {}
    for item in objects:
        relative = item["Key"].removeprefix(root_prefix)
        if relative:
            traces.setdefault(relative.split("/", 1)[0], []).append(item)

    trace_jobs: list[TraceJobObjects] = []
    completed = 0
    for name, trace_objects in traces.items():
        latest_modified: datetime | None = None
        completed_trace = False
        for item in trace_objects:
            modified = item.get("LastModified")
            if not isinstance(modified, datetime):
                raise ValueError(f"Object {item['Key']!r} is missing LastModified metadata")
            if latest_modified is None or modified > latest_modified:
                latest_modified = modified
            relative = item["Key"].removeprefix(root_prefix)
            completed_trace = completed_trace or relative.endswith("/result.json")
        assert latest_modified is not None
        completed += completed_trace
        trace_jobs.append(TraceJobObjects(name, latest_modified, tuple(trace_objects), completed_trace))

    trace_jobs.sort(key=lambda trace: (trace.last_modified, trace.name), reverse=True)
    selected = trace_jobs if trace_sync_limit == 0 else trace_jobs[:trace_sync_limit]
    return selected, len(trace_jobs), completed


def trace_selection_manifest(
    selected: list[TraceJobObjects],
    inventory: TraceInventory,
    trace_sync_limit: int,
    fleet_available: int,
    fleet_selected: int,
) -> dict[str, Any]:
    """Describe this job's share of a bounded, fleet-wide trace selection."""
    return {
        "selection": "latest_object_store_last_modified_across_active_rl_jobs",
        "trace_sync_limit": trace_sync_limit,
        "available_traces": inventory.available,
        "selected_traces": len(selected),
        "omitted_traces": inventory.available - len(selected),
        "fleet_available_traces": fleet_available,
        "fleet_selected_traces": fleet_selected,
        "selected": [
            {"name": trace.name, "last_modified": trace.last_modified.isoformat(), "completed": trace.completed}
            for trace in selected
        ],
    }


def collect_trace_inventory(job: IrisRlJob) -> TraceInventory:
    """List all remote objects for one job before selecting a fleet-wide subset."""
    uri = iris_trials_uri(job)
    bucket, prefix = split_s3_uri(uri)
    base = kubectl_base(COREWEAVE_CLUSTERS[job.cluster.name], SimpleNamespace(kubeconfig=None, kube_context=None))
    client = object_store_client(base, COREWEAVE_CLUSTERS[job.cluster.name])
    root_prefix = f"{prefix.rstrip('/')}/"
    objects = iter_objects(client, bucket, root_prefix)
    traces, available, completed = recent_trace_jobs(objects, root_prefix, trace_sync_limit=0)
    return TraceInventory(job, bucket, root_prefix, client, tuple(traces), available, completed)


def select_recent_fleet_traces(
    inventories: list[TraceInventory], trace_sync_limit: int
) -> dict[tuple[str, str], list[TraceJobObjects]]:
    """Select the newest trace directories globally, using S3 modification metadata."""
    candidates = [(inventory, trace) for inventory in inventories for trace in inventory.traces]
    candidates.sort(
        key=lambda candidate: (
            candidate[1].last_modified,
            candidate[0].job.cluster.name,
            candidate[0].job.job_id,
            candidate[1].name,
        ),
        reverse=True,
    )
    if trace_sync_limit:
        candidates = candidates[:trace_sync_limit]
    selected: dict[tuple[str, str], list[TraceJobObjects]] = {}
    for inventory, trace in candidates:
        key = (inventory.job.cluster.name, inventory.job.job_id)
        selected.setdefault(key, []).append(trace)
    return selected


def sync_trace_inventory(
    inventory: TraceInventory,
    destination: Path,
    selected: list[TraceJobObjects],
    max_non_log_bytes: int,
    trace_sync_limit: int,
    fleet_available: int,
    fleet_selected: int,
    progress: ProgressReporter | None = None,
) -> tuple[str, int, int, str | None]:
    """Mirror the globally selected traces for one job without deleting prior evidence."""
    destination.mkdir(exist_ok=True)
    write_json(
        destination / "sync_selection.json",
        trace_selection_manifest(selected, inventory, trace_sync_limit, fleet_available, fleet_selected),
    )
    copied = skipped = 0
    skipped_objects: list[dict[str, Any]] = []
    candidate_objects = sum(len(trace.objects) for trace in selected)
    candidate_bytes = sum(int(item["Size"]) for trace in selected for item in trace.objects)
    if progress:
        progress.phase(
            f"trace transfer {inventory.job.cluster.name}/{inventory.job.short_name}: "
            f"{len(selected):,} traces, {candidate_objects:,} objects, "
            f"{candidate_bytes / 1_048_576:.1f} MiB candidate payload"
        )
    inspected = 0
    try:
        for trace in selected:
            for item in trace.objects:
                inspected += 1
                relative = item["Key"].removeprefix(inventory.root_prefix)
                size = int(item["Size"])
                if max_non_log_bytes and size > max_non_log_bytes and not is_log_object(relative):
                    skipped += 1
                    skipped_objects.append({"key": relative, "size": size, "reason": "non_log_size_limit"})
                    if progress and (inspected == candidate_objects or inspected % 25 == 0):
                        progress.phase(
                            f"trace transfer {inventory.job.short_name}: "
                            f"{inspected:,}/{candidate_objects:,} objects inspected; "
                            f"{copied:,} downloaded, {skipped:,} size-skipped"
                        )
                    continue
                local_path = destination / safe_relative_path(item["Key"], inventory.root_prefix)
                if local_path.exists() and local_path.stat().st_size == size:
                    if progress and (inspected == candidate_objects or inspected % 25 == 0):
                        progress.phase(
                            f"trace transfer {inventory.job.short_name}: "
                            f"{inspected:,}/{candidate_objects:,} objects inspected; "
                            f"{copied:,} downloaded, {skipped:,} size-skipped"
                        )
                    continue
                local_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    inventory.client.download_file(inventory.bucket, item["Key"], str(local_path))
                except ClientError as error:
                    if not is_missing_object_error(error):
                        raise
                    skipped += 1
                    skipped_objects.append({"key": relative, "size": size, "reason": "missing_after_listing"})
                    continue
                copied += 1
                if progress and (inspected == candidate_objects or inspected % 25 == 0):
                    progress.phase(
                        f"trace transfer {inventory.job.short_name}: "
                        f"{inspected:,}/{candidate_objects:,} objects inspected; "
                        f"{copied:,} downloaded, {skipped:,} size-skipped"
                    )
    except Exception as error:  # object stores may race a currently-uploading trace
        write_json(destination / "skipped_objects.json", skipped_objects)
        return (
            f"partial: fleet {len(selected):,}/{inventory.available:,} selected "
            f"({fleet_selected:,}/{fleet_available:,} total); {copied:,} copied, {skipped:,} skipped",
            inventory.available,
            inventory.completed,
            str(error)[-ERROR_DETAIL_CHARS:],
        )
    write_json(destination / "skipped_objects.json", skipped_objects)
    scope = "all" if trace_sync_limit == 0 else "newest"
    return (
        f"fleet {scope} {len(selected):,}/{inventory.available:,} selected "
        f"({fleet_selected:,}/{fleet_available:,} total); {copied:,} copied, {skipped:,} skipped",
        inventory.available,
        inventory.completed,
        None,
    )


@dataclass(frozen=True)
class ParsedMetrics:
    step: int | None
    total: int | None
    metrics: dict[str, Any]
    tis_enabled: bool | None
    error: str | None


def parse_metrics(finelog: Path) -> ParsedMetrics:
    if not finelog.exists():
        return ParsedMetrics(None, None, {}, None, None)
    try:
        text = finelog.read_text(errors="replace")
    except OSError as error:
        return ParsedMetrics(None, None, {}, None, f"could not read finelog: {error}")
    progress = PROGRESS_PATTERN.findall(text)
    step = int(progress[-1][0]) if progress else None
    total = int(progress[-1][1]) if progress else None
    tis_enabled = parse_tis_enabled(text)
    result = parse_training_metrics_result(text)
    if result.records:
        latest = result.records[-1]
        return ParsedMetrics(
            latest.step,
            total,
            latest.metrics,
            tis_enabled,
            training_metrics_parse_error(result.malformed_lines),
        )
    if result.malformed_lines:
        return ParsedMetrics(step, total, {}, tis_enabled, training_metrics_parse_error(result.malformed_lines))
    return ParsedMetrics(step, total, {}, tis_enabled, None)


def display_metric(value: Any | None, precision: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{precision}g}"
    return str(value)


def tis_ratio_summary(metrics: dict[str, Any]) -> str:
    """Render the best available TIS ratio diagnostic without mislabeling it.

    Worker ``train_status`` keys are namespaced under ``policy/`` by the
    trainer before they reach ``WANDB_MIRROR``. Older runs predate that worker
    diagnostic but expose the inference/train importance-ratio mean directly.
    That legacy value is useful, but it is not mean absolute log-ratio, so keep
    the label distinct.
    """
    log_ratio = metric_value(metrics, *TIS_LOG_RATIO_ABS_MEAN_KEYS)
    if log_ratio is not None:
        return f"TIS |log r|={display_metric(log_ratio)}"

    importance_ratio = metric_value(metrics, *TIS_IMPORTANCE_RATIO_MEAN_KEYS)
    return f"TIS r={display_metric(importance_ratio)}"


def tis_alignment_summary(metrics: dict[str, Any]) -> str:
    """Render served-token versus trainer-token exact alignment for TIS."""
    return f"TIS exact={display_metric(metric_value(metrics, *TIS_EXACT_MATCH_KEYS))}"


def tis_diagnostic_summaries(metrics: dict[str, Any], enabled: bool | None) -> tuple[str, ...]:
    """Render available TIS diagnostics or explain why they are absent."""
    summaries: list[str] = []
    if metric_value(metrics, *TIS_EXACT_MATCH_KEYS) is not None:
        summaries.append(tis_alignment_summary(metrics))
    if (
        metric_value(metrics, *TIS_LOG_RATIO_ABS_MEAN_KEYS) is not None
        or metric_value(metrics, *TIS_IMPORTANCE_RATIO_MEAN_KEYS) is not None
    ):
        summaries.append(tis_ratio_summary(metrics))

    if summaries:
        if enabled is False:
            summaries.append("TIS correction disabled")
        return tuple(summaries)
    if enabled is False:
        return ("TIS disabled",)
    if enabled is True:
        return ("TIS enabled; metrics missing",)
    return ("TIS unavailable",)


def token_probability_shift_summary(metrics: dict[str, Any]) -> str:
    """Render the center and tail of per-token old/new policy log-probability movement."""
    # A widening p99/max tail relative to the mean signals that a few tokens
    # carry extreme changes; these values are not literal top-k probability mass.
    values = (
        metric_value(metrics, *POLICY_LOG_RATIO_ABS_MEAN_KEYS),
        metric_value(metrics, *POLICY_LOG_RATIO_ABS_P99_KEYS),
        metric_value(metrics, *POLICY_LOG_RATIO_ABS_MAX_KEYS),
    )
    rendered = "/".join(display_metric(value) for value in values)
    return f"token |Δlog p| μ/p99/max={rendered}"


def terminal_signal(finelog: Path) -> str | None:
    """Return the first actionable error line from the latest Iris attempt.

    Explicit TRACE, DEBUG, INFO, and WARNING lines are non-terminal even when
    their message embeds a traceback. When the retained finelog contains an Iris
    setup boundary, failures before the latest rank-zero boundary are stale.
    """
    if not finelog.exists():
        return None
    try:
        tail = finelog.read_text(errors="replace")[-2_000_000:]
    except OSError:
        return None
    attempt_boundaries = list(IRIS_ATTEMPT_BOUNDARY_PATTERN.finditer(tail))
    if attempt_boundaries:
        tail = tail[attempt_boundaries[-1].start() :]
    for line in tail.splitlines():
        if NON_ERROR_LOG_LEVEL_PATTERN.search(line):
            continue
        if any(pattern.search(line) for pattern in ERROR_PATTERNS):
            return line
    return None


def sync_warning(errors: tuple[str, ...]) -> str | None:
    """Render a stable table cell for artifact-sync errors, never raw proxy bodies."""
    if not errors:
        return None
    first_error = errors[0]
    if "Ray/vLLM" in first_error:
        return "Ray/vLLM log sync unavailable; local diagnostic saved"
    return first_error[-90:]


def _monitor_error(scope: str, operation: str, error: object) -> MonitorError:
    message = str(error).strip() or type(error).__name__
    return MonitorError(scope, operation, message)


def _state_cell(state: str) -> StyledCell:
    if state in {"running", "succeeded"}:
        tone = "success"
    elif state == AWAITING_PLACEMENT:
        tone = "warning"
    else:
        tone = "error"
    return StyledCell(state, tone)


def effective_state(job: RlJob) -> str:
    """Return ``awaiting placement`` when the workload has not started running.

    Iris marks the root RL job running (state 3) as soon as the controller is
    accepted, which can be long before any task is placed on a node. The active
    task state distinguishes a job that is merely queued from one whose pods are
    up and consuming resources.
    """
    if job.state in PRE_RUNNING_STATES:
        return AWAITING_PLACEMENT
    if job.state == "running" and job.task_state in PRE_RUNNING_STATES:
        return AWAITING_PLACEMENT
    return job.state


def job_filter_values(job: MonitoredJob, *, now_ms: int) -> dict[str, str]:
    """Return the pre-sync RL job fields available to ``--filter``."""
    return {
        "cluster": job.cluster.name,
        "job": job.job_id,
        "name": job.short_name,
        "dataset": job.dataset,
        "type": "RL",
        "state": effective_state(job),
        "submitted": datetime.fromtimestamp(job.submitted_at_ms / 1000, UTC).strftime("%m-%d %H:%M"),
        "duration": format_duration(job.submitted_at_ms, job.finished_at_ms, now_ms=now_ms),
    }


def report_row(job: MonitoredJob, artifacts: ArtifactResult, directory: Path) -> list[object]:
    """Build one status row; monitor failures belong in the separate error report."""
    parsed = parse_metrics(directory / "finelog.log")
    step, total, metrics = parsed.step, parsed.total, parsed.metrics
    step_display = "—" if step is None else f"{step}/{total if total is not None else '—'}"
    reward = metric_value(metrics, *REWARD_KEYS)
    policy_loss = metric_value(metrics, *POLICY_LOSS_KEYS)
    grad_norm = metric_value(metrics, *GRAD_NORM_KEYS)
    entropy = metric_value(metrics, *ENTROPY_KEYS)
    signal = terminal_signal(directory / "finelog.log")
    trend = "; ".join(
        (
            f"entropy={display_metric(entropy)}",
            *tis_diagnostic_summaries(metrics, parsed.tis_enabled),
            token_probability_shift_summary(metrics),
        )
    )
    if signal:
        trend = "workload error detected; see error report"
    elif step is None:
        trend += "; bring-up/buffer (metrics not emitted)"
    return [
        f"{job.cluster.name}/{job.short_name}",
        job.dataset,
        _state_cell(effective_state(job)),
        step_display,
        display_metric(reward),
        display_metric(policy_loss),
        display_metric(grad_norm),
        artifacts.traces,
        StyledCell(trend, "error" if signal else "muted"),
    ]


def sync_jupiter_job(
    job: JupiterRlJob,
    bundle_root: Path,
    *,
    scope: SyncScope,
    jupiter_host: str = DEFAULT_JUPITER_HOST,
    jupiter_trace_sync_limit: int = DEFAULT_JUPITER_TRACE_SYNC_LIMIT,
    max_non_log_bytes: int = DEFAULT_MAX_NON_LOG_BYTES,
    progress: ProgressReporter | None = None,
) -> tuple[ArtifactResult, Path]:
    """Sync one explicit Jupiter run, including its bounded GPFS trace selection."""
    bundle = job_bundle(bundle_root, job.cluster.name, job.bundle_job_id)
    directory = bundle.directory
    directory.mkdir(parents=True, exist_ok=True)
    if progress:
        progress.phase(f"bounded GPFS artifact sync {job.cluster.name}/{job.short_name}")
    result = sync_jupiter_artifacts(
        JupiterRunSpec(job.job_id, job.experiment_dir, job.launcher_job_name),
        directory,
        host=jupiter_host,
        trace_sync_limit=jupiter_trace_sync_limit,
        max_non_log_bytes=max_non_log_bytes,
        scope="status" if scope == "status" else "full",
        finelog_tail_lines=CURRENT_FINELOG_MAX_LINES,
    )
    return (
        ArtifactResult(
            result.finelog,
            "not applicable (Jupiter)",
            result.ray_logs,
            result.traces,
            None,
            None,
            result.errors,
            slurm_logs=result.slurm_logs,
            trace_selected=result.trace_selected,
        ),
        directory,
    )


def sync_iris_job(
    job: IrisRlJob,
    bundle_root: Path,
    *,
    scope: SyncScope,
    progress: ProgressReporter | None = None,
) -> tuple[ArtifactResult, Path]:
    """Sync Iris evidence except traces, which use one fleet-wide budget."""
    bundle = job_bundle(bundle_root, job.cluster.name, job.bundle_job_id)
    directory = bundle.directory
    directory.mkdir(parents=True, exist_ok=True)
    if scope == "status":
        if progress:
            progress.phase(f"current finelog tail {job.cluster.name}/{job.short_name}")
        finelog, error = fetch_finelog(job, directory, scope="current")
        return ArtifactResult(
            finelog,
            "not requested",
            "not requested",
            "not requested",
            None,
            None,
            (error,) if error else (),
        ), directory
    errors: list[str] = []
    if progress:
        progress.phase(f"finelog sync {job.cluster.name}/{job.short_name}")
    finelog, error = fetch_finelog(job, directory)
    if error:
        errors.append(error)
    if scope == "terminal":
        return ArtifactResult(
            finelog,
            "not requested (terminal)",
            "not requested (terminal)",
            "not requested (terminal)",
            None,
            None,
            tuple(errors),
        ), directory
    if progress:
        progress.phase(f"pod + Ray/vLLM log sync {job.cluster.name}/{job.short_name}")
    pod_logs, ray_logs, pod_errors = sync_pod_and_ray_logs(job, directory, progress=progress)
    errors.extend(pod_errors)
    return ArtifactResult(finelog, pod_logs, ray_logs, "pending fleet selection", None, None, tuple(errors)), directory


def sync_fleet_trace_jobs(
    job_directories: list[tuple[IrisRlJob, Path]],
    max_non_log_bytes: int,
    trace_sync_limit: int,
    progress: ProgressReporter | None = None,
) -> dict[tuple[str, str], tuple[str, int | None, int | None, str | None]]:
    """Select at most one global trace budget, then sync each job's selected share."""
    statuses: dict[tuple[str, str], tuple[str, int | None, int | None, str | None]] = {}
    inventories: list[TraceInventory] = []
    for index, (job, _) in enumerate(job_directories, start=1):
        key = (job.cluster.name, job.job_id)
        try:
            if progress:
                progress.phase(f"trace inventory {index}/{len(job_directories)} {job.cluster.name}/{job.short_name}")
            inventories.append(collect_trace_inventory(job))
        except Exception as error:
            # Object-store failures must degrade one row, not abort the fleet-wide report.
            statuses[key] = ("unavailable", None, None, str(error)[-ERROR_DETAIL_CHARS:])

    selected = select_recent_fleet_traces(inventories, trace_sync_limit)
    fleet_available = sum(inventory.available for inventory in inventories)
    fleet_selected = sum(len(traces) for traces in selected.values())
    if progress:
        progress.phase(
            f"trace selection: {fleet_selected:,}/{fleet_available:,} newest trace jobs across "
            f"{len(inventories):,} RL jobs"
        )
    directories = {(job.cluster.name, job.job_id): directory for job, directory in job_directories}
    for inventory in inventories:
        key = (inventory.job.cluster.name, inventory.job.job_id)
        statuses[key] = sync_trace_inventory(
            inventory,
            directories[key] / "trace_jobs",
            selected.get(key, []),
            max_non_log_bytes,
            trace_sync_limit,
            fleet_available,
            fleet_selected,
            progress,
        )
    return statuses


def write_job_manifest(
    job: MonitoredJob,
    bundle_root: Path,
    directory: Path,
    artifacts: ArtifactResult,
    max_non_log_bytes: int,
    trace_sync_limit: int,
) -> None:
    bundle = job_bundle(bundle_root, job.cluster.name, job.bundle_job_id)
    write_bundle_manifest(
        bundle,
        {
            "kind": "rl",
            "job": asdict(job),
            "job_directory": str(directory),
            "trials_uri": job.trials_uri,
            "synced_at": datetime.now(UTC).isoformat(),
            "max_non_log_bytes": max_non_log_bytes,
            "trace_sync_limit": trace_sync_limit,
            "artifacts": asdict(artifacts),
        },
    )


def validate_args(args: argparse.Namespace) -> None:
    """Validate cross-option monitor invariants before any remote access."""
    if args.max_non_log_bytes < 0:
        raise ValueError("--max-non-log-bytes must be non-negative")
    if args.trace_sync_limit < 0:
        raise ValueError("--trace-sync-limit must be non-negative")
    if args.jupiter_trace_sync_limit < 0:
        raise ValueError("--jupiter-trace-sync-limit must be non-negative")
    if args.hours < 0:
        raise ValueError("--hours must be non-negative")
    if args.jupiter_only and not args.jupiter_run:
        raise ValueError("--jupiter-only requires at least one --jupiter-run")
    validate_jupiter_host(args.jupiter_host)
    if not args.all_users and not re.fullmatch(r"[A-Za-z0-9_-]+", args.user):
        raise ValueError("--user may contain only letters, numbers, _ and -")


def discover_selected_jobs(
    args: argparse.Namespace,
    progress: ProgressReporter,
    *,
    now_ms: int,
) -> tuple[list[MonitoredJob], list[MonitorError]]:
    """Discover the default Iris fleet and resolve explicitly named Jupiter runs."""
    jobs: list[MonitoredJob] = []
    errors: list[MonitorError] = []
    scope_user = None if args.all_users else args.user
    submitted_since_ms = None if args.hours == 0 else now_ms - int(args.hours * 3_600_000)
    if not args.jupiter_only:
        for cluster in CLUSTERS:
            progress.phase(f"discovering RL jobs on {cluster.name}")
            try:
                found, discovery_errors = discover_rl_jobs(
                    cluster,
                    scope_user,
                    submitted_since_ms=submitted_since_ms,
                )
            except Exception as error:
                found, discovery_errors = [], [str(error)]
            active_count = sum(not job.is_terminal for job in found)
            terminal_count = len(found) - active_count
            window_label = "all history" if args.hours == 0 else f"submitted in last {args.hours:g}h"
            progress.phase(
                f"discovery {cluster.name}: {active_count:,} active, {terminal_count:,} succeeded/failed; {window_label}"
            )
            jobs.extend(found)
            errors.extend(_monitor_error(cluster.name, "job discovery", error) for error in discovery_errors)

    for run in args.jupiter_run:
        progress.phase(f"checking explicit Jupiter Slurm job {run.job_id}")
        try:
            status = query_jupiter_job_status(run, host=args.jupiter_host)
        except Exception as error:
            status = JupiterJobStatus("unknown", run.job_name, str(error))
        if status.error:
            errors.append(_monitor_error(f"{JUPITER_CLUSTER.name}/{run.job_id}", "Slurm status", status.error))
        jobs.append(
            JupiterRlJob(
                cluster=JUPITER_CLUSTER,
                job_id=run.job_id,
                state=status.state,
                submitted_at_ms=now_ms,
                entrypoint="",
                experiment_dir=run.experiment_dir,
                launcher_job_name=status.job_name,
            )
        )
    return jobs, errors


def sync_per_job_evidence(
    settings: MonitorSettings,
    jobs: list[MonitoredJob],
    progress: ProgressReporter,
) -> list[SyncedJob]:
    """Sync each job through its backend-specific artifact transport."""
    synced_jobs: list[SyncedJob] = []
    ordered_jobs = sorted(jobs, key=lambda item: (item.cluster.name, item.submitted_at_ms, item.job_id))
    for index, job in enumerate(ordered_jobs, start=1):
        try:
            progress.phase(f"job evidence {index}/{len(ordered_jobs)} {job.cluster.name}/{job.short_name}")
            scope = "status" if settings.status_only else "terminal" if job.is_terminal else "full"
            artifacts, directory = job.sync_artifacts(settings, progress, scope=scope)
        except Exception as error:
            directory = job_directory(settings.bundle_root, job)
            artifacts = ArtifactResult(
                "unavailable",
                "unavailable",
                "unavailable",
                "unavailable",
                None,
                None,
                (f"{type(error).__name__}: {error}",),
            )
        synced_jobs.append(SyncedJob(job, artifacts, directory))
    return synced_jobs


def apply_iris_trace_budget(
    settings: MonitorSettings,
    synced_jobs: list[SyncedJob],
    progress: ProgressReporter,
) -> list[SyncedJob]:
    """Apply the global Iris trace budget without touching Jupiter selections."""
    active_iris_job_directories = [
        (item.job, item.directory) for item in synced_jobs if item.job.uses_iris_trace_budget
    ]
    if settings.status_only:
        return synced_jobs
    if not active_iris_job_directories:
        progress.phase("no active Iris RL jobs; fleet object-store trace sync skipped")
        return synced_jobs

    progress.phase("starting fleet-wide trace inventory and transfer")
    try:
        trace_statuses = sync_fleet_trace_jobs(
            active_iris_job_directories,
            settings.max_non_log_bytes,
            settings.iris_trace_sync_limit,
            progress,
        )
    except Exception as error:
        trace_statuses = {
            (job.cluster.name, job.job_id): (
                "unavailable",
                None,
                None,
                f"{type(error).__name__}: {error}",
            )
            for job, _directory in active_iris_job_directories
        }

    synchronized: list[SyncedJob] = []
    for item in synced_jobs:
        job, artifacts, directory = item.job, item.artifacts, item.directory
        if not job.uses_iris_trace_budget:
            synchronized.append(item)
            continue
        traces, started, completed, trace_error = trace_statuses.get(
            (job.cluster.name, job.job_id),
            ("unavailable", None, None, "fleet trace sync returned no status"),
        )
        errors_for_job = artifacts.errors + ((f"trace sync: {trace_error}",) if trace_error else ())
        synchronized.append(
            SyncedJob(
                job,
                replace(
                    artifacts,
                    traces=traces,
                    trace_started=started,
                    trace_completed=completed,
                    errors=errors_for_job,
                ),
                directory,
            )
        )
    return synchronized


def build_job_report_entry(settings: MonitorSettings, synced: SyncedJob) -> JobReportEntry:
    """Inspect one local bundle, write its manifest, and render its status row."""
    job, artifacts, directory = synced.job, synced.artifacts, synced.directory
    scope = f"{job.cluster.name}/{job.job_id}"
    errors = [_monitor_error(scope, "artifact sync", error) for error in artifacts.errors]
    parsed_metrics = parse_metrics(directory / "finelog.log")
    if parsed_metrics.error:
        errors.append(_monitor_error(scope, "Finelog parse", parsed_metrics.error))
    signal = terminal_signal(directory / "finelog.log")
    if signal:
        errors.append(_monitor_error(scope, "workload signal", signal))
    try:
        write_job_manifest(
            job,
            settings.bundle_root,
            directory,
            artifacts,
            settings.max_non_log_bytes,
            job.trace_sync_limit(settings),
        )
    except Exception as error:
        errors.append(_monitor_error(scope, "manifest write", error))
    try:
        row = report_row(job, artifacts, directory)
    except Exception as error:
        errors.append(_monitor_error(scope, "row rendering", error))
        row = [
            f"{job.cluster.name}/{job.short_name}",
            job.dataset,
            _state_cell(effective_state(job)),
            "—",
            "—",
            "—",
            "—",
            "unavailable",
            StyledCell("status unavailable; see error report", "error"),
        ]
    return JobReportEntry(
        key=f"{job.cluster.name}/{job.job_id}",
        value=JobReportValue(job.cluster.name, str(directory), artifacts),
        row=row,
        errors=tuple(errors),
    )


def collect_report_data(
    settings: MonitorSettings,
    synced_jobs: list[SyncedJob],
    errors: list[MonitorError],
) -> ReportData:
    """Build report rows and manifests while retaining per-job failures."""
    entries = [build_job_report_entry(settings, synced) for synced in synced_jobs]
    for entry in entries:
        errors.extend(entry.errors)
    return ReportData(
        rows=[entry.row for entry in entries],
        jobs={entry.key: entry.value for entry in entries},
        errors=errors,
    )


def publish_report(
    args: argparse.Namespace,
    report_directory: Path,
    checked_at: datetime,
    rows: list[list[object]],
    job_report: dict[str, JobReportValue],
    errors: list[MonitorError],
    budgets: list[UserBudget],
    progress: ProgressReporter,
) -> None:
    """Render, persist, and print one fleet status report."""
    headers = ["Job", "Dataset", "State", "Step", "Reward", "Policy Loss", "Grad Norm", "Traces", "Trend"]
    table = (
        box_table(headers, rows) if rows else "No RL jobs matched the selected Iris window or explicit Jupiter runs."
    )
    terminal_table = box_table(headers, rows, color=sys.stdout.isatty()) if rows else table
    filter_suffix = f"; filters={','.join(args.filter)}" if args.filter else ""
    window = "all" if args.hours == 0 else f"{args.hours:g}h"
    timestamp = checked_at.strftime("%Y%m%dT%H%M%SZ")
    error_report_path = write_error_report(
        report_directory,
        timestamp,
        "RL monitor errors",
        checked_at,
        errors,
    )
    error_summary = f"Monitor errors: {len(errors)}; details: {error_report_path}"
    heading = f"# RL status — {checked_at.isoformat()}; Iris submitted={window}{filter_suffix}"
    budget_section = f"{render_budget_section(args.user, budgets)}\n\n" if budgets else ""
    report = f"{heading}\n\n{budget_section}{table}\n\n{error_summary}\n"
    report_path = report_directory / f"{timestamp}.md"
    report_path.write_text(report)
    (report_directory / "latest.md").write_text(report)
    write_json(
        report_directory / "latest.json",
        {
            "checked_at": checked_at.isoformat(),
            "jobs": {key: asdict(value) for key, value in job_report.items()},
            "budgets": [asdict(budget) for budget in budgets],
            "report": str(report_path),
            "error_count": len(errors),
            "error_report": str(error_report_path),
        },
    )
    progress.phase("report written; printing status table")
    print(f"{heading}\n\n{budget_section}{terminal_table}\n\n{error_summary}")


def main() -> int:
    args = parse_args()
    validate_args(args)
    filters = parse_regex_filters(
        args.filter,
        {"cluster", "job", "name", "dataset", "type", "state", "submitted", "duration"},
    )
    args.bundle_root.mkdir(parents=True, exist_ok=True)
    report_directory = args.bundle_root / "reports" / "rl"
    report_directory.mkdir(parents=True, exist_ok=True)
    progress = ProgressReporter(enabled=not args.quiet_progress)
    checked_at = datetime.now(UTC)
    now_ms = int(checked_at.timestamp() * 1000)
    settings = MonitorSettings(
        bundle_root=args.bundle_root,
        status_only=args.status_only,
        max_non_log_bytes=args.max_non_log_bytes,
        iris_trace_sync_limit=args.trace_sync_limit,
        jupiter_host=args.jupiter_host,
        jupiter_trace_sync_limit=args.jupiter_trace_sync_limit,
    )

    jobs, errors = discover_selected_jobs(args, progress, now_ms=now_ms)
    jobs = filter_records(jobs, filters, lambda job: job_filter_values(job, now_ms=now_ms))
    synced_jobs = sync_per_job_evidence(settings, jobs, progress)
    synced_jobs = apply_iris_trace_budget(settings, synced_jobs, progress)
    report_data = collect_report_data(settings, synced_jobs, errors)
    budgets: list[UserBudget] = []
    if not args.all_users and not args.jupiter_only:
        budgets = fetch_user_budgets(args.user, progress)
    publish_report(
        args,
        report_directory,
        checked_at,
        report_data.rows,
        report_data.jobs,
        report_data.errors,
        budgets,
        progress,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
