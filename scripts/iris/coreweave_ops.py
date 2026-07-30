#!/usr/bin/env python3
"""CoreWeave Iris pod and object-store operations for job diagnostics.

The collector discovers the active launcher pod, downloads its Ray/vLLM logs,
and copies a deterministic sample of completed Harbor trial directories from
the job's object-store output.  It obtains short-lived object-store credentials
from the in-cluster task secret without writing those credentials to disk.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import logging
import random
import subprocess
import sys
import tarfile
import time
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

import boto3
from botocore.config import Config


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.iris.coreweave_clusters import CLUSTERS as COREWEAVE_CLUSTERS, ClusterConfig  # noqa: E402
from scripts.iris.iris_ops import DNS_ATTEMPTS, DNS_INITIAL_BACKOFF, job_id_parts  # noqa: E402


LOGGER = logging.getLogger(__name__)
NAMESPACE = "iris"
TASK_SECRET = "iris-task-env"
RAY_LOG_DIR = "/tmp/ray/session_latest/logs"
GPU_RL_PYTHON = "/opt/openthoughts/envs/rl/bin/python"
RAY_LOG_PATTERNS = (
    "worker-*.out",
    "worker-*.err",
    "python-core-driver-*.log",
    "gcs_server.*",
    "monitor.*",
    "dashboard.*",
    "raylet.*",
)
DEFAULT_MAX_TRIAL_FILE_BYTES = 20 * 1024 * 1024
DEFAULT_MAX_VLLM_LOG_BYTES = 100 * 1024 * 1024
RAY_LOG_SYNC_MANIFEST = ".ray-vllm-sync-manifest.json"
TRANSIENT_KUBECTL_EXEC_MARKERS = (
    "<!doctype html",
    "<html",
    "command terminated with exit code 1",
    "connection reset",
    "error dialing backend",
    "stream error",
    "unexpected eof",
    "i/o timeout",
)
IRIS_JOB_ID_LABEL = "iris.job_id"
KUEUE_POD_GROUP_LABEL = "kueue.x-k8s.io/pod-group-name"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "job",
        help="Iris task name, such as /benjaminfeuer/glm52-pilot-codecontests-r7.",
    )
    parser.add_argument("--cluster", choices=sorted(COREWEAVE_CLUSTERS), default="cw-rno2a")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible sampling.")
    parser.add_argument(
        "--max-trial-file-bytes",
        type=int,
        default=DEFAULT_MAX_TRIAL_FILE_BYTES,
        help="Skip individual trial objects larger than this bound (default: 20 MiB).",
    )
    parser.add_argument(
        "--max-vllm-log-bytes",
        type=int,
        default=DEFAULT_MAX_VLLM_LOG_BYTES,
        help="Skip individual Ray/vLLM log files larger than this bound (default: 100 MiB).",
    )
    parser.add_argument("--pod", help="Exact pod name. Required when job matching is ambiguous.")
    parser.add_argument("--container", default="task")
    parser.add_argument(
        "--trials-s3",
        help="Explicit S3 prefix containing trial directories; bypasses command-line discovery.",
    )
    parser.add_argument("--kubeconfig", type=Path)
    parser.add_argument("--kube-context")
    args = parser.parse_args()
    if args.sample_size <= 0:
        parser.error("--sample-size must be positive")
    if args.max_trial_file_bytes <= 0 or args.max_vllm_log_bytes <= 0:
        parser.error("file-size limits must be positive")
    return args


def command(args: list[str], *, input_text: str | None = None, timeout: int | None = None) -> str:
    for attempt in range(DNS_ATTEMPTS):
        result = subprocess.run(
            args,
            input=input_text,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return result.stdout
        stderr = result.stderr.strip()
        if not is_transient_kubectl_exec_failure(stderr) or attempt == DNS_ATTEMPTS - 1:
            raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(args)}\n{stderr}")
        time.sleep(DNS_INITIAL_BACKOFF * 2**attempt)
    raise AssertionError("unreachable")


def kubectl_base(cluster: ClusterConfig, args: argparse.Namespace) -> list[str]:
    kubeconfig = args.kubeconfig or cluster.kubeconfig
    context = args.kube_context if args.kube_context is not None else cluster.context
    return ["kubectl", "--kubeconfig", str(kubeconfig), "--context", context]


def task_short_name(job: str) -> str:
    return job_id_parts(job)[-1]


def iris_pod_group_prefix(job: str) -> str:
    """Return the stable Kueue pod-group prefix Iris derives from a full job id."""
    canonical_job = "/" + "/".join(job_id_parts(job))
    job_hash = hashlib.sha256(canonical_job.encode()).hexdigest()[:16]
    return f"iris-pg-{job_hash}-"


def find_pod(base: list[str], args: argparse.Namespace) -> str:
    if args.pod:
        return args.pod
    pods = json.loads(command([*base, "-n", NAMESPACE, "get", "pods", "-o", "json"]))["items"]
    running_pods = [pod for pod in pods if pod.get("status", {}).get("phase") == "Running"]
    needle = task_short_name(args.job).lower()
    # Exact names and labels remain useful for short jobs. Long iris.job_id labels
    # truncate at Kubernetes' 63-character limit, so they are never used as a
    # prefix: same-day sibling arms can share the truncated value.
    labeled_job_id = args.job.lstrip("/").replace("/", ".")
    candidates = sorted(
        pod["metadata"]["name"]
        for pod in running_pods
        if (
            needle in pod["metadata"]["name"].lower()
            or pod.get("metadata", {}).get("labels", {}).get(IRIS_JOB_ID_LABEL) == labeled_job_id
        )
    )
    # Iris hashes the complete parent job id into the Kueue pod-group label.
    # The trailing attempt number changes on retry, hence prefix matching here.
    if not candidates:
        pod_group_prefix = iris_pod_group_prefix(args.job)
        candidates = sorted(
            pod["metadata"]["name"]
            for pod in running_pods
            if pod.get("metadata", {}).get("labels", {}).get(KUEUE_POD_GROUP_LABEL, "").startswith(pod_group_prefix)
        )
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise RuntimeError(f"No running Iris pod found for {args.job!r}; pass --pod if it is not running.")
    raise RuntimeError(f"Ambiguous running pods for {args.job!r}: {', '.join(candidates)}. Pass --pod.")


def pod_command_line(base: list[str], pod: str, container: str) -> list[str]:
    raw = command(
        [
            *base,
            "-n",
            NAMESPACE,
            "exec",
            pod,
            "-c",
            container,
            "--",
            "sh",
            "-c",
            "tr '\\000' '\\n' < /proc/1/cmdline",
        ]
    )
    return [part for part in raw.splitlines() if part]


def resolve_runtime_python(base: list[str], pod: str, container: str) -> str:
    """Return the canonical GPU-RL Python after verifying it exists in ``pod``.

    The gpu-rl image deliberately does not put a bare ``python`` on ``PATH``.
    Keeping this resolution here makes every CoreWeave reader use the same
    pinned runtime and fail explicitly if an image violates that contract.
    """
    try:
        command(
            [
                *base,
                "-n",
                NAMESPACE,
                "exec",
                pod,
                "-c",
                container,
                "--",
                "sh",
                "-c",
                'test -x "$1"',
                "sh",
                GPU_RL_PYTHON,
            ]
        )
    except RuntimeError as error:
        raise RuntimeError(f"Canonical GPU-RL Python {GPU_RL_PYTHON} is unavailable in {pod}: {error}") from error
    return GPU_RL_PYTHON


def resolve_container_python(base: list[str], pod: str, container: str) -> str:
    """Return a Python interpreter available in an arbitrary task container."""
    path = command(
        [
            *base,
            "-n",
            NAMESPACE,
            "exec",
            pod,
            "-c",
            container,
            "--",
            "sh",
            "-c",
            "command -v python3 || command -v python",
        ]
    ).strip()
    if not path:
        raise RuntimeError(f"No Python interpreter is available in {pod}/{container}.")
    return path


def option_value(command_line: list[str], option: str) -> str | None:
    for index, value in enumerate(command_line):
        if value == option and index + 1 < len(command_line):
            return command_line[index + 1]
        if value.startswith(f"{option}="):
            return value.removeprefix(f"{option}=")
    return None


def discover_trials_prefix(command_line: list[str]) -> str:
    experiments_dir = option_value(command_line, "--experiments_dir")
    job_name = option_value(command_line, "--job_name")
    if not experiments_dir or not job_name:
        raise RuntimeError("Could not discover --experiments_dir and --job_name from pod PID 1; pass --trials-s3.")
    return f"{experiments_dir.rstrip('/')}/{job_name.strip('/')}"


def object_store_client(base: list[str], cluster: ClusterConfig) -> Any:
    secret = json.loads(command([*base, "-n", NAMESPACE, "get", "secret", TASK_SECRET, "-o", "json"]))
    data = secret.get("data", {})

    def decode(name: str) -> str | None:
        encoded = data.get(name)
        return base64.b64decode(encoded).decode() if encoded else None

    access_key = decode("AWS_ACCESS_KEY_ID")
    secret_key = decode("AWS_SECRET_ACCESS_KEY")
    if not access_key or not secret_key:
        raise RuntimeError(f"{TASK_SECRET} does not contain AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
    return boto3.session.Session(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        aws_session_token=decode("AWS_SESSION_TOKEN"),
    ).client(
        "s3",
        endpoint_url=cluster.object_endpoint,
        config=Config(s3={"addressing_style": "virtual"}),
    )


def split_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Expected an s3:// URI, got {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/").rstrip("/")


def iter_objects(client: Any, bucket: str, prefix: str) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        objects.extend(page.get("Contents", []))
    return objects


def completed_trials(client: Any, bucket: str, prefix: str) -> list[str]:
    prefix_with_slash = f"{prefix.rstrip('/')}/"
    trials = {
        relative.split("/", 1)[0]
        for item in iter_objects(client, bucket, prefix_with_slash)
        if (relative := item["Key"].removeprefix(prefix_with_slash)).count("/") == 1
        and relative.endswith("/result.json")
    }
    return sorted(trials)


def safe_relative_path(key: str, prefix: str) -> Path:
    relative = PurePosixPath(key.removeprefix(prefix).lstrip("/"))
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise RuntimeError(f"Refusing unsafe object key {key!r}")
    return Path(*relative.parts)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def save_task_log(base: list[str], pod: str, container: str, destination: Path) -> None:
    result = subprocess.run(
        [*base, "-n", NAMESPACE, "logs", pod, "-c", container, "--tail", "5000"],
        text=True,
        capture_output=True,
    )
    destination.write_text(result.stdout)
    if result.returncode:
        LOGGER.warning("Could not collect complete task log: %s", result.stderr.strip())


def ray_log_inventory(
    base: list[str],
    pod: str,
    container: str,
    *,
    patterns: tuple[str, ...] | None = RAY_LOG_PATTERNS,
    python_executable: str | None = None,
) -> list[dict[str, Any]]:
    """List Ray logs, or every Ray log when ``patterns`` is ``None``."""
    # This value is embedded into a Python ``-c`` program, not JSON.  In
    # particular, the all-logs mode uses ``None`` (JSON's ``null`` is invalid
    # Python and previously made the RL watcher's full inventory fail).
    serialized_patterns = repr(patterns)
    script = """
import json
from pathlib import Path
root = Path('/tmp/ray/session_latest/logs')
patterns = __PATTERNS__
root = Path('/tmp/ray/session_latest/logs')
files = set(root.rglob('*')) if patterns is None else {path for pattern in patterns for path in root.glob(pattern)}
files = {path for path in files if path.is_file()}
print(json.dumps([
    {
        'path': str(path.relative_to(root)),
        'size': path.stat().st_size,
        # A same-named Ray log may be rotated and recreated.  Its inode lets
        # the local collector distinguish that from an append-only growth.
        'inode': path.stat().st_ino,
    }
    for path in sorted(files)
]))
""".replace("__PATTERNS__", serialized_patterns)
    runtime_python = python_executable or resolve_container_python(base, pod, container)
    raw = command([*base, "-n", NAMESPACE, "exec", pod, "-c", container, "--", runtime_python, "-c", script])
    return json.loads(raw)


def _safe_ray_log_path(value: str) -> Path:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise RuntimeError(f"Refusing unsafe Ray/vLLM log path {value!r}")
    return Path(*path.parts)


def _read_ray_log_sync_manifest(destination: Path) -> dict[str, dict[str, int]]:
    path = destination / RAY_LOG_SYNC_MANIFEST
    try:
        loaded = json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError):
        return {}
    files = loaded.get("files") if isinstance(loaded, dict) else None
    if not isinstance(files, dict):
        return {}
    return {
        name: entry
        for name, entry in files.items()
        if isinstance(name, str)
        and isinstance(entry, dict)
        and isinstance(entry.get("inode"), int)
        and isinstance(entry.get("size"), int)
    }


def _write_ray_log_sync_manifest(destination: Path, inventory: list[dict[str, Any]]) -> None:
    files = {
        str(item["path"]): {"inode": int(item["inode"]), "size": int(item["size"])}
        for item in inventory
        if isinstance(item.get("inode"), int)
    }
    write_json(destination / RAY_LOG_SYNC_MANIFEST, {"version": 1, "files": files})


def plan_ray_log_delta(inventory: list[dict[str, Any]], destination: Path) -> list[dict[str, Any]]:
    """Choose append offsets for an append-only Ray/vLLM log mirror.

    A growing file can be appended only when the previous inode and exact
    local size agree with the durable manifest.  Any uncertainty (a missing
    manifest, rotation, truncation, or local edit) deliberately falls back to
    a complete replacement.
    """
    previous = _read_ray_log_sync_manifest(destination)
    transfers: list[dict[str, Any]] = []
    for item in inventory:
        path = str(item["path"])
        size = int(item["size"])
        inode = item.get("inode")
        local_path = destination / _safe_ray_log_path(path)
        prior = previous.get(path)
        offset = 0
        if (
            isinstance(inode, int)
            and prior is not None
            and prior["inode"] == inode
            and 0 <= prior["size"] <= size
            and local_path.is_file()
            and local_path.stat().st_size == prior["size"]
        ):
            offset = prior["size"]
        if offset != size:
            transfers.append({**item, "offset": offset})
    return transfers


def _delta_archive_script(transfers: list[dict[str, Any]]) -> str:
    """Build a streaming tar program carrying only the requested log suffixes."""
    return """
import io
import json
import sys
import tarfile
from pathlib import Path

root = Path('/tmp/ray/session_latest/logs')
# The transfer plan arrives over stdin, not as a ``python -c`` argument.  A
# first sync can contain hundreds of files; putting that JSON in the kubectl
# exec command overflows the controller proxy's request-header limit.
requests = json.load(sys.stdin)
with tarfile.open(fileobj=sys.stdout.buffer, mode='w|') as archive:
    for request in requests:
        relative = request['path']
        source = root / relative
        stat = source.stat()
        offset = int(request['offset'])
        expected_size = int(request['size'])
        expected_inode = request.get('inode')
        if not source.is_file() or stat.st_size < expected_size:
            raise RuntimeError(f'Ray log changed while collecting: {relative}')
        if expected_inode is not None and stat.st_ino != expected_inode:
            raise RuntimeError(f'Ray log rotated while collecting: {relative}')
        info = tarfile.TarInfo(relative)
        info.size = expected_size - offset
        info.pax_headers = {
            'otagent.offset': str(offset),
            'otagent.inode': str(stat.st_ino),
        }
        with source.open('rb') as handle:
            handle.seek(offset)
            archive.addfile(info, handle)
"""


def _extract_ray_log_delta(archive: tarfile.TarFile, destination: Path) -> None:
    for member in archive:
        if not member.isfile():
            raise RuntimeError(f"Refusing non-file Ray/vLLM archive member {member.name!r}")
        offset_text = member.pax_headers.get("otagent.offset")
        if offset_text is None:
            raise RuntimeError(f"Ray/vLLM delta member {member.name!r} has no append offset")
        try:
            offset = int(offset_text)
        except ValueError as error:
            raise RuntimeError(f"Invalid append offset for Ray/vLLM log {member.name!r}") from error
        target = destination / _safe_ray_log_path(member.name)
        target.parent.mkdir(parents=True, exist_ok=True)
        source = archive.extractfile(member)
        if source is None:
            raise RuntimeError(f"Could not extract Ray/vLLM log member {member.name!r}")
        if offset:
            if not target.is_file() or target.stat().st_size != offset:
                raise RuntimeError(f"Local Ray/vLLM log changed before append: {member.name!r} at offset {offset}")
            mode = "ab"
        else:
            mode = "wb"
        with target.open(mode) as output:
            while chunk := source.read(1024 * 1024):
                output.write(chunk)


def save_ray_logs(
    base: list[str],
    pod: str,
    container: str,
    inventory: list[dict[str, Any]],
    max_bytes: int,
    destination: Path,
    *,
    incremental: bool = False,
    python_executable: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected = [item for item in inventory if item["size"] <= max_bytes]
    skipped = [item for item in inventory if item["size"] > max_bytes]
    if not selected:
        return selected, skipped

    transfers = plan_ray_log_delta(selected, destination) if incremental else selected
    if not transfers:
        _write_ray_log_sync_manifest(destination, selected)
        return selected, skipped

    archive_error: tarfile.TarError | None = None
    stderr = ""
    return_code = 0
    for attempt in range(DNS_ATTEMPTS):
        remote_command = (
            [
                python_executable or resolve_container_python(base, pod, container),
                "-c",
                _delta_archive_script(transfers),
            ]
            if incremental
            else ["tar", "-C", RAY_LOG_DIR, "-cf", "-", *(item["path"] for item in transfers)]
        )
        exec_args = ["exec", *(["-i"] if incremental else []), pod, "-c", container, "--", *remote_command]
        process = subprocess.Popen(
            [*base, "-n", NAMESPACE, *exec_args],
            stdin=subprocess.PIPE if incremental else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if incremental:
            assert process.stdin is not None
            process.stdin.write(json.dumps(transfers).encode())
            process.stdin.close()
        assert process.stdout is not None
        archive_error = None
        try:
            with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
                if incremental:
                    _extract_ray_log_delta(archive, destination)
                else:
                    for member in archive:
                        archive.extract(member, destination, filter="data")
        except tarfile.TarError as error:
            archive_error = error
        stderr = process.stderr.read().decode() if process.stderr else ""
        return_code = process.wait()
        if not archive_error and return_code == 0:
            if incremental:
                _write_ray_log_sync_manifest(destination, selected)
            return selected, skipped
        if not is_transient_kubectl_exec_failure(stderr) or attempt == DNS_ATTEMPTS - 1:
            break
        time.sleep(DNS_INITIAL_BACKOFF * 2**attempt)

    error_path = destination / "ray-vllm-sync-error.txt"
    error_path.write_text(stderr or "no stderr captured\n")
    raise RuntimeError(
        f"Could not archive Ray/vLLM logs (exit={return_code}; diagnostics saved to {error_path})"
    ) from archive_error


def is_transient_kubectl_exec_failure(stderr: str) -> bool:
    """Return whether a failed pod exec looks like a retryable transport failure."""
    message = stderr.lower()
    return any(marker in message for marker in TRANSIENT_KUBECTL_EXEC_MARKERS)


def sync_trials(
    client: Any,
    bucket: str,
    prefix: str,
    selected_trials: list[str],
    max_file_bytes: int,
    destination: Path,
) -> tuple[int, list[dict[str, Any]]]:
    copied = 0
    skipped: list[dict[str, Any]] = []
    root_prefix = f"{prefix.rstrip('/')}/"
    for trial in selected_trials:
        for item in iter_objects(client, bucket, f"{root_prefix}{trial}/"):
            key = item["Key"]
            if item["Size"] > max_file_bytes:
                skipped.append({"key": key, "size": item["Size"], "reason": "size_limit"})
                continue
            local_path = destination / safe_relative_path(key, root_prefix)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, key, str(local_path))
            copied += 1
    return copied, skipped


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cluster = COREWEAVE_CLUSTERS[args.cluster]
    base = kubectl_base(cluster, args)
    pod = find_pod(base, args)
    command_line = pod_command_line(base, pod, args.container)
    trials_uri = args.trials_s3 or discover_trials_prefix(command_line)
    bucket, prefix = split_s3_uri(trials_uri)

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    collection_dir = args.output_dir / "diagnostics" / f"{task_short_name(args.job)}-{timestamp}"
    collection_dir.mkdir(parents=True, exist_ok=False)
    (collection_dir / "vllm-logs").mkdir()
    (collection_dir / "trials").mkdir()
    manifest: dict[str, Any] = {
        "collection_started_at": datetime.now(UTC).isoformat(),
        "job": args.job,
        "cluster": args.cluster,
        "pod": pod,
        "container": args.container,
        "trials_s3": trials_uri,
        "sample_size_requested": args.sample_size,
        "seed": args.seed,
        "max_trial_file_bytes": args.max_trial_file_bytes,
        "max_vllm_log_bytes": args.max_vllm_log_bytes,
    }
    write_json(collection_dir / "manifest.json", manifest)

    LOGGER.info("Collecting task and Ray/vLLM logs from %s", pod)
    save_task_log(base, pod, args.container, collection_dir / "pod-task-tail.log")
    inventory = ray_log_inventory(base, pod, args.container)
    saved_logs, skipped_logs = save_ray_logs(
        base, pod, args.container, inventory, args.max_vllm_log_bytes, collection_dir / "vllm-logs"
    )
    manifest["ray_logs"] = {"saved": saved_logs, "skipped": skipped_logs}
    write_json(collection_dir / "manifest.json", manifest)

    LOGGER.info("Listing completed trials under s3://%s/%s", bucket, prefix)
    client = object_store_client(base, cluster)
    available_trials = completed_trials(client, bucket, prefix)
    sample_size = min(args.sample_size, len(available_trials))
    selected_trials = sorted(random.Random(args.seed).sample(available_trials, sample_size))
    LOGGER.info("Syncing %d of %d completed trials", len(selected_trials), len(available_trials))
    copied, skipped_objects = sync_trials(
        client,
        bucket,
        prefix,
        selected_trials,
        args.max_trial_file_bytes,
        collection_dir / "trials",
    )
    manifest.update(
        {
            "collection_completed_at": datetime.now(UTC).isoformat(),
            "completed_trials_available": len(available_trials),
            "selected_trials": selected_trials,
            "trial_objects_copied": copied,
            "trial_objects_skipped": skipped_objects,
        }
    )
    write_json(collection_dir / "manifest.json", manifest)
    print(collection_dir)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, subprocess.SubprocessError) as error:
        LOGGER.error("%s", error)
        sys.exit(1)
