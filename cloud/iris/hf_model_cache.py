"""Region-local caching for immutable Hugging Face model snapshots."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import posixpath
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Generator

from rigging.filesystem.cluster_config import marin_temp_bucket
from rigging.filesystem.distributed_lock import HEARTBEAT_INTERVAL, DistributedLease, LeaseLostError, create_lock

from cloud.iris.artifacts import ArtifactSource, fs_and_path, materialize, read_json, write_json
from marinskyrl.hf_model import validate_hf_model_weights
from marinskyrl.resource_locator import join_resource_path
from marinskyrl.speculative_decoding import is_hugging_face_commit

_CACHE_COMPLETE_MARKER = ".marinskyrl-cache.json"
_CACHE_PREFIX = "marinskyrl/hf-models"
_CACHE_POLL_INTERVAL = 10.0
_DOWNLOAD_ATTEMPT_TIMEOUT = 600
_DOWNLOAD_ATTEMPTS = 6


@dataclass(frozen=True)
class CachedHuggingFaceModel:
    """An immutable Hub model to mirror and materialize before Ray starts."""

    model_id: str
    revision: str
    local_path: str

    def __post_init__(self) -> None:
        if self.model_id.count("/") != 1:
            raise ValueError(f"Invalid Hugging Face model ID: {self.model_id!r}")
        if not is_hugging_face_commit(self.revision):
            raise ValueError("Cached Hugging Face models require a full lowercase commit SHA")
        if not Path(self.local_path).is_absolute():
            raise ValueError("Cached Hugging Face models require an absolute local path")


def _cache_slug(model_id: str, revision: str) -> str:
    identity = f"{model_id}@{revision}"
    readable = identity.replace("/", "--")
    digest = hashlib.sha256(identity.encode()).hexdigest()[:16]
    return f"{readable}-{digest}"


@contextlib.contextmanager
def _heartbeat(lock: DistributedLease) -> Generator[None, None, None]:
    stop = threading.Event()
    error: list[BaseException] = []

    def refresh() -> None:
        while not stop.wait(HEARTBEAT_INTERVAL):
            try:
                lock.refresh()
            except BaseException as exc:
                error.append(exc)
                stop.set()

    thread = threading.Thread(target=refresh, name="hf-model-cache-heartbeat", daemon=True)
    thread.start()
    try:
        yield
        if error:
            raise LeaseLostError("lost the Hugging Face model-cache lease") from error[0]
    finally:
        stop.set()
        thread.join()


def _cache_metadata(marker_uri: str) -> dict[str, object] | None:
    value = read_json(marker_uri)
    if not isinstance(value, dict):
        if value is None:
            return None
        raise ValueError(f"Invalid Hugging Face model-cache marker: {marker_uri}")
    return value


def _is_cache_complete(marker_uri: str, model_id: str, revision: str) -> bool:
    metadata = _cache_metadata(marker_uri)
    if metadata is None:
        return False
    expected = {"model_id": model_id, "revision": revision}
    if metadata != expected:
        raise ValueError(f"Hugging Face model-cache identity mismatch at {marker_uri}: {metadata!r} != {expected!r}")
    return True


def _upload_snapshot(filesystem, cache_path: str, snapshot: Path) -> None:
    for local_path in sorted(path for path in snapshot.rglob("*") if path.is_file()):
        relative = local_path.relative_to(snapshot)
        if relative.parts[0] == ".cache":
            continue
        destination = posixpath.join(cache_path, relative.as_posix())
        parent = posixpath.dirname(destination)
        if parent:
            filesystem.makedirs(parent, exist_ok=True)
        filesystem.put_file(str(local_path), destination)


def download_hugging_face_snapshot(
    model_id: str,
    *,
    revision: str | None,
    destination: Path | None = None,
    allow_patterns: tuple[str, ...] = (),
) -> Path:
    """Download a Hub snapshot in an online child process and return its local path."""
    child_env = {
        key: value for key, value in os.environ.items() if key not in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    }
    child_env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    code = (
        "import json, sys\n"
        "from huggingface_hub import snapshot_download\n"
        "p = snapshot_download(\n"
        "    sys.argv[1],\n"
        "    revision=sys.argv[2] or None,\n"
        "    local_dir=sys.argv[3] or None,\n"
        "    allow_patterns=json.loads(sys.argv[4]) or None,\n"
        ")\n"
        "print('PRESTAGE_LOCAL_DIR=' + p)\n"
    )
    last_error = ""
    for attempt in range(1, _DOWNLOAD_ATTEMPTS + 1):
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    code,
                    model_id,
                    revision or "",
                    str(destination) if destination is not None else "",
                    json.dumps(allow_patterns),
                ],
                env=child_env,
                capture_output=True,
                text=True,
                timeout=_DOWNLOAD_ATTEMPT_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            last_error = f"download stalled for more than {_DOWNLOAD_ATTEMPT_TIMEOUT} seconds"
        else:
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    if line.startswith("PRESTAGE_LOCAL_DIR="):
                        return Path(line.split("=", 1)[1])
                last_error = "snapshot_download did not report its local directory"
            else:
                last_error = (result.stderr or result.stdout or "unknown error")[-800:]
        if attempt < _DOWNLOAD_ATTEMPTS:
            time.sleep(min(30, 2**attempt))
    raise RuntimeError(f"Hugging Face snapshot download failed for {model_id}@{revision}: {last_error}")


def ensure_hugging_face_model_cache(
    model_id: str,
    revision: str,
    *,
    ttl_days: int,
    source_prefix: str,
) -> str:
    """Mirror one immutable Hub snapshot and return its region-local cache URI."""
    cache_uri = marin_temp_bucket(
        ttl_days,
        prefix=f"{_CACHE_PREFIX}/{_cache_slug(model_id, revision)}",
        source_prefix=source_prefix,
    ).rstrip("/")
    filesystem, cache_path = fs_and_path(cache_uri)
    marker_uri = join_resource_path(cache_uri, _CACHE_COMPLETE_MARKER)
    if _is_cache_complete(marker_uri, model_id, revision):
        return cache_uri

    lock = create_lock(f"{cache_uri}.lock")
    while not lock.try_acquire():
        if _is_cache_complete(marker_uri, model_id, revision):
            return cache_uri
        time.sleep(_CACHE_POLL_INTERVAL)

    try:
        if _is_cache_complete(marker_uri, model_id, revision):
            return cache_uri
        with _heartbeat(lock), tempfile.TemporaryDirectory(prefix="marinskyrl-hf-model-") as scratch:
            snapshot = download_hugging_face_snapshot(model_id, revision=revision, destination=Path(scratch))
            filesystem.makedirs(cache_path, exist_ok=True)
            _upload_snapshot(filesystem, cache_path, snapshot)
            write_json(marker_uri, {"model_id": model_id, "revision": revision})
        return cache_uri
    finally:
        lock.release()


def stage_cached_hugging_face_model(
    model: CachedHuggingFaceModel,
    *,
    ttl_days: int,
    source_prefix: str,
) -> None:
    """Mirror one Hub model once, then materialize it on the current node."""
    cache_uri = ensure_hugging_face_model_cache(
        model.model_id,
        model.revision,
        ttl_days=ttl_days,
        source_prefix=source_prefix,
    )
    materialize(
        ArtifactSource(uri=cache_uri, identity=model.revision, local_path=model.local_path),
        validate=validate_hf_model_weights,
    )
