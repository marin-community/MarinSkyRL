"""Region-local caching for immutable Hugging Face model snapshots."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import posixpath
import re
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Generator

from rigging.filesystem.cluster_config import marin_temp_bucket
from rigging.filesystem.distributed_lock import HEARTBEAT_INTERVAL, DistributedLease, LeaseLostError, create_lock

from cloud.iris.artifacts import ArtifactSource, fs_and_path, materialize

_CACHE_COMPLETE_MARKER = ".marinskyrl-cache.json"
_CACHE_PREFIX = "marinskyrl/hf-models"
_POLL_INTERVAL = 10.0
_DOWNLOAD_ATTEMPT_TIMEOUT = 600
_DOWNLOAD_ATTEMPTS = 6
_HF_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class CachedHuggingFaceModel:
    """An immutable Hub model to mirror and materialize before Ray starts."""

    model_id: str
    revision: str
    local_path: str

    def __post_init__(self) -> None:
        if self.model_id.count("/") != 1:
            raise ValueError(f"Invalid Hugging Face model ID: {self.model_id!r}")
        if _HF_COMMIT_PATTERN.fullmatch(self.revision) is None:
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


def _cache_metadata(filesystem, marker_path: str) -> dict[str, object] | None:
    if not filesystem.exists(marker_path):
        return None
    with filesystem.open(marker_path) as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"Invalid Hugging Face model-cache marker: {marker_path}")
    return value


def _completed_cache(filesystem, marker_path: str, model_id: str, revision: str) -> bool:
    metadata = _cache_metadata(filesystem, marker_path)
    if metadata is None:
        return False
    expected = {"model_id": model_id, "revision": revision}
    if metadata != expected:
        raise ValueError(f"Hugging Face model-cache identity mismatch at {marker_path}: {metadata!r} != {expected!r}")
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


def _download_snapshot(model_id: str, revision: str, destination: Path) -> None:
    """Download through a clean child process even when rollout ranks are offline."""
    child_env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    }
    child_env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    code = (
        "import sys\n"
        "from huggingface_hub import snapshot_download\n"
        "snapshot_download(sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3])\n"
    )
    last_error = ""
    for attempt in range(1, _DOWNLOAD_ATTEMPTS + 1):
        try:
            result = subprocess.run(
                [sys.executable, "-c", code, model_id, revision, str(destination)],
                env=child_env,
                capture_output=True,
                text=True,
                timeout=_DOWNLOAD_ATTEMPT_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            last_error = f"download stalled for more than {_DOWNLOAD_ATTEMPT_TIMEOUT} seconds"
        else:
            if result.returncode == 0:
                return
            last_error = (result.stderr or result.stdout or "unknown error")[-800:]
        if attempt < _DOWNLOAD_ATTEMPTS:
            time.sleep(min(30, 2**attempt))
    raise RuntimeError(f"Hugging Face snapshot download failed for {model_id}@{revision}: {last_error}")


def cache_hugging_face_model(
    model_id: str,
    revision: str,
    *,
    ttl_days: int,
    source_prefix: str,
    poll_interval: float = _POLL_INTERVAL,
) -> str:
    """Mirror one immutable Hub snapshot to the region-local TTL bucket."""
    cache_uri = marin_temp_bucket(
        ttl_days,
        prefix=f"{_CACHE_PREFIX}/{_cache_slug(model_id, revision)}",
        source_prefix=source_prefix,
    ).rstrip("/")
    filesystem, cache_path = fs_and_path(cache_uri)
    marker_path = posixpath.join(cache_path, _CACHE_COMPLETE_MARKER)
    if _completed_cache(filesystem, marker_path, model_id, revision):
        return cache_uri

    lock = create_lock(f"{cache_uri}.lock")
    while not lock.try_acquire():
        if _completed_cache(filesystem, marker_path, model_id, revision):
            return cache_uri
        time.sleep(poll_interval)

    try:
        if _completed_cache(filesystem, marker_path, model_id, revision):
            return cache_uri
        with _heartbeat(lock), tempfile.TemporaryDirectory(prefix="marinskyrl-hf-model-") as scratch:
            snapshot = Path(scratch)
            _download_snapshot(model_id, revision, snapshot)
            filesystem.makedirs(cache_path, exist_ok=True)
            _upload_snapshot(filesystem, cache_path, snapshot)
            with filesystem.open(marker_path, "w") as destination:
                json.dump({"model_id": model_id, "revision": revision}, destination, sort_keys=True)
        return cache_uri
    finally:
        lock.release()


def _validate_draft_model(names: set[str], source: str) -> None:
    if "config.json" not in names:
        raise ValueError(f"Draft model is missing config.json: {source}")
    if not any(name.endswith((".safetensors", ".bin")) for name in names):
        raise ValueError(f"Draft model has no weight shards: {source}")


def stage_cached_hugging_face_model(
    model: CachedHuggingFaceModel,
    *,
    ttl_days: int,
    source_prefix: str,
) -> None:
    """Mirror one Hub model once, then materialize it on the current node."""
    cache_uri = cache_hugging_face_model(
        model.model_id,
        model.revision,
        ttl_days=ttl_days,
        source_prefix=source_prefix,
    )
    materialize(
        ArtifactSource(uri=cache_uri, identity=model.revision, local_path=model.local_path),
        validate=_validate_draft_model,
    )
