"""Region-local caching for immutable Hugging Face model snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import posixpath
import tempfile
import time

from fsspec.spec import AbstractFileSystem
from huggingface_hub import snapshot_download
from rigging.filesystem.cluster_config import marin_temp_bucket
from rigging.filesystem.distributed_lock import create_lock, lease_refresh

from cloud.iris.artifacts import ArtifactSource, fs_and_path, materialize, read_json, write_json
from marinskyrl.hf_model import hugging_face_hub_online, immutable_model_cache_key, validate_hf_model_weights
from marinskyrl.resource_locator import is_hugging_face_repo_id, join_resource_path
from marinskyrl.speculative_decoding import is_hugging_face_commit

_CACHE_COMPLETE_MARKER = ".marinskyrl-cache.json"
_CACHE_PREFIX = "marinskyrl/hf-models"
_CACHE_POLL_INTERVAL = 10.0


@dataclass(frozen=True)
class CachedHuggingFaceModel:
    """An immutable Hub model to mirror and materialize before Ray starts."""

    model_id: str
    revision: str
    local_path: str

    def __post_init__(self) -> None:
        if not is_hugging_face_repo_id(self.model_id):
            raise ValueError(f"Invalid Hugging Face model ID: {self.model_id!r}")
        if not is_hugging_face_commit(self.revision):
            raise ValueError("Cached Hugging Face models require a full lowercase commit SHA")
        if not Path(self.local_path).is_absolute():
            raise ValueError("Cached Hugging Face models require an absolute local path")


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


def _upload_snapshot(filesystem: AbstractFileSystem, cache_path: str, snapshot: Path) -> None:
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
    """Download a Hub snapshot while preserving the runtime's offline environment."""
    with hugging_face_hub_online():
        snapshot = snapshot_download(
            model_id,
            revision=revision,
            local_dir=destination,
            allow_patterns=list(allow_patterns) or None,
        )
    return Path(snapshot)


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
        prefix=f"{_CACHE_PREFIX}/{immutable_model_cache_key(model_id, revision)}",
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
        with lease_refresh(lock):
            with tempfile.TemporaryDirectory(prefix="marinskyrl-hf-model-") as scratch:
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
