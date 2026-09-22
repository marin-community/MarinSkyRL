"""Immutable, manifest-verified Hugging Face mirrors for Iris model loading."""

from __future__ import annotations

import json
from pathlib import Path
import posixpath
import tempfile
import time

from fsspec.spec import AbstractFileSystem
from huggingface_hub import HfApi, snapshot_download
from rigging.filesystem.cluster_config import marin_temp_bucket
from rigging.filesystem.distributed_lock import create_lock, lease_refresh

from cloud.iris.artifacts import atomic_directory_update, fs_and_path, read_json, write_json
from marinskyrl.hf_model import (
    hugging_face_hub_online,
    immutable_model_cache_key,
)
from marinskyrl.model_manifest import (
    MODEL_MANIFEST_FILENAME,
    ModelManifest,
    sha256_file,
    snapshot_model_manifest,
)
from marinskyrl.resource_locator import join_resource_path
from marinskyrl.speculative_decoding import is_hugging_face_commit

_CACHE_PREFIX = "marinskyrl/hf-models"
_CACHE_POLL_INTERVAL = 10.0
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")


def load_model_manifest(model_uri: str) -> ModelManifest:
    marker_uri = join_resource_path(model_uri, MODEL_MANIFEST_FILENAME)
    return ModelManifest.from_mapping(read_json(marker_uri), marker_uri)


def _cached_manifest(cache_uri: str, model_id: str, revision: str) -> ModelManifest | None:
    marker_uri = join_resource_path(cache_uri, MODEL_MANIFEST_FILENAME)
    filesystem, marker_path = fs_and_path(marker_uri)
    if not filesystem.exists(marker_path):
        return None
    manifest = load_model_manifest(cache_uri)
    if manifest.model_id != model_id or manifest.revision != revision:
        raise ValueError(
            f"Hugging Face model-cache identity mismatch at {cache_uri}: "
            f"{manifest.model_id}@{manifest.revision} != {model_id}@{revision}"
        )
    return manifest


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


def resolve_hugging_face_revision(model_id: str, revision: str | None) -> str:
    """Resolve a branch, tag, or omitted revision to one immutable Hub commit."""
    if revision and is_hugging_face_commit(revision):
        return revision
    with hugging_face_hub_online():
        resolved = HfApi().model_info(model_id, revision=revision).sha
    if not resolved or not is_hugging_face_commit(resolved):
        raise ValueError(f"Hugging Face did not resolve {model_id}@{revision or 'main'} to a commit")
    return resolved


def ensure_hugging_face_model_cache(
    model_id: str,
    revision: str,
    *,
    ttl_days: int,
    source_prefix: str,
) -> tuple[str, ModelManifest]:
    """Mirror one immutable Hub snapshot once and return its URI and manifest."""
    revision = resolve_hugging_face_revision(model_id, revision)
    cache_uri = marin_temp_bucket(
        ttl_days,
        prefix=f"{_CACHE_PREFIX}/{immutable_model_cache_key(model_id, revision)}",
        source_prefix=source_prefix,
    ).rstrip("/")
    filesystem, cache_path = fs_and_path(cache_uri)
    if manifest := _cached_manifest(cache_uri, model_id, revision):
        return cache_uri, manifest

    lock = create_lock(f"{cache_uri}.lock")
    while not lock.try_acquire():
        if manifest := _cached_manifest(cache_uri, model_id, revision):
            return cache_uri, manifest
        time.sleep(_CACHE_POLL_INTERVAL)

    try:
        if manifest := _cached_manifest(cache_uri, model_id, revision):
            return cache_uri, manifest
        with lease_refresh(lock):
            with tempfile.TemporaryDirectory(prefix="marinskyrl-hf-model-") as scratch:
                snapshot = download_hugging_face_snapshot(model_id, revision=revision, destination=Path(scratch))
                manifest = snapshot_model_manifest(snapshot, model_id, revision)
                filesystem.makedirs(cache_path, exist_ok=True)
                _upload_snapshot(filesystem, cache_path, snapshot)
            write_json(join_resource_path(cache_uri, MODEL_MANIFEST_FILENAME), manifest.model_dump(mode="json"))
        return cache_uri, manifest
    finally:
        lock.release()


def _is_weight(path: str) -> bool:
    return path.endswith(_WEIGHT_SUFFIXES)


def stage_model_metadata(model_uri: str, manifest: ModelManifest, local_path: str) -> None:
    """Cache verified model metadata locally while excluding all weight shards."""
    target = Path(local_path)
    metadata_files = tuple(entry for entry in manifest.files if not _is_weight(entry.path))
    if not metadata_files:
        raise ValueError(f"Model manifest contains no metadata: {model_uri}")

    def matches() -> bool:
        expected_paths = {entry.path for entry in metadata_files} | {MODEL_MANIFEST_FILENAME}
        actual_paths = {path.relative_to(target).as_posix() for path in target.rglob("*") if path.is_file()}
        return actual_paths == expected_paths and all(
            (path := target / entry.path).is_file()
            and path.stat().st_size == entry.size
            and sha256_file(path) == entry.sha256
            for entry in metadata_files
        )

    if matches():
        return
    with atomic_directory_update(target, staging_prefix=f".{target.name}.metadata-") as staging:
        staging.mkdir()
        filesystem, root = fs_and_path(model_uri)
        for entry in metadata_files:
            destination = staging / entry.path
            destination.parent.mkdir(parents=True, exist_ok=True)
            filesystem.get_file(posixpath.join(root, entry.path), str(destination))
            if destination.stat().st_size != entry.size or sha256_file(destination) != entry.sha256:
                raise ValueError(f"Model metadata checksum mismatch for {entry.path}: {model_uri}")
        (staging / MODEL_MANIFEST_FILENAME).write_text(
            json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        )
