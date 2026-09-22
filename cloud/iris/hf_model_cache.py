"""Immutable, manifest-verified Hugging Face mirrors for Iris model loading."""

from __future__ import annotations

import json
import os
from pathlib import Path
import posixpath
import tempfile
import time
from typing import Literal
from urllib.parse import urlsplit

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


def _s3_endpoint_environment() -> str:
    """Classify the environment hint, never the storage client's actual endpoint."""
    endpoint = os.environ.get("AWS_ENDPOINT_URL")
    if "FSSPEC_S3" in os.environ:
        try:
            settings = json.loads(os.environ["FSSPEC_S3"])
        except (TypeError, ValueError):
            return "unknown"
        if not isinstance(settings, dict):
            return "unknown"
        if settings.get("endpoint_url") not in (None, ""):
            endpoint = settings["endpoint_url"]

    if endpoint in (None, ""):
        return "unset"
    if not isinstance(endpoint, str):
        return "unknown"
    try:
        parsed = urlsplit(endpoint)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            return "unknown"
    except ValueError:
        return "unknown"
    return "coreweave_in_cluster" if parsed.hostname == "cwlota.com" else "other"


def load_model_manifest(model_uri: str) -> ModelManifest:
    marker_uri = join_resource_path(model_uri, MODEL_MANIFEST_FILENAME)
    return ModelManifest.from_mapping(read_json(marker_uri), marker_uri)


def _cached_manifest(
    cache_uri: str,
    model_id: str,
    revision: str,
    tokenizer_mode: Literal["embedded", "policy"],
) -> ModelManifest | None:
    marker_uri = join_resource_path(cache_uri, MODEL_MANIFEST_FILENAME)
    filesystem, marker_path = fs_and_path(marker_uri)
    if not filesystem.exists(marker_path):
        return None
    # The manifest is the completion record. A malformed or mismatched marker
    # identifies a partial cache that the next lock holder can rebuild.
    try:
        manifest = load_model_manifest(cache_uri)
    except ValueError:
        return None
    if manifest.model_id != model_id or manifest.revision != revision or manifest.tokenizer_mode != tokenizer_mode:
        return None
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
    tokenizer_mode: Literal["embedded", "policy"] = "embedded",
) -> tuple[str, ModelManifest]:
    """Mirror one immutable Hub snapshot once and return its URI and manifest."""
    started = time.monotonic()
    revision = resolve_hugging_face_revision(model_id, revision)

    def log(message: str) -> None:
        print(f"[hf-cache] model={model_id} revision={revision} {message}", flush=True)

    cache_identity = model_id if tokenizer_mode == "embedded" else f"{model_id}#tokenizer={tokenizer_mode}"
    cache_uri = marin_temp_bucket(
        ttl_days,
        prefix=f"{_CACHE_PREFIX}/{immutable_model_cache_key(cache_identity, revision)}",
        source_prefix=source_prefix,
    ).rstrip("/")
    filesystem, cache_path = fs_and_path(cache_uri)
    if manifest := _cached_manifest(cache_uri, model_id, revision, tokenizer_mode):
        log(f"role=hit total_seconds={time.monotonic() - started:.3f}")
        return cache_uri, manifest

    lock = create_lock(f"{cache_uri}.lock")
    waited = False
    while not lock.try_acquire():
        if not waited:
            log("role=waiting")
            waited = True
        if manifest := _cached_manifest(cache_uri, model_id, revision, tokenizer_mode):
            log(f"role=waiter_hit total_seconds={time.monotonic() - started:.3f}")
            return cache_uri, manifest
        time.sleep(_CACHE_POLL_INTERVAL)

    try:
        if manifest := _cached_manifest(cache_uri, model_id, revision, tokenizer_mode):
            log(f"role=lock_hit total_seconds={time.monotonic() - started:.3f}")
            return cache_uri, manifest
        log(
            f"role=publisher hf_token_present={bool(os.environ.get('HF_TOKEN'))} "
            f"s3_endpoint_env_class={_s3_endpoint_environment()}"
        )
        with lease_refresh(lock):
            if filesystem.exists(cache_path):
                filesystem.rm(cache_path, recursive=True)
            with tempfile.TemporaryDirectory(prefix="marinskyrl-hf-model-") as scratch:
                log("phase=download started")
                phase_started = time.monotonic()
                snapshot = download_hugging_face_snapshot(model_id, revision=revision, destination=Path(scratch))
                log(f"phase=download seconds={time.monotonic() - phase_started:.3f}")
                log("phase=manifest started")
                phase_started = time.monotonic()
                manifest = snapshot_model_manifest(
                    snapshot,
                    model_id,
                    revision,
                    tokenizer_mode=tokenizer_mode,
                )
                log(
                    f"phase=manifest seconds={time.monotonic() - phase_started:.3f} "
                    f"files={len(manifest.files)} bytes={sum(entry.size for entry in manifest.files)}"
                )
                log("phase=publication started")
                phase_started = time.monotonic()
                filesystem.makedirs(cache_path, exist_ok=True)
                _upload_snapshot(filesystem, cache_path, snapshot)
            write_json(join_resource_path(cache_uri, MODEL_MANIFEST_FILENAME), manifest.model_dump(mode="json"))
            log(f"phase=publication seconds={time.monotonic() - phase_started:.3f}")
        log(f"role=publisher completed total_seconds={time.monotonic() - started:.3f}")
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
