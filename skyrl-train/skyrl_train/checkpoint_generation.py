"""Immutable checkpoint attempts and their single-object commit record.

Legacy checkpoints have payloads directly below ``global_step_N``. New attempts
live below ``global_step_N/_attempts/<id>``; readers never enter that prefix
until the step-level commit record points to a completed attempt.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid

from marinskyrl.checkpoint_paths import GLOBAL_STEP_PREFIX

from skyrl_train.checkpoint_listing import extract_step_from_path
from skyrl_train.hf_export_schema import TRAINER_STATE_FILENAME
from skyrl_train.io import io

ATTEMPTS_DIRECTORY = "_attempts"
COMMIT_FILENAME = "checkpoint_commit.json"
MANIFEST_FILENAME = "checkpoint_manifest.json"
SHUTDOWN_OVERLAYS_DIRECTORY = "_shutdown_buffers"
SHUTDOWN_OVERLAY_COMMIT_FILENAME = "shutdown_buffer_commit.json"
_ATTEMPT_ID = re.compile(r"^[0-9a-f]{32}$")
_STEP_DIRECTORY = re.compile(rf"{re.escape(GLOBAL_STEP_PREFIX)}\d+")
_SCHEMA_VERSION = 1


def new_attempt_path(step_path: str) -> str:
    """Return a fresh attempt path without mutating the step's committed state."""
    if not _STEP_DIRECTORY.fullmatch(os.path.basename(step_path.rstrip("/"))):
        raise ValueError(f"Not a global-step checkpoint path: {step_path}")
    return os.path.join(step_path.rstrip("/"), ATTEMPTS_DIRECTORY, uuid.uuid4().hex)


def _attempt_id(step_path: str, attempt_path: str) -> str:
    step_path = step_path.rstrip("/")
    attempt_id = os.path.basename(attempt_path.rstrip("/"))
    expected = os.path.join(step_path, ATTEMPTS_DIRECTORY, attempt_id)
    if not _ATTEMPT_ID.fullmatch(attempt_id) or attempt_path.rstrip("/") != expected:
        raise ValueError(f"Attempt path is outside step checkpoint: {attempt_path}")
    return attempt_id


def _json_bytes(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _inventory(attempt_path: str) -> dict[str, int]:
    """Inventory exact object names/sizes; S3 find returns scheme-less keys."""
    normalized_root = attempt_path.removeprefix("s3://").rstrip("/")
    prefix = f"{normalized_root}/"
    files = {}
    for path, size in io.find_files(attempt_path).items():
        normalized_path = path.removeprefix("s3://")
        if not normalized_path.startswith(prefix):
            raise ValueError(f"Checkpoint object escaped attempt prefix: {path}")
        relative = normalized_path[len(prefix) :]
        if not relative or relative.startswith("/") or ".." in relative.split("/"):
            raise ValueError(f"Invalid checkpoint object name: {path}")
        files[relative] = int(size)
    return files


def commit_attempt(step_path: str, attempt_path: str, *, required_files: set[str]) -> dict:
    """Publish an attempt only after required files and its inventory are durable.

    A failure before the last write leaves the prior commit record untouched.
    Replacing an existing record is allowed so a replayed step can commit a new
    complete generation without overwriting the previous attempt's payload.
    """
    step = extract_step_from_path(step_path)
    attempt_id = _attempt_id(step_path, attempt_path)
    files = _inventory(attempt_path)
    missing = sorted(required_files - files.keys())
    empty = sorted(path for path in required_files if path in files and files[path] <= 0)
    if missing or empty:
        raise RuntimeError(f"Checkpoint attempt incomplete: missing={missing}, empty={empty}")
    manifest = {
        "schema_version": _SCHEMA_VERSION,
        "step": step,
        "attempt_id": attempt_id,
        "created_at_unix_seconds": time.time(),
        "required_files": sorted(required_files),
        "files": dict(sorted(files.items())),
    }
    manifest_bytes = _json_bytes(manifest)
    io.write_bytes_atomic(os.path.join(attempt_path, MANIFEST_FILENAME), manifest_bytes)
    commit = {
        "schema_version": _SCHEMA_VERSION,
        "step": step,
        "attempt_id": attempt_id,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    }
    io.write_bytes_atomic(os.path.join(step_path, COMMIT_FILENAME), _json_bytes(commit))
    return commit


def resolve_checkpoint_payload(step_path: str, *, verify_files: bool = False) -> str:
    """Resolve a committed attempt, or a pre-versioning legacy checkpoint.

    An uncommitted new attempt is never treated as a legacy checkpoint merely
    because its step prefix exists.
    """
    commit_path = os.path.join(step_path, COMMIT_FILENAME)
    if not io.exists(commit_path):
        if io.exists(os.path.join(step_path, TRAINER_STATE_FILENAME)):
            return step_path
        raise FileNotFoundError(f"No committed checkpoint at {step_path}")

    commit = json.loads(io.read_bytes(commit_path))
    if commit.get("schema_version") != _SCHEMA_VERSION or commit.get("step") != extract_step_from_path(step_path):
        raise ValueError(f"Invalid checkpoint commit record at {commit_path}")
    attempt_id = commit.get("attempt_id")
    if not isinstance(attempt_id, str) or not _ATTEMPT_ID.fullmatch(attempt_id):
        raise ValueError(f"Invalid checkpoint attempt ID at {commit_path}")
    attempt_path = os.path.join(step_path, ATTEMPTS_DIRECTORY, attempt_id)
    manifest_bytes = io.read_bytes(os.path.join(attempt_path, MANIFEST_FILENAME))
    if hashlib.sha256(manifest_bytes).hexdigest() != commit.get("manifest_sha256"):
        raise ValueError(f"Checkpoint manifest digest mismatch at {attempt_path}")
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema_version") != _SCHEMA_VERSION or manifest.get("step") != commit["step"]:
        raise ValueError(f"Invalid checkpoint manifest at {attempt_path}")
    if manifest.get("attempt_id") != attempt_id:
        raise ValueError(f"Checkpoint attempt mismatch at {attempt_path}")
    required_files = manifest.get("required_files")
    files = manifest.get("files")
    if not isinstance(required_files, list) or not isinstance(files, dict):
        raise ValueError(f"Invalid checkpoint inventory at {attempt_path}")
    if any(not isinstance(path, str) or files.get(path, 0) <= 0 for path in required_files):
        raise ValueError(f"Checkpoint manifest omits a required file at {attempt_path}")
    if verify_files:
        actual_files = _inventory(attempt_path)
        for path, expected_size in files.items():
            if actual_files.get(path) != expected_size:
                raise ValueError(f"Checkpoint object missing or size changed: {attempt_path}/{path}")
    return attempt_path


def _payload_step_and_id(payload_path: str) -> tuple[str, str]:
    payload_path = payload_path.rstrip("/")
    if os.path.basename(os.path.dirname(payload_path)) == ATTEMPTS_DIRECTORY:
        step_path = os.path.dirname(os.path.dirname(payload_path))
        return step_path, _attempt_id(step_path, payload_path)
    if not _STEP_DIRECTORY.fullmatch(os.path.basename(payload_path)):
        raise ValueError(f"Not a checkpoint payload path: {payload_path}")
    return payload_path, "legacy"


def new_shutdown_overlay_path(payload_path: str) -> str:
    """Stage shutdown-only buffer state outside the immutable model generation."""
    step_path, _ = _payload_step_and_id(payload_path)
    return os.path.join(step_path, SHUTDOWN_OVERLAYS_DIRECTORY, uuid.uuid4().hex)


def commit_shutdown_overlay(payload_path: str, overlay_path: str, artifact_name: str) -> None:
    """Atomically select a complete shutdown buffer for one committed base."""
    step_path, base_id = _payload_step_and_id(payload_path)
    overlay_id = os.path.basename(overlay_path.rstrip("/"))
    if not _ATTEMPT_ID.fullmatch(overlay_id) or overlay_path.rstrip("/") != os.path.join(
        step_path, SHUTDOWN_OVERLAYS_DIRECTORY, overlay_id
    ):
        raise ValueError(f"Shutdown overlay path is outside checkpoint step: {overlay_path}")
    if os.path.basename(artifact_name) != artifact_name:
        raise ValueError(f"Invalid shutdown overlay artifact name: {artifact_name}")
    artifact_path = os.path.join(overlay_path, artifact_name)
    size = io.file_size(artifact_path)
    if size <= 0:
        raise RuntimeError(f"Shutdown overlay is empty: {artifact_path}")
    marker = {
        "schema_version": _SCHEMA_VERSION,
        "step": extract_step_from_path(step_path),
        "base_attempt_id": base_id,
        "overlay_id": overlay_id,
        "artifact_name": artifact_name,
        "artifact_size": size,
    }
    io.write_bytes_atomic(os.path.join(step_path, SHUTDOWN_OVERLAY_COMMIT_FILENAME), _json_bytes(marker))


def shutdown_buffer_artifact_path(payload_path: str, artifact_name: str) -> str:
    """Return the matching committed shutdown overlay, or the inline state."""
    if extract_step_from_path(payload_path) < 0:
        return os.path.join(payload_path, artifact_name)
    step_path, base_id = _payload_step_and_id(payload_path)
    marker_path = os.path.join(step_path, SHUTDOWN_OVERLAY_COMMIT_FILENAME)
    inline_path = os.path.join(payload_path, artifact_name)
    if not io.exists(marker_path):
        return inline_path
    marker = json.loads(io.read_bytes(marker_path))
    if marker.get("schema_version") != _SCHEMA_VERSION or marker.get("step") != extract_step_from_path(step_path):
        raise ValueError(f"Invalid shutdown overlay marker at {marker_path}")
    if marker.get("base_attempt_id") != base_id:
        return inline_path
    overlay_id = marker.get("overlay_id")
    if not isinstance(overlay_id, str) or not _ATTEMPT_ID.fullmatch(overlay_id):
        raise ValueError(f"Invalid shutdown overlay ID at {marker_path}")
    if marker.get("artifact_name") != artifact_name:
        raise ValueError(f"Shutdown overlay artifact mismatch at {marker_path}")
    artifact_path = os.path.join(step_path, SHUTDOWN_OVERLAYS_DIRECTORY, overlay_id, artifact_name)
    if io.file_size(artifact_path) != marker.get("artifact_size"):
        raise ValueError(f"Shutdown overlay object missing or size changed: {artifact_path}")
    return artifact_path
