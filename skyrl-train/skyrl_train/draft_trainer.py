"""Dedicated online draft training and bounded node-to-node artifact transport."""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
import hashlib
import os
from pathlib import Path, PurePosixPath
import shutil
from typing import Any
from uuid import uuid4

import ray

from marinskyrl.hf_model import sha256_file
from skyrl_train.inference_engines.vllm.online_eagle_trainer import (
    ONLINE_EAGLE_SCRATCH_ROOT,
    TRAINER_STATE_FILENAME,
    OnlineEagleTrainingJob,
    OnlineEagleUpdateResult,
    preserve_online_eagle_failure,
    publish_speculator_checkpoint,
    publish_online_eagle_failure_bundle,
    remove_online_eagle_scratch,
    restore_speculator_checkpoint,
    run_training_job,
)


DIRECTORY_BUNDLE_FORMAT = "marinskyrl-ray-directory-bundle"
DIRECTORY_BUNDLE_VERSION = 1
OBJECT_STORE_CHUNK_BYTES = 64 * 1024 * 1024
IGNORED_CACHE_DIRECTORY = ".cache"


def _relative_bundle_path(value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("DraftTrainer bundle paths must be nonempty strings")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"DraftTrainer bundle path must be a normalized relative path: {value!r}")
    return Path(*path.parts)


def bundle_directory_for_ray(
    source: str | Path,
    *,
    put: Callable[[bytes], Any] = ray.put,
    chunk_bytes: int = OBJECT_STORE_CHUNK_BYTES,
    excluded_relative_paths: Collection[str] = (),
) -> dict[str, Any]:
    """Put one immutable directory into bounded Ray objects without driver materialization."""
    root = Path(source)
    if not root.is_dir():
        raise FileNotFoundError(f"DraftTrainer bundle source is not a directory: {root}")
    if isinstance(chunk_bytes, bool) or not isinstance(chunk_bytes, int) or chunk_bytes <= 0:
        raise ValueError("DraftTrainer object-store chunk size must be a positive integer")

    excluded = {_relative_bundle_path(path) for path in excluded_relative_paths}
    files: list[dict[str, Any]] = []
    total_bytes = 0
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if relative in excluded:
            continue
        if path.is_symlink():
            raise ValueError(f"DraftTrainer bundle source contains a symbolic link: {path}")
        if not path.is_file() or IGNORED_CACHE_DIRECTORY in path.parts:
            continue
        file_hasher = hashlib.sha256()
        chunks: list[dict[str, Any]] = []
        with path.open("rb") as stream:
            while data := stream.read(chunk_bytes):
                file_hasher.update(data)
                chunks.append(
                    {
                        "bytes": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(),
                        "object_ref": put(data),
                    }
                )
        size = path.stat().st_size
        total_bytes += size
        files.append(
            {
                "path": relative.as_posix(),
                "bytes": size,
                "sha256": file_hasher.hexdigest(),
                "chunks": chunks,
            }
        )
    if not files:
        raise ValueError(f"DraftTrainer bundle source contains no files: {root}")
    return {
        "format": DIRECTORY_BUNDLE_FORMAT,
        "format_version": DIRECTORY_BUNDLE_VERSION,
        "source_name": root.name,
        "total_bytes": total_bytes,
        "files": files,
    }


def materialize_ray_directory_bundle(
    bundle: Mapping[str, Any],
    destination: str | Path,
    *,
    get: Callable[[Any], bytes] = ray.get,
) -> Path:
    """Atomically materialize and verify a directory bundle on the consuming node."""
    if bundle.get("format") != DIRECTORY_BUNDLE_FORMAT or bundle.get("format_version") != DIRECTORY_BUNDLE_VERSION:
        raise ValueError("Unsupported DraftTrainer directory bundle")
    raw_files = bundle.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise ValueError("DraftTrainer directory bundle has no file inventory")

    target = Path(destination)
    if target.exists():
        raise FileExistsError(f"DraftTrainer bundle destination already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(f".{target.name}.tmp-{uuid4().hex}")
    staging.mkdir()
    seen: set[Path] = set()
    total_bytes = 0
    try:
        for raw_file in raw_files:
            if not isinstance(raw_file, Mapping):
                raise ValueError("DraftTrainer bundle file entries must be mappings")
            relative = _relative_bundle_path(raw_file.get("path"))
            if relative in seen:
                raise ValueError(f"Duplicate DraftTrainer bundle path: {relative}")
            seen.add(relative)
            chunks = raw_file.get("chunks")
            if not isinstance(chunks, list):
                raise ValueError(f"DraftTrainer bundle file has no chunks: {relative}")
            output = staging / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            file_hasher = hashlib.sha256()
            size = 0
            with output.open("wb") as stream:
                for raw_chunk in chunks:
                    if not isinstance(raw_chunk, Mapping):
                        raise ValueError(f"DraftTrainer bundle chunk is invalid: {relative}")
                    data = get(raw_chunk.get("object_ref"))
                    if not isinstance(data, bytes):
                        raise TypeError(f"DraftTrainer object-store chunk is not bytes: {relative}")
                    if len(data) != raw_chunk.get("bytes") or hashlib.sha256(data).hexdigest() != raw_chunk.get(
                        "sha256"
                    ):
                        raise ValueError(f"DraftTrainer object-store chunk digest mismatch: {relative}")
                    stream.write(data)
                    file_hasher.update(data)
                    size += len(data)
            if size != raw_file.get("bytes") or file_hasher.hexdigest() != raw_file.get("sha256"):
                raise ValueError(f"DraftTrainer materialized file digest mismatch: {relative}")
            total_bytes += size
        if total_bytes != bundle.get("total_bytes"):
            raise ValueError("DraftTrainer materialized directory byte count mismatch")
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def validate_materialized_bundle(bundle: Mapping[str, Any], directory: str | Path) -> None:
    """Require a materialized directory to match the immutable bundle inventory."""
    root = Path(directory)
    expected = {
        str(_relative_bundle_path(item["path"])): {"bytes": item["bytes"], "sha256": item["sha256"]}
        for item in bundle["files"]
    }
    actual = {
        str(path.relative_to(root)): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(root.rglob("*"))
        if path.is_file() and IGNORED_CACHE_DIRECTORY not in path.parts
    }
    if actual != expected:
        raise ValueError(f"DraftTrainer materialized inventory mismatch: {root}")


class DraftTrainer:
    """Long-lived one-GPU owner for online draft update state and lineage."""

    def __init__(self, *, initial_draft_dir: str, initial_draft_revision: str, process_id: str):
        initial = Path(initial_draft_dir)
        if not initial.is_absolute():
            raise ValueError("DraftTrainer initial draft path must be absolute")
        if not initial_draft_revision:
            raise ValueError("DraftTrainer initial draft revision must be nonempty")
        self._initial_draft_dir = str(initial)
        self._served_draft_dir = str(initial)
        self._served_draft_revision = initial_draft_revision
        self._node_id = str(ray.get_runtime_context().get_node_id()) if ray.is_initialized() else None
        self._gpu_ids = [str(gpu_id) for gpu_id in ray.get_gpu_ids()] if ray.is_initialized() else []
        self._process_root = ONLINE_EAGLE_SCRATCH_ROOT / process_id
        self._pending_candidate_dir: str | None = None
        self._pending_draft_revision: str | None = None

    def update(self, raw_job: Mapping[str, Any], capture_bundle: Mapping[str, Any]) -> dict[str, Any]:
        """Materialize one sealed capture, train it, and return an immutable candidate bundle."""
        if self._pending_candidate_dir is not None:
            raise RuntimeError("DraftTrainer has an uncommitted candidate")
        job = OnlineEagleTrainingJob.from_mapping(raw_job)
        if job.parent_draft_revision != self._served_draft_revision:
            raise RuntimeError(
                "DraftTrainer update parent does not match its served lineage: "
                f"expected {self._served_draft_revision}, got {job.parent_draft_revision}"
            )
        capture_dir = Path(job.capture_dir)
        materialize_ray_directory_bundle(capture_bundle, capture_dir)
        validate_materialized_bundle(capture_bundle, capture_dir)
        job = OnlineEagleTrainingJob.from_mapping(
            {
                **job.to_mapping(),
                "draft_model_dir": self._served_draft_dir,
            }
        )
        try:
            result = run_training_job(job)
        except Exception as error:
            failure_dir = None
            preservation_error = None
            try:
                failure_dir = preserve_online_eagle_failure(job, error)
                published = publish_online_eagle_failure_bundle(failure_dir, job.failure_artifact_path)
                failure_artifact_path = published["path"]
            except Exception as failure_error:
                failure_artifact_path = job.failure_artifact_path
                preservation_error = f"{type(failure_error).__name__}: {failure_error}"
            result = OnlineEagleUpdateResult(
                active=True,
                accepted=False,
                step=job.step,
                error=f"{type(error).__name__}: {error}",
                failure_dir=failure_dir,
                failure_artifact_path=failure_artifact_path,
                failure_preservation_error=preservation_error,
            ).to_mapping()
            remove_online_eagle_scratch(capture_dir.parent)
            return {"result": result, "candidate_bundle": None}

        candidate_bundle = None
        if result.accepted:
            assert result.candidate_dir is not None
            assert result.draft_revision is not None
            candidate_bundle = bundle_directory_for_ray(
                result.candidate_dir,
                excluded_relative_paths={TRAINER_STATE_FILENAME},
            )
            self._pending_candidate_dir = result.candidate_dir
            self._pending_draft_revision = result.draft_revision
        remove_online_eagle_scratch(capture_dir.parent)
        return {"result": result.to_mapping(), "candidate_bundle": candidate_bundle}

    def commit(self, draft_revision: str) -> dict[str, Any]:
        """Commit the candidate after every serving engine activated the same digest."""
        if self._pending_draft_revision != draft_revision or self._pending_candidate_dir is None:
            raise RuntimeError(
                f"DraftTrainer cannot commit {draft_revision!r}; pending={self._pending_draft_revision!r}"
            )
        previous = self._served_draft_dir
        self._served_draft_dir = self._pending_candidate_dir
        self._served_draft_revision = draft_revision
        self._pending_candidate_dir = None
        self._pending_draft_revision = None
        previous_path = Path(previous).resolve()
        process_root = self._process_root.resolve()
        if previous_path != Path(self._initial_draft_dir).resolve() and previous_path.is_relative_to(process_root):
            remove_online_eagle_scratch(previous_path)
        return {"draft_revision": draft_revision, "served_draft_dir": self._served_draft_dir}

    def rollback(self, draft_revision: str) -> dict[str, Any]:
        """Discard an unserved candidate after rejection or failed collective activation."""
        if self._pending_draft_revision != draft_revision or self._pending_candidate_dir is None:
            raise RuntimeError(
                f"DraftTrainer cannot roll back {draft_revision!r}; pending={self._pending_draft_revision!r}"
            )
        candidate = self._pending_candidate_dir
        self._pending_candidate_dir = None
        self._pending_draft_revision = None
        remove_online_eagle_scratch(candidate)
        return {"draft_revision": draft_revision, "served_draft_dir": self._served_draft_dir}

    def restore(self, source: str, destination: str) -> dict[str, Any]:
        """Restore the exact paired draft and FP32 trainer state on the trainer node."""
        if self._pending_candidate_dir is not None:
            raise RuntimeError("DraftTrainer cannot restore with an uncommitted candidate")
        manifest = restore_speculator_checkpoint(source, destination)
        self._served_draft_dir = destination
        self._served_draft_revision = manifest["draft_revision"]
        return manifest

    def publish(self, destination: str, draft_revision: str, served_target_revision: str) -> dict[str, Any]:
        """Publish the full served draft and private trainer state from its owner."""
        if draft_revision != self._served_draft_revision:
            raise RuntimeError(
                f"DraftTrainer cannot publish {draft_revision!r}; served={self._served_draft_revision!r}"
            )
        return publish_speculator_checkpoint(
            self._served_draft_dir,
            destination,
            draft_revision=draft_revision,
            served_target_revision=served_target_revision,
        )

    def cleanup(self) -> dict[str, Any]:
        """Release this actor's process-scoped capture, candidate, and failure scratch."""
        remove_online_eagle_scratch(self._process_root)
        self._pending_candidate_dir = None
        self._pending_draft_revision = None
        return {"path": str(self._process_root)}

    def status(self) -> dict[str, Any]:
        return {
            "served_draft_dir": self._served_draft_dir,
            "served_draft_revision": self._served_draft_revision,
            "node_id": self._node_id,
            "gpu_ids": self._gpu_ids,
            "pending_candidate_dir": self._pending_candidate_dir,
            "pending_draft_revision": self._pending_draft_revision,
        }


def create_draft_trainer(*, initial_draft_dir: str, initial_draft_revision: str, process_id: str):
    """Create the dedicated GPU actor without embedding it in an inference placement group."""
    actor = ray.remote(num_gpus=1, max_restarts=0)(DraftTrainer)
    return actor.options(runtime_env={"env_vars": {"TORCH_COMPILE_DISABLE": "1"}}).remote(
        initial_draft_dir=initial_draft_dir,
        initial_draft_revision=initial_draft_revision,
        process_id=process_id,
    )
