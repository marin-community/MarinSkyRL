"""Select and lazily materialize packed TaskTrove Clean tasks."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import shutil
import tarfile
import tempfile
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence
from urllib.parse import quote

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq
from filelock import FileLock

from marinskyrl.task_sources import TaskTroveParquetSource, TaskTroveTagMatch

_METADATA_COLUMNS = ("source", "tags", "mode", "path", "dockerfile_id")
_REQUIRED_TYPES = {
    "source": pa.string(),
    "tags": pa.list_(pa.string()),
    "mode": pa.string(),
    "path": pa.string(),
    "dockerfile_id": pa.string(),
    "task_binary": pa.binary(),
}
_REQUIRED_TASK_FILES = ("instruction.md", "task.toml", "environment/Dockerfile")


class TaskTroveSchemaError(ValueError):
    """The packed dataset does not satisfy the TaskTrove Clean read contract."""


class EmptyTaskSelectionError(ValueError):
    """A valid TaskTrove predicate selected no tasks."""


class PackedTaskArchiveError(ValueError):
    """A selected task archive is unsafe or incomplete."""


@dataclass(frozen=True)
class PackedTaskReference:
    dataset_path: str
    dataset_uri: str
    dataset_identity: str
    verifier_ref: str
    row_group: int
    row: int
    source: str
    path: str
    dockerfile_id: str
    mode: str

    def stable_uri(self) -> str:
        identity = quote(self.dataset_identity, safe="")
        source = quote(self.source, safe="")
        path = quote(self.path, safe="/")
        return f"tasktrove://{identity}/{source}/{path}"

    def uid(self) -> str:
        return f"{self.dataset_identity}/{self.source}/{self.path}"


@dataclass(frozen=True)
class TaskSelectionSummary:
    references: tuple[PackedTaskReference, ...]
    digest: str
    distinct_environment_count: int


def _validate_schema(schema: pa.Schema) -> None:
    for name, expected in _REQUIRED_TYPES.items():
        index = schema.get_field_index(name)
        if index < 0:
            raise TaskTroveSchemaError(f"TaskTrove dataset is missing required column {name!r}")
        actual = schema.field(index).type
        if actual != expected:
            raise TaskTroveSchemaError(f"TaskTrove column {name!r} must have type {expected}, found {actual}")


def _row_matches(source: str, tags: tuple[str, ...], mode: str, selection) -> bool:
    if selection.sources and source not in selection.sources:
        return False
    if selection.modes and mode not in selection.modes:
        return False
    if not selection.tags:
        return True
    row_tags = set(tags)
    requested = set(selection.tags)
    if selection.tag_match is TaskTroveTagMatch.ALL:
        return requested <= row_tags
    return bool(requested & row_tags)


def _limit_key(reference: PackedTaskReference, seed: int) -> bytes:
    value = f"{seed}\0{reference.source}\0{reference.path}".encode()
    return hashlib.sha256(value).digest()


def _selection_digest(references: Iterable[PackedTaskReference]) -> str:
    digest = hashlib.sha256()
    for reference in references:
        identity = {
            "dataset_identity": reference.dataset_identity,
            "dockerfile_id": reference.dockerfile_id,
            "mode": reference.mode,
            "path": reference.path,
            "row": reference.row,
            "row_group": reference.row_group,
            "source": reference.source,
        }
        digest.update(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


@contextmanager
def _parquet_file(path: str):
    if "://" not in path or path.startswith("file://"):
        yield pq.ParquetFile(path.removeprefix("file://"))
        return
    storage_options = {}
    if path.startswith(("s3://", "s3a://")):
        style = os.environ.get("OT_AGENT_S3_ADDRESSING_STYLE", "virtual")
        storage_options = {"config_kwargs": {"s3": {"addressing_style": style}}}
    with fsspec.open(path, "rb", **storage_options) as handle:
        yield pq.ParquetFile(handle)


def select_task_references(
    source: TaskTroveParquetSource,
    *,
    dataset_path: str | None = None,
) -> TaskSelectionSummary:
    """Return deterministic metadata matches without reading task payload columns."""
    path = dataset_path or source.uri
    references: list[PackedTaskReference] = []
    observed_sources: set[str] = set()
    observed_tags: set[str] = set()
    with _parquet_file(path) as parquet:
        _validate_schema(parquet.schema_arrow)
        for row_group in range(parquet.num_row_groups):
            rows = parquet.read_row_group(row_group, columns=list(_METADATA_COLUMNS)).to_pylist()
            for row_index, row in enumerate(rows):
                row_source = row["source"]
                row_tags = tuple(row["tags"])
                row_mode = row["mode"]
                observed_sources.add(row_source)
                observed_tags.update(row_tags)
                if not _row_matches(row_source, row_tags, row_mode, source.selection):
                    continue
                references.append(
                    PackedTaskReference(
                        dataset_path=path,
                        dataset_uri=source.uri,
                        dataset_identity=source.identity,
                        verifier_ref=source.verifier_ref,
                        row_group=row_group,
                        row=row_index,
                        source=row_source,
                        path=row["path"],
                        dockerfile_id=row["dockerfile_id"],
                        mode=row_mode,
                    )
                )
    missing_sources = set(source.selection.sources) - observed_sources
    if missing_sources:
        raise EmptyTaskSelectionError(f"Unknown TaskTrove sources: {sorted(missing_sources)}")
    missing_tags = set(source.selection.tags) - observed_tags
    if missing_tags:
        raise EmptyTaskSelectionError(f"Unknown TaskTrove tags: {sorted(missing_tags)}")
    if not references:
        raise EmptyTaskSelectionError("TaskTrove selection matched no tasks")
    if source.selection.limit is not None:
        references = sorted(references, key=lambda value: _limit_key(value, source.selection.seed))[
            : source.selection.limit
        ]
    references.sort(key=lambda value: (value.row_group, value.row))
    result = tuple(references)
    return TaskSelectionSummary(
        references=result,
        digest=_selection_digest(result),
        distinct_environment_count=len({reference.dockerfile_id for reference in result}),
    )


def packed_task_reference(value: dict) -> PackedTaskReference:
    return PackedTaskReference(**value)


def _safe_archive_files(blob: bytes) -> dict[str, tuple[bytes, int]]:
    raw = gzip.decompress(blob) if blob[:2] == b"\x1f\x8b" else blob
    files: dict[str, tuple[bytes, int]] = {}
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        for member in archive.getmembers():
            path = PurePosixPath(member.name.removeprefix("./"))
            if path.is_absolute() or ".." in path.parts:
                raise PackedTaskArchiveError(f"Unsafe task archive path: {member.name!r}")
            normalized = path.as_posix()
            if path.parts and path.parts[0] == "solution":
                raise PackedTaskArchiveError("Packed TaskTrove task contains a solution")
            if not member.isfile():
                if member.isdir():
                    continue
                raise PackedTaskArchiveError(f"Task archive contains a link or special file: {member.name!r}")
            if normalized in files:
                raise PackedTaskArchiveError(f"Task archive contains duplicate path: {normalized!r}")
            handle = archive.extractfile(member)
            if handle is None:
                raise PackedTaskArchiveError(f"Cannot read task archive member: {member.name!r}")
            files[normalized] = (handle.read(), member.mode)
    missing = set(_REQUIRED_TASK_FILES) - files.keys()
    if missing:
        raise PackedTaskArchiveError(f"Packed task is missing required files: {sorted(missing)}")
    return files


class PackedTaskMaterializer:
    """Materialize only rollout-assigned task archives into a node-local cache."""

    def __init__(self, cache_root: Path) -> None:
        self.cache_root = cache_root

    def _task_path(self, reference: PackedTaskReference) -> Path:
        identity = hashlib.sha256(reference.dataset_identity.encode()).hexdigest()[:16]
        task = hashlib.sha256(f"{reference.source}\0{reference.path}".encode()).hexdigest()[:20]
        return self.cache_root / "tasks" / identity / task

    def _write_task(self, target: Path, blob: bytes) -> None:
        files = _safe_archive_files(blob)
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
        try:
            for relative, (content, mode) in files.items():
                destination = staging / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content)
                destination.chmod(mode & 0o777)
            (staging / ".marinskyrl-complete").write_text("1\n")
            try:
                os.replace(staging, target)
            except OSError:
                if not (target / ".marinskyrl-complete").is_file():
                    raise
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    def materialize_batch(self, references: Sequence[PackedTaskReference]) -> dict[PackedTaskReference, Path]:
        """Extract each unique assigned task, reading each needed row group once."""
        results: dict[PackedTaskReference, Path] = {}
        pending: dict[tuple[str, int], list[PackedTaskReference]] = defaultdict(list)
        for reference in dict.fromkeys(references):
            target = self._task_path(reference)
            if (target / ".marinskyrl-complete").is_file():
                results[reference] = target
            else:
                pending[(reference.dataset_path, reference.row_group)].append(reference)

        for (dataset_path, row_group), group in pending.items():
            lock_path = self.cache_root / "locks" / hashlib.sha256(
                f"{group[0].dataset_identity}\0{row_group}".encode()
            ).hexdigest()
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            with FileLock(lock_path.with_suffix(".lock")):
                remaining = [reference for reference in group if not (self._task_path(reference) / ".marinskyrl-complete").is_file()]
                if remaining:
                    rows = pq.ParquetFile(dataset_path).read_row_group(
                        row_group,
                        columns=["source", "mode", "path", "dockerfile_id", "task_binary"],
                    )
                    values = rows.to_pylist()
                    for reference in remaining:
                        row = values[reference.row]
                        actual = (row["source"], row["path"], row["dockerfile_id"], row["mode"])
                        expected = (reference.source, reference.path, reference.dockerfile_id, reference.mode)
                        if actual != expected:
                            raise PackedTaskArchiveError(
                                f"Packed task identity changed at row group {row_group}, row {reference.row}"
                            )
                        self._write_task(self._task_path(reference), row["task_binary"])
                for reference in group:
                    results[reference] = self._task_path(reference)
        return results
