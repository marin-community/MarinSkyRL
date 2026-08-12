"""Artifact storage and materialization."""

from __future__ import annotations

import json
import os
import posixpath
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import fsspec
from fsspec.spec import AbstractFileSystem

CHECKPOINT_MARKER_FILENAME = "latest_ckpt_global_step.txt"
SOURCE_MANIFEST_FILENAME = ".marinskyrl-source.json"
S3_ADDRESSING_STYLE_ENV = "OT_AGENT_S3_ADDRESSING_STYLE"


@dataclass(frozen=True)
class ArtifactSource:
    uri: str
    identity: str
    local_path: str


@dataclass(frozen=True)
class FileEntry:
    path: str
    size: int


@dataclass(frozen=True)
class MaterializedArtifact:
    source: ArtifactSource
    files: tuple[FileEntry, ...]


def fs_and_path(uri: str) -> tuple[AbstractFileSystem, str]:
    """Resolve a URI, using virtual-hosted addressing for Marin's S3 store."""
    storage_options = None
    if uri.startswith(("s3://", "s3a://")):
        style = os.environ.get(S3_ADDRESSING_STYLE_ENV, "virtual")
        storage_options = {"config_kwargs": {"s3": {"addressing_style": style}}}
    filesystem, _, paths = fsspec.get_fs_token_paths(uri, storage_options=storage_options)
    return filesystem, paths[0]


def relative_object_key(root: str, path: str) -> str:
    """Return ``path`` below ``root`` or reject a non-descendant object key."""
    normalized_root = posixpath.normpath(root)
    normalized_path = posixpath.normpath(path)
    relative = posixpath.relpath(normalized_path, normalized_root)
    if relative == ".." or relative.startswith("../") or posixpath.isabs(relative):
        raise ValueError(f"Object key {path!r} is not below source root {root!r}")
    return relative


def _source_inventory(uri: str) -> tuple[AbstractFileSystem, tuple[tuple[str, FileEntry], ...]]:
    filesystem, source_path = fs_and_path(uri)
    source_files = sorted(path for path in filesystem.find(source_path) if not filesystem.isdir(path))
    if not source_files:
        raise ValueError(f"Artifact source contains no files: {uri}")
    inventory = tuple(
        (path, FileEntry(path=relative_object_key(source_path, path), size=int(filesystem.info(path)["size"])))
        for path in source_files
    )
    return filesystem, inventory


def _copy_inventory(
    filesystem: AbstractFileSystem,
    inventory: tuple[tuple[str, FileEntry], ...],
    destination: Path,
) -> tuple[FileEntry, ...]:
    destination.mkdir(parents=True, exist_ok=False)
    for source_path, entry in inventory:
        local_path = destination / entry.path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        filesystem.get_file(source_path, str(local_path))
        actual_size = local_path.stat().st_size
        if actual_size != entry.size:
            raise ValueError(f"Staging size mismatch for {entry.path}: expected {entry.size}, found {actual_size}")
    return tuple(entry for _, entry in inventory)


def copy_tree(source_uri: str, destination: Path) -> tuple[FileEntry, ...]:
    """Copy an object-store tree to an empty local directory and verify file sizes."""
    filesystem, inventory = _source_inventory(source_uri)
    return _copy_inventory(filesystem, inventory, destination)


def _materialization_matches(target: Path, source: ArtifactSource, inventory: tuple[FileEntry, ...]) -> bool:
    manifest_path = target / SOURCE_MANIFEST_FILENAME
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError):
        return False
    expected = {
        "source_uri": source.uri,
        "source_identity": source.identity,
        "files": [asdict(entry) for entry in inventory],
    }
    if manifest != expected:
        return False
    expected_paths = {entry.path for entry in inventory}
    local_paths = {
        path.relative_to(target).as_posix()
        for path in target.rglob("*")
        if path.is_file() and path.name != SOURCE_MANIFEST_FILENAME
    }
    return local_paths == expected_paths and all(
        (target / entry.path).stat().st_size == entry.size for entry in inventory
    )


def materialize(
    source: ArtifactSource,
    *,
    validate: Callable[[set[str], str], None] | None = None,
) -> MaterializedArtifact:
    """Materialize one immutable artifact into its declared node-local path."""
    _, remote_inventory = _source_inventory(source.uri)
    inventory = tuple(entry for _, entry in remote_inventory)
    target = Path(source.local_path).resolve()
    if _materialization_matches(target, source, inventory):
        return MaterializedArtifact(source=source, files=inventory)

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    backup: Path | None = None
    try:
        staging.rmdir()
        copied_inventory = copy_tree(source.uri, staging)
        if copied_inventory != inventory:
            raise ValueError(f"Artifact source changed while it was being staged: {source.uri}")
        if validate is not None:
            validate({entry.path for entry in inventory}, source.uri)
        manifest = {
            "source_uri": source.uri,
            "source_identity": source.identity,
            "files": [asdict(entry) for entry in inventory],
        }
        (staging / SOURCE_MANIFEST_FILENAME).write_text(json.dumps(manifest, sort_keys=True))
        if target.exists():
            backup = Path(tempfile.mkdtemp(prefix=f".{target.name}.old-", dir=target.parent))
            backup.rmdir()
            os.replace(target, backup)
        os.replace(staging, target)
    except BaseException:
        if backup is not None and backup.exists() and not target.exists():
            os.replace(backup, target)
        raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        if backup is not None and backup.exists():
            shutil.rmtree(backup)
    return MaterializedArtifact(source=source, files=inventory)
