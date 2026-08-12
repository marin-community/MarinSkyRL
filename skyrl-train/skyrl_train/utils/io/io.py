"""
File I/O utilities for handling both local filesystem and cloud storage (S3/GCS).

This module provides a unified interface for file operations that works with:
- Local filesystem paths
- S3 paths (s3://bucket/path)
- Google Cloud Storage paths (gs://bucket/path or gcs://bucket/path)

Uses fsspec for cloud storage abstraction.
"""

import json
import os
import tempfile
import time
from contextlib import contextmanager
from collections.abc import Sequence
from pathlib import Path

import fsspec
from loguru import logger
from marinskyrl.resource_locator import is_cloud_uri
from .s3fs import get_s3_fs, s3_refresh_if_expiring, call_with_s3_retry

_HF_WEIGHT_INDEX_FILENAME = "model.safetensors.index.json"


def is_cloud_path(path: str) -> bool:
    """Check if the given path is a cloud storage path."""
    return is_cloud_uri(path)


def _get_filesystem(path: str):
    """Get the appropriate filesystem for the given path."""
    if not is_cloud_path(path):
        return fsspec.filesystem("file")

    proto = path.split("://", 1)[0]
    if proto == "s3":
        fs = get_s3_fs()
        s3_refresh_if_expiring(fs)
        return fs
    return fsspec.filesystem(proto)


def open_file(path: str, mode: str = "rb"):
    """Open a file using fsspec, works with both local and cloud paths."""
    if not is_cloud_path(path):
        return fsspec.open(path, mode)

    fs = _get_filesystem(path)
    norm = fs._strip_protocol(path)
    if path.startswith("s3://"):
        return call_with_s3_retry(fs, fs.open, norm, mode)
    return fs.open(norm, mode)


def write_bytes_atomic(path: str, payload: bytes) -> None:
    """Write one object; local paths use fsync plus atomic replacement."""
    if is_cloud_path(path):
        with open_file(path, "wb") as destination:
            destination.write(payload)
        return

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_path = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(file_descriptor, "wb") as temporary:
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def makedirs(path: str, exist_ok: bool = True) -> None:
    """Create directories. Only applies to local filesystem paths."""
    if not is_cloud_path(path):
        os.makedirs(path, exist_ok=exist_ok)


def exists(path: str) -> bool:
    """Check if a file or directory exists."""
    fs = _get_filesystem(path)
    if is_cloud_path(path) and path.startswith("s3://"):
        return call_with_s3_retry(fs, fs.exists, path)
    return fs.exists(path)


def verify_hf_model_export(export_path: str) -> None:
    """Reject an HF export unless its safetensors weights are all present."""
    index_path = os.path.join(export_path, _HF_WEIGHT_INDEX_FILENAME)
    if not exists(index_path):
        unsharded_path = os.path.join(export_path, "model.safetensors")
        if exists(unsharded_path):
            return
        raise RuntimeError(f"HF export has no safetensors weights at {export_path}")

    try:
        with open_file(index_path, "r") as source:
            index = json.load(source)
    except (OSError, TypeError, ValueError) as error:
        raise RuntimeError(f"HF export has an unreadable safetensors index at {index_path}: {error}") from error

    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError(f"HF export has an empty safetensors weight map at {index_path}")

    shard_values = list(weight_map.values())
    invalid_shards = [shard for shard in shard_values if not isinstance(shard, str) or os.path.basename(shard) != shard]
    if invalid_shards:
        raise RuntimeError(f"HF export index contains invalid safetensors shard paths: {invalid_shards}")
    shards = sorted(set(shard_values))
    missing_shards = [shard for shard in shards if not exists(os.path.join(export_path, shard))]
    if missing_shards:
        raise RuntimeError(
            f"HF export is missing {len(missing_shards)} referenced safetensors shard(s): {missing_shards[:5]}"
        )


def isdir(path: str) -> bool:
    """Check if path is a directory."""
    fs = _get_filesystem(path)
    if is_cloud_path(path) and path.startswith("s3://"):
        return call_with_s3_retry(fs, fs.isdir, path)
    return fs.isdir(path)


def list_dir(path: str) -> list[str]:
    """List contents of a directory."""
    fs = _get_filesystem(path)
    if is_cloud_path(path) and path.startswith("s3://"):
        return call_with_s3_retry(fs, fs.ls, path, detail=False)
    return fs.ls(path, detail=False)


def remove(path: str) -> None:
    """Remove a file or directory."""
    fs = _get_filesystem(path)
    if is_cloud_path(path) and path.startswith("s3://"):
        if call_with_s3_retry(fs, fs.isdir, path):
            call_with_s3_retry(fs, fs.rm, path, recursive=True)
        else:
            call_with_s3_retry(fs, fs.rm, path)
        return
    if fs.isdir(path):
        fs.rm(path, recursive=True)
    else:
        fs.rm(path)


def upload_directory(local_path: str, cloud_path: str) -> None:
    """Upload a local directory to cloud storage."""
    if not is_cloud_path(cloud_path):
        raise ValueError(f"Destination must be a cloud path, got: {cloud_path}")

    fs = _get_filesystem(cloud_path)
    source_path = local_path.rstrip("/") + "/"
    if cloud_path.startswith("s3://"):
        call_with_s3_retry(fs, fs.put, source_path, fs._strip_protocol(cloud_path), recursive=True)
    else:
        fs.put(source_path, cloud_path, recursive=True)
    logger.info(f"Uploaded {local_path} to {cloud_path}")


def _upload_hf_model_directory(local_path: str, cloud_path: str) -> None:
    """Publish an HF model with weight progress and its safetensors index last."""
    if not is_cloud_path(cloud_path):
        raise ValueError(f"Destination must be a cloud path, got: {cloud_path}")
    verify_hf_model_export(local_path)

    source_root = Path(local_path)
    files = sorted(path for path in source_root.rglob("*") if path.is_file())
    index_files = [path for path in files if path.relative_to(source_root).as_posix() == _HF_WEIGHT_INDEX_FILENAME]
    weight_files = [path for path in files if path.suffix == ".safetensors"]
    other_files = [path for path in files if path not in weight_files and path not in index_files]

    filesystem = _get_filesystem(cloud_path)
    destination_root = filesystem._strip_protocol(cloud_path).rstrip("/")

    def publish_file(path: Path, shard_index: int | None = None) -> None:
        relative_path = path.relative_to(source_root).as_posix()
        destination = f"{destination_root}/{relative_path}"
        started = time.monotonic()
        if shard_index is not None:
            logger.info(
                f"Publishing HF weight shard {shard_index}/{len(weight_files)}: {relative_path} "
                f"({path.stat().st_size} bytes)"
            )
        if cloud_path.startswith("s3://"):
            call_with_s3_retry(filesystem, filesystem.put, str(path), destination, recursive=False)
        else:
            filesystem.put(str(path), destination, recursive=False)
        if shard_index is not None:
            logger.info(
                f"Published HF weight shard {shard_index}/{len(weight_files)}: {relative_path} "
                f"({path.stat().st_size} bytes in {time.monotonic() - started:.1f}s)"
            )

    for shard_index, path in enumerate(weight_files, start=1):
        publish_file(path, shard_index)
    for path in [*other_files, *index_files]:
        publish_file(path)

    logger.info(f"Published HF model directory to {cloud_path}")


def download_directory(cloud_path: str, local_path: str) -> None:
    """Download a cloud directory to local storage."""
    if not is_cloud_path(cloud_path):
        raise ValueError(f"Source must be a cloud path, got: {cloud_path}")

    fs = _get_filesystem(cloud_path)
    # The trailing separator makes fsspec copy the directory CONTENTS instead of
    # nesting the directory under the destination. It must be appended AFTER
    # _strip_protocol, which rstrips separators and would silently undo it.
    if cloud_path.startswith("s3://"):
        source_path = fs._strip_protocol(cloud_path) + "/"
        call_with_s3_retry(fs, fs.get, source_path, local_path, recursive=True)
    else:
        fs.get(cloud_path.rstrip("/") + "/", local_path, recursive=True)
    logger.info(f"Downloaded {cloud_path} to {local_path}")


def download_file(cloud_path: str, local_path: str) -> None:
    """Download one cloud object atomically to a local file."""
    if not is_cloud_path(cloud_path):
        raise ValueError(f"Source must be a cloud path, got: {cloud_path}")

    fs = _get_filesystem(cloud_path)
    local_dir = os.path.dirname(local_path)
    if local_dir:
        os.makedirs(local_dir, exist_ok=True)
    partial_path = f"{local_path}.partial"
    try:
        if cloud_path.startswith("s3://"):
            call_with_s3_retry(fs, fs.get, fs._strip_protocol(cloud_path), partial_path, recursive=False)
        else:
            fs.get(cloud_path, partial_path, recursive=False)
        os.replace(partial_path, local_path)
    finally:
        if os.path.exists(partial_path):
            os.remove(partial_path)
    logger.info(f"Downloaded {cloud_path} to {local_path}")


@contextmanager
def local_read_files(input_paths: Sequence[str]):
    """Stage an explicit set of cloud objects locally without reading sibling objects."""
    paths = list(input_paths)
    cloud_paths = [is_cloud_path(path) for path in paths]
    if any(cloud_paths) and not all(cloud_paths):
        raise ValueError("input_paths must be entirely local or entirely cloud-backed")

    if not any(cloud_paths):
        missing_paths = [path for path in paths if not exists(path)]
        if missing_paths:
            raise FileNotFoundError(f"Paths do not exist: {missing_paths}")
        yield paths
        return

    with tempfile.TemporaryDirectory() as temp_dir:
        local_paths = []
        for index, input_path in enumerate(paths):
            local_path = os.path.join(temp_dir, str(index), os.path.basename(input_path))
            download_file(input_path, local_path)
            local_paths.append(local_path)
        yield local_paths


@contextmanager
def local_work_dir(output_path: str):
    """
    Context manager that provides a local working directory.

    For local paths, returns the path directly.
    For cloud paths, creates a temporary directory and uploads content at the end.

    Args:
        output_path: The final destination path (local or cloud)

    Yields:
        str: Local directory path to work with

    Example:
        with local_work_dir("s3://bucket/model") as work_dir:
            # Save files to work_dir
            model.save_pretrained(work_dir)
            # Files are automatically uploaded to s3://bucket/model at context exit
    """
    if is_cloud_path(output_path):
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                yield temp_dir
            finally:
                # Upload everything from temp_dir to cloud path
                upload_directory(temp_dir, output_path)
                logger.info(f"Uploaded directory contents to {output_path}")
    else:
        # For local paths, ensure directory exists and use it directly
        makedirs(output_path, exist_ok=True)
        yield output_path


@contextmanager
def local_hf_model_dir(output_path: str, *, publish: bool = True):
    """Stage an HF model and let one distributed writer publish weights before the index."""
    index_path = os.path.join(output_path, _HF_WEIGHT_INDEX_FILENAME)
    if publish and exists(index_path):
        remove(index_path)
        logger.info(f"Removed stale HF weight index before serialization: {index_path}")

    if is_cloud_path(output_path):
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
            if publish:
                _upload_hf_model_directory(temp_dir, output_path)
        return

    makedirs(output_path, exist_ok=True)
    try:
        yield output_path
    except BaseException:
        if publish and exists(index_path):
            remove(index_path)
        raise


@contextmanager
def local_read_dir(input_path: str):
    """
    Context manager that provides a local directory with content from input_path.

    For local paths, returns the path directly.
    For cloud paths, downloads content to a temporary directory.

    Args:
        input_path: The source path (local or cloud)

    Yields:
        str: Local directory path containing the content

    Example:
        with local_read_dir("s3://bucket/model") as read_dir:
            # Load files from read_dir
            model = AutoModel.from_pretrained(read_dir)
    """
    if is_cloud_path(input_path):
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download everything from cloud path to temp_dir
            download_directory(input_path, temp_dir)
            logger.info(f"Downloaded directory contents from {input_path}")
            yield temp_dir
    else:
        # For local paths, use directly (but check it exists)
        if not exists(input_path):
            raise FileNotFoundError(f"Path does not exist: {input_path}")
        yield input_path
