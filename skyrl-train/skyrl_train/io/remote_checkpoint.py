"""Bounded local views of remote distributed checkpoints."""

from contextlib import contextmanager
from pathlib import Path
import posixpath
import tempfile
from urllib.parse import urlsplit

from marinskyrl.resource_locator import join_resource_path
from skyrl_train.io import io


def _object_key(value: str) -> str:
    parsed = urlsplit(value)
    if not parsed.scheme:
        return value
    return f"{parsed.netloc}{parsed.path}"


def _relative_object_key(root: str, path: str) -> str:
    normalized_root = posixpath.normpath(_object_key(root))
    normalized_path = posixpath.normpath(_object_key(path))
    relative = posixpath.relpath(normalized_path, normalized_root)
    if relative == ".." or relative.startswith("../") or posixpath.isabs(relative):
        raise ValueError(f"Checkpoint object {path!r} is not below {root!r}")
    return relative


@contextmanager
def remote_checkpoint_metadata(checkpoint_dir: str):
    """Stage checkpoint control files while leaving DCP rank tensors remote."""
    with tempfile.TemporaryDirectory(prefix="megatron-metadata-") as directory:
        root = Path(directory)
        for filename in io.find_files(checkpoint_dir):
            relative = _relative_object_key(checkpoint_dir, filename)
            if relative.endswith(".distcp"):
                continue
            destination = root / relative
            if not destination.resolve().is_relative_to(root):
                raise ValueError(f"Invalid checkpoint metadata path: {relative}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(io.read_bytes(join_resource_path(checkpoint_dir, relative)))
        yield directory
