"""Bounded local views of remote distributed checkpoints."""

from contextlib import contextmanager
from pathlib import Path
import tempfile

from skyrl_train.io import io


@contextmanager
def remote_checkpoint_metadata(checkpoint_dir: str):
    """Stage checkpoint control files while leaving DCP rank tensors remote."""
    prefix = checkpoint_dir.split("://", 1)[-1].rstrip("/") + "/"
    with tempfile.TemporaryDirectory(prefix="megatron-metadata-") as directory:
        root = Path(directory)
        for filename in io.find_files(checkpoint_dir):
            relative = filename.split("://", 1)[-1].removeprefix(prefix)
            if relative.endswith(".distcp"):
                continue
            destination = root / relative
            if not destination.resolve().is_relative_to(root):
                raise ValueError(f"Invalid checkpoint metadata path: {relative}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(io.read_bytes(f"{checkpoint_dir.rstrip('/')}/{relative}"))
        yield directory
