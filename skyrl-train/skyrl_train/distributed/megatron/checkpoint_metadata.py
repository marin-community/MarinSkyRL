"""Bounded local metadata views of remote Megatron checkpoints."""

from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
import tempfile

import torch
from torch.distributed import checkpoint
from torch.distributed.checkpoint._fsspec_filesystem import FileSystem as FsspecFileSystem

from marinskyrl.remote_io import create_s3_filesystem
from marinskyrl.resource_locator import join_resource_path, relative_resource_path
from skyrl_train.io import io
from skyrl_train.io.checkpoint_reader import RecordCheckpointReader


def _stage_common_state(checkpoint_dir: str, destination: Path) -> None:
    # Core 0.19 stores common data as ShardedObject("common_state", ..., (1,), (0,)).
    # Its reader still accepts a common.pt view. Read just that DCP record; copying
    # its containing rank shard would also download the model and optimizer tensors.
    key = "common_state/shard_0_1"
    state = {key: BytesIO()}
    reader = RecordCheckpointReader(checkpoint_dir)
    reader.fs = FsspecFileSystem()
    reader.fs.fs = create_s3_filesystem(default_cache_type="none")
    reader.path = checkpoint_dir
    checkpoint.load(state, storage_reader=reader, no_dist=True)
    value = state[key]
    if isinstance(value, BytesIO):
        value.seek(0)
        value = torch.load(value, weights_only=True)
    torch.save(value[0], destination)


@contextmanager
def remote_checkpoint_metadata(checkpoint_dir: str):
    """Stage Megatron control files while leaving DCP rank tensors remote."""
    with tempfile.TemporaryDirectory(prefix="megatron-metadata-") as directory:
        root = Path(directory).resolve()
        for filename in io.find_files(checkpoint_dir):
            relative = relative_resource_path(checkpoint_dir, filename)
            if relative.endswith(".distcp"):
                continue
            destination = root / relative
            if not destination.resolve().is_relative_to(root):
                raise ValueError(f"Invalid checkpoint metadata path: {relative}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(io.read_bytes(join_resource_path(checkpoint_dir, relative)))
        if not (root / "common.pt").exists():
            _stage_common_state(checkpoint_dir, root / "common.pt")
        yield directory
