from io import BytesIO
from pathlib import Path

from fsspec.implementations.local import LocalFileSystem
import pytest
import torch
from torch.distributed import checkpoint

from skyrl_train.distributed.megatron import checkpoint_metadata


def test_remote_common_state_reads_one_record_without_copying_rank_tensors(tmp_path, monkeypatch) -> None:
    source = tmp_path / "checkpoint"
    common_state = {"optimizer_recipe": "MuonH", "optimizer_recipe_step": 3, "lr_scheduler": {"num_steps": 3}}
    serialized = BytesIO()
    torch.save([common_state], serialized)
    # The common record shares a rank file with a much larger, unrelated tensor.
    tensor = torch.arange(2**20)
    checkpoint.save(
        {"common_state/shard_0_1": serialized, "model.weight": tensor},
        storage_writer=checkpoint.FileSystemWriter(source),
    )
    bytes_read = []

    class CountingReader:
        def __init__(self, file):
            self.file = file

        def __getattr__(self, name):
            return getattr(self.file, name)

        def __enter__(self):
            self.file.__enter__()
            return self

        def __exit__(self, *args):
            return self.file.__exit__(*args)

        def read(self, size=-1):
            value = self.file.read(size)
            bytes_read.append(len(value))
            return value

    class CountingFilesystem(LocalFileSystem):
        def open(self, path, mode="rb", **kwargs):
            file = super().open(path, mode, **kwargs)
            return CountingReader(file) if mode == "rb" and str(path).endswith(".distcp") else file

    monkeypatch.setattr(checkpoint_metadata, "create_s3_filesystem", lambda **_kwargs: CountingFilesystem())
    with checkpoint_metadata.remote_checkpoint_metadata(str(source)) as local_dir:
        root = Path(local_dir)
        assert torch.load(root / "common.pt", weights_only=True) == common_state
        assert not tuple(root.glob("*.distcp"))
    assert 0 < sum(bytes_read) < tensor.nbytes


def test_remote_checkpoint_metadata_never_downloads_rank_tensor_shards(monkeypatch) -> None:
    checkpoint = "s3://bucket/checkpoints/global_step_7/policy"
    objects = {
        "bucket/checkpoints/global_step_7/policy/.metadata": b"dcp metadata",
        "bucket/checkpoints/global_step_7/policy/common.pt": b"common state",
        "bucket/checkpoints/global_step_7/policy/__0_0.distcp": b"rank zero tensors",
        "bucket/checkpoints/global_step_7/policy/__1_0.distcp": b"rank one tensors",
    }
    reads = []
    monkeypatch.setattr(
        checkpoint_metadata.io, "find_files", lambda _path: {key: len(value) for key, value in objects.items()}
    )

    def read_bytes(path: str) -> bytes:
        key = path.removeprefix("s3://")
        reads.append(key)
        return objects[key]

    monkeypatch.setattr(checkpoint_metadata.io, "read_bytes", read_bytes)

    with checkpoint_metadata.remote_checkpoint_metadata(checkpoint) as local_dir:
        root = Path(local_dir)
        assert (root / ".metadata").read_bytes() == b"dcp metadata"
        assert (root / "common.pt").read_bytes() == b"common state"
        assert not tuple(root.glob("*.distcp"))

    assert reads == [
        "bucket/checkpoints/global_step_7/policy/.metadata",
        "bucket/checkpoints/global_step_7/policy/common.pt",
    ]


def test_remote_checkpoint_metadata_rejects_objects_outside_the_checkpoint_prefix(monkeypatch) -> None:
    checkpoint = "s3://bucket/checkpoints/global_step_7/policy"
    monkeypatch.setattr(
        checkpoint_metadata.io,
        "find_files",
        lambda _path: {"bucket/checkpoints/global_step_7/other/common.pt": 12},
    )

    with pytest.raises(ValueError, match="is not below root"):
        with checkpoint_metadata.remote_checkpoint_metadata(checkpoint):
            pass
