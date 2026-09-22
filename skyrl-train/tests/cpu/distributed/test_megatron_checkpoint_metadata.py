from pathlib import Path

import pytest

from skyrl_train.distributed.megatron import checkpoint_metadata


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
