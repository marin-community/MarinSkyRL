import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fsspec.spec import AbstractBufferedFile, AbstractFileSystem
from safetensors.torch import save_file
import torch

from cloud.iris.hf_model_cache import stage_model_metadata
from marinskyrl.model_manifest import snapshot_model_manifest
from skyrl_train.io import io
from skyrl_train.io.remote_safetensors import RemoteSafetensorsTensorStore, lazy_first_dim_patterns_for_bridge


class CountingBufferedFile(AbstractBufferedFile):
    def read(self, length=-1):
        payload = super().read(length)
        self.fs.reads += 1
        self.fs.bytes_read += len(payload)
        return payload


class CountingFileSystem(AbstractFileSystem):
    protocol = "s3"
    cachable = False

    def __init__(self, payload: bytes, **kwargs):
        super().__init__(**kwargs)
        self.payload = payload
        self.opens = 0
        self.reads = 0
        self.bytes_read = 0
        self.fetched_bytes = 0

    def info(self, path, **kwargs):
        return {"name": path, "type": "file", "size": len(self.payload)}

    def _open(self, path, mode="rb", block_size=None, cache_type="readahead", **kwargs):
        self.opens += 1
        return CountingBufferedFile(
            self, path, mode, block_size=50 * 2**20, cache_type=cache_type, size=len(self.payload)
        )

    def cat_file(self, path, start=None, end=None, **kwargs):
        payload = self.payload[start:end]
        self.fetched_bytes += len(payload)
        return payload


def _write_index(metadata_dir: Path, weight_map: dict[str, str]) -> None:
    metadata_dir.mkdir()
    (metadata_dir / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))


@pytest.mark.parametrize("indexed", [True, False])
def test_single_key_and_expert_slice_reads_fetch_only_requested_bytes(
    tmp_path: Path, monkeypatch, indexed: bool
) -> None:
    shard = tmp_path / "model.safetensors"
    tensors = {f"layer.{index}.weight": torch.arange(8 + index, dtype=torch.float32) for index in range(3)}
    tensors["experts.weight"] = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    tensors["unrequested.weight"] = torch.ones(1024)
    save_file(tensors, shard)
    filesystem = CountingFileSystem(shard.read_bytes())
    local_filesystem_for = io._get_filesystem
    monkeypatch.setattr(
        io, "_get_filesystem", lambda path: filesystem if path.startswith("s3://") else local_filesystem_for(path)
    )
    metadata = tmp_path / "metadata"
    if indexed:
        _write_index(metadata, {key: shard.name for key in tensors})
    else:
        metadata.mkdir()
    store = RemoteSafetensorsTensorStore("s3://bucket/policy", metadata, lazy_first_dim_patterns=("experts.weight",))
    for index in (2, 0, 1):
        key = f"layer.{index}.weight"
        torch.testing.assert_close(store.load_tensors([key])[key], tensors[key])
    experts = store.load_tensors(["experts.weight"])["experts.weight"]
    for index in (2, 0, 1):
        torch.testing.assert_close(experts[index], tensors["experts.weight"][index])
    unrequested = tensors["unrequested.weight"]
    expected_bytes = len(filesystem.payload) - unrequested.numel() * unrequested.element_size()
    assert store.read_stats.opens == filesystem.opens == 1
    assert filesystem.fetched_bytes == filesystem.bytes_read == store.read_stats.bytes_read == expected_bytes


@pytest.mark.parametrize("indexed", [True, False])
def test_loads_only_requested_tensor_range_without_local_weight_files(tmp_path: Path, indexed: bool) -> None:
    remote = tmp_path / "remote"
    remote.mkdir()
    weight_file = "model-00001-of-00001.safetensors" if indexed else "model.safetensors"
    save_file(
        {
            "layer.0.weight": torch.arange(8, dtype=torch.float32),
            "layer.1.weight": torch.arange(1_000_000, dtype=torch.float32),
        },
        remote / weight_file,
    )
    metadata = tmp_path / "metadata"
    if indexed:
        _write_index(metadata, {"layer.0.weight": weight_file, "layer.1.weight": weight_file})
    else:
        metadata.mkdir()

    store = RemoteSafetensorsTensorStore(str(remote), metadata)
    loaded = store.load_tensors(["layer.0.weight"])

    torch.testing.assert_close(loaded["layer.0.weight"], torch.arange(8, dtype=torch.float32))
    assert store.read_stats.bytes_read < (remote / weight_file).stat().st_size
    assert not tuple(metadata.glob("*.safetensors"))


def test_invalid_index_is_not_replaced_by_single_file_fallback(tmp_path: Path) -> None:
    remote = tmp_path / "remote"
    remote.mkdir()
    save_file({"weight": torch.ones(2)}, remote / "model.safetensors")
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    (metadata / "model.safetensors.index.json").write_text("not JSON")

    with pytest.raises(ValueError, match="Invalid safetensors weight index"):
        RemoteSafetensorsTensorStore(str(remote), metadata)


def test_auto_bridge_uses_registered_bridge_remote_slice_patterns() -> None:
    registered_bridge = SimpleNamespace(REMOTE_FIRST_DIM_SLICE_PATTERNS=("model.layers.*.mlp.experts.*",))
    auto_bridge = SimpleNamespace(_model_bridge=registered_bridge)

    assert lazy_first_dim_patterns_for_bridge(auto_bridge) == registered_bridge.REMOTE_FIRST_DIM_SLICE_PATTERNS


def test_twelve_rank_simulation_records_bounded_remote_reads_and_local_disk(tmp_path: Path) -> None:
    remote = tmp_path / "remote"
    remote.mkdir()
    shard_name = "model-00001-of-00001.safetensors"
    rank_count = 12
    stacked_key = "model.layers.0.mlp.experts.down_proj.weight"
    save_file(
        {stacked_key: torch.arange(rank_count * 100_000, dtype=torch.float32).reshape(rank_count, 100_000)},
        remote / shard_name,
    )
    (remote / "config.json").write_text("{}")
    (remote / "tokenizer.json").write_text("{}")
    manifest = snapshot_model_manifest(remote, model_id="snowball/policy", revision="pinned")

    metadata_dirs = [tmp_path / f"rank-{rank}" for rank in range(rank_count)]
    for metadata in metadata_dirs:
        stage_model_metadata(str(remote), manifest, str(metadata))
    stores = [
        RemoteSafetensorsTensorStore(
            str(remote),
            metadata,
            lazy_first_dim_patterns=("model.layers.*.mlp.experts.*_proj.weight",),
        )
        for metadata in metadata_dirs
    ]
    for rank, store in enumerate(stores):
        stacked = store.load_tensors([stacked_key])[stacked_key]
        torch.testing.assert_close(
            stacked[rank],
            torch.arange(rank * 100_000, (rank + 1) * 100_000, dtype=torch.float32),
        )

    artifact_bytes = (remote / shard_name).stat().st_size
    s3_bytes_read = sum(store.read_stats.bytes_read for store in stores)
    local_disk_high_water_bytes = sum(
        path.stat().st_size for metadata in metadata_dirs for path in metadata.rglob("*") if path.is_file()
    )

    assert s3_bytes_read < artifact_bytes * 1.1
    assert local_disk_high_water_bytes < artifact_bytes * 0.01
    assert not tuple(tmp_path.glob("rank-*/*.safetensors"))
