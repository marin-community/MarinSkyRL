import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from safetensors.torch import save_file
import torch

from cloud.iris.hf_model_cache import stage_model_metadata
from marinskyrl.model_manifest import snapshot_model_manifest
from skyrl_train.io.remote_safetensors import RemoteSafetensorsTensorStore, lazy_first_dim_patterns_for_bridge


def _write_index(metadata_dir: Path, weight_map: dict[str, str]) -> None:
    metadata_dir.mkdir()
    (metadata_dir / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))


def test_loads_only_requested_tensor_range_without_local_weight_files(tmp_path: Path) -> None:
    remote = tmp_path / "remote"
    remote.mkdir()
    save_file(
        {
            "layer.0.weight": torch.arange(8, dtype=torch.float32),
            "layer.1.weight": torch.arange(1_000_000, dtype=torch.float32),
        },
        remote / "model-00001-of-00001.safetensors",
    )
    metadata = tmp_path / "metadata"
    _write_index(
        metadata,
        {
            "layer.0.weight": "model-00001-of-00001.safetensors",
            "layer.1.weight": "model-00001-of-00001.safetensors",
        },
    )

    store = RemoteSafetensorsTensorStore(str(remote), metadata)
    loaded = store.load_tensors(["layer.0.weight"])

    torch.testing.assert_close(loaded["layer.0.weight"], torch.arange(8, dtype=torch.float32))
    assert store.bytes_read < (remote / "model-00001-of-00001.safetensors").stat().st_size
    assert not tuple(metadata.glob("*.safetensors"))


def test_single_file_export_without_index_streams_only_requested_tensor(tmp_path: Path) -> None:
    remote = tmp_path / "levanter-export"
    remote.mkdir()
    save_file(
        {
            "model.embed_tokens.weight": torch.arange(8, dtype=torch.float32),
            "model.layers.0.weight": torch.arange(1_000_000, dtype=torch.float32),
        },
        remote / "model.safetensors",
    )
    metadata = tmp_path / "metadata"
    metadata.mkdir()

    store = RemoteSafetensorsTensorStore(str(remote), metadata)
    loaded = store.load_tensors(["model.embed_tokens.weight"])

    assert store.get_all_keys() == ["model.embed_tokens.weight", "model.layers.0.weight"]
    torch.testing.assert_close(loaded["model.embed_tokens.weight"], torch.arange(8, dtype=torch.float32))
    assert store.bytes_read < (remote / "model.safetensors").stat().st_size
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


def test_auto_bridge_uses_registered_bridge_remote_slice_patterns(tmp_path: Path) -> None:
    key = "model.layers.0.mlp.experts.down_proj.weight"
    metadata = tmp_path / "metadata"
    _write_index(metadata, {key: "model.safetensors"})

    registered_bridge = SimpleNamespace(REMOTE_FIRST_DIM_SLICE_PATTERNS=("model.layers.*.mlp.experts.*",))
    auto_bridge = SimpleNamespace(_model_bridge=registered_bridge)

    patterns = lazy_first_dim_patterns_for_bridge(auto_bridge)
    store = RemoteSafetensorsTensorStore("s3://models/snowball", metadata, lazy_first_dim_patterns=patterns)

    assert patterns == registered_bridge.REMOTE_FIRST_DIM_SLICE_PATTERNS
    assert store._lazy_first_dim_keys == {key}


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
    s3_bytes_read = sum(store.bytes_read for store in stores)
    local_disk_high_water_bytes = sum(
        path.stat().st_size for metadata in metadata_dirs for path in metadata.rglob("*") if path.is_file()
    )

    assert s3_bytes_read < artifact_bytes * 1.1
    assert local_disk_high_water_bytes < artifact_bytes * 0.01
    assert not tuple(tmp_path.glob("rank-*/*.safetensors"))
