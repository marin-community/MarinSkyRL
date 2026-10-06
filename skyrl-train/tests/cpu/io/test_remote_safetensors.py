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
import skyrl_train.io.remote_safetensors as remote_safetensors
from skyrl_train.io.remote_safetensors import (
    RemoteSafetensorsTensorStore,
    lazy_first_dim_patterns_for_bridge,
    prefetch_items_from_tasks,
)


class CountingBufferedFile(AbstractBufferedFile):
    def read(self, length=-1):
        payload = super().read(length)
        self.fs.reads += 1
        self.fs.bytes_read += len(payload)
        return payload


class CountingFileSystem(AbstractFileSystem):
    protocol = "s3"
    cachable = False

    def __init__(self, payload: bytes, shards: dict[str, bytes] | None = None, **kwargs):
        super().__init__(**kwargs)
        self.payload = payload
        self.shards = shards or {}
        self.opens = 0
        self.reads = 0
        self.bytes_read = 0
        self.fetched_bytes = 0
        self.gets = 0

    def info(self, path, **kwargs):
        return {"name": path, "type": "file", "size": len(self.shards.get(path, self.payload))}

    def _open(self, path, mode="rb", block_size=None, cache_type="readahead", **kwargs):
        self.opens += 1
        return CountingBufferedFile(
            self,
            path,
            mode,
            block_size=50 * 2**20,
            cache_type=cache_type,
            size=len(self.shards.get(path, self.payload)),
        )

    def cat_file(self, path, start=None, end=None, **kwargs):
        self.gets += 1
        payload = self.shards.get(path, self.payload)[start:end]
        self.fetched_bytes += len(payload)
        return payload


def _write_index(metadata_dir: Path, weight_map: dict[str, str]) -> None:
    metadata_dir.mkdir()
    (metadata_dir / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))


@pytest.mark.parametrize("header_metadata_bytes", [0, 9 * 1024**2], ids=["small-header", "multi-read-header"])
def test_conversion_task_prefetch_preserves_rank_owned_tensors_and_reduces_range_fetches(
    tmp_path, monkeypatch, header_metadata_bytes
):
    tensors = {f"dense.{index}": torch.arange(8, dtype=torch.float32) + index for index in range(24)}
    tensors["dense.1"] = tensors["dense.1"].to(torch.bfloat16)
    tensors["experts"] = torch.arange(24, dtype=torch.bfloat16).reshape(3, 8)
    tensors["outside"] = torch.arange(8, dtype=torch.float32)
    shards = {}
    for name in ("a.safetensors", "b.safetensors"):
        shard = tmp_path / name
        save_file(
            {key: value for key, value in tensors.items() if key.startswith("dense.") == (name == "a.safetensors")},
            shard,
            metadata={"annotation": "x" * header_metadata_bytes},
        )
        shards[f"bucket/policy/{name}"] = shard.read_bytes()
    metadata = tmp_path / "metadata"
    _write_index(metadata, {key: "a.safetensors" if key.startswith("dense.") else "b.safetensors" for key in tensors})
    tasks = [
        SimpleNamespace(
            megatron_module=object(),
            mapping=SimpleNamespace(
                hf_param={"left": f"dense.{index}", "right": f"dense.{index + 1}"}, megatron_param="dense"
            ),
        )
        for index in range(0, 24, 2)
    ]
    tasks += [
        SimpleNamespace(
            megatron_module=object(), mapping=SimpleNamespace(hf_param="experts", megatron_param=str(index))
        )
        for index in (2, 0)
    ]
    tasks += [
        SimpleNamespace(megatron_module=None, mapping=SimpleNamespace(hf_param="outside", megatron_param="other_pp")),
        SimpleNamespace(
            megatron_module=object(), mapping=SimpleNamespace(hf_param="synthesized", megatron_param="synthesized")
        ),
        tasks[0],
    ]
    results = []
    for prefetched in (False, True):
        filesystem = CountingFileSystem(b"", shards=shards)
        monkeypatch.setattr(io, "_get_filesystem", lambda path: filesystem)
        store = RemoteSafetensorsTensorStore("s3://bucket/policy", metadata, lazy_first_dim_patterns=("experts",))
        if prefetched:
            store.plan_prefetch(prefetch_items_from_tasks(tasks, {"experts"}, int))
        for index in range(24):
            value = store.load_tensors([f"dense.{index}"])[f"dense.{index}"]
            assert torch.equal(value, tensors[f"dense.{index}"])
            assert value.dtype == tensors[f"dense.{index}"].dtype
        experts = store.load_tensors(["experts"])["experts"]
        for index in (-1, 0):
            assert torch.equal(experts[index], tensors["experts"][index])
        with pytest.raises(TypeError):
            experts[:]
        store.close()
        assert store.read_stats.prefetch_misses == store.read_stats.prefetch_unused == 0
        assert store.read_stats.gets == filesystem.gets
        results.append((store.read_stats.bytes_read, filesystem.gets))
    assert results[0][0] == results[1][0]
    assert results[1][1] < results[0][1]


def test_prefetch_large_tensors_split_ranges_and_keep_windows_bounded(tmp_path, monkeypatch):
    tensors = {str(index): torch.full((65 * 1024**2,), index, dtype=torch.uint8) for index in range(3)}
    shard = tmp_path / "model.safetensors"
    save_file(tensors, shard)
    metadata = tmp_path / "metadata"
    _write_index(metadata, {key: shard.name for key in tensors})
    filesystem = CountingFileSystem(shard.read_bytes())
    monkeypatch.setattr(io, "_get_filesystem", lambda path: filesystem)
    monkeypatch.setattr(remote_safetensors, "_PREFETCH_WINDOW_BYTES", 140 * 1024**2)
    cat_ranges = filesystem.cat_ranges

    def bounded_ranges(paths, starts, ends, **kwargs):
        sizes = [end - start for start, end in zip(starts, ends, strict=True)]
        assert sum(sizes) <= 140 * 1024**2
        assert max(sizes) <= 64 * 1024**2
        return cat_ranges(paths, starts, ends, **kwargs)

    monkeypatch.setattr(filesystem, "cat_ranges", bounded_ranges)
    store = RemoteSafetensorsTensorStore("s3://bucket/policy", metadata)
    store.plan_prefetch((key, None) for key in tensors)
    try:
        for key, tensor in tensors.items():
            assert torch.equal(store.load_tensors([key])[key], tensor)
    finally:
        store.close()
    assert store.read_stats.prefetch_misses == store.read_stats.prefetch_unused == 0


@pytest.mark.parametrize("failure", [False, True], ids=["fallback", "range-error"])
def test_prefetch_unplanned_key_reads_through_and_range_errors_propagate(tmp_path, monkeypatch, failure):
    shard = tmp_path / "model.safetensors"
    tensors = {"planned": torch.arange(8, dtype=torch.float32), "outside": torch.arange(8, dtype=torch.float32) + 10}
    save_file(tensors, shard)
    metadata = tmp_path / "metadata"
    _write_index(metadata, {key: shard.name for key in tensors})
    filesystem = CountingFileSystem(shard.read_bytes())
    monkeypatch.setattr(io, "_get_filesystem", lambda path: filesystem)
    store = RemoteSafetensorsTensorStore("s3://bucket/policy", metadata)
    store.plan_prefetch([("planned", None)])
    try:
        if failure:
            monkeypatch.setattr(
                filesystem, "cat_ranges", lambda *args, **kwargs: [RuntimeError("object-store range failed")]
            )
            with pytest.raises(RuntimeError, match="object-store range failed"):
                store.load_tensors(["planned"])
        else:
            assert torch.equal(store.load_tensors(["outside"])["outside"], tensors["outside"])
            assert torch.equal(store.load_tensors(["planned"])["planned"], tensors["planned"])
            assert store.read_stats.prefetch_misses == 1
    finally:
        store.close()


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
