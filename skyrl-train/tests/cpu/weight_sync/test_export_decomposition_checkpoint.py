"""The probe's checkpoint writer and header reader agree on a real sharded safetensors file."""

import json

import pytest
import torch
from safetensors.torch import save_file
from skyrl_train.entrypoints.probe_export_decomposition import checkpoint_inventory, write_snowball_width_checkpoint

TOY_SHAPE = {
    "hidden_size": 64,
    "intermediate_size": 64,
    "shared_expert_intermediate_size": 64,
    "num_local_experts": 8,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 64,
    "sliding_window": 16,
    "vocab_size": 100,
}


def test_inventory_round_trips_the_written_checkpoint(tmp_path):
    written = write_snowball_width_checkpoint(tmp_path, 2, TOY_SHAPE)
    read = checkpoint_inventory(tmp_path)
    assert sorted(map(tuple, written)) == sorted(map(tuple, read))
    assert len(read) > 0 and len({name for name, _, _ in read}) == len(read)
    assert {dtype for name, _, dtype in read if name.endswith(".mlp.router.bias")} == {"float32"}
    assert {dtype for name, _, dtype in read if not name.endswith(".mlp.router.bias")} == {"bfloat16"}
    experts = [shape for name, shape, _ in read if ".mlp.experts." in name]
    assert experts and all(len(shape) == 3 and shape[0] == 8 for shape in experts)
    index = json.loads((tmp_path / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"]) == {name for name, _, _ in read}


def test_inventory_reads_headers_of_a_handwritten_shard(tmp_path):
    save_file(
        {"a.weight": torch.zeros(2, 3, dtype=torch.bfloat16), "b.bias": torch.ones(4)}, str(tmp_path / "x.safetensors")
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": {"a.weight": "x.safetensors", "b.bias": "x.safetensors"}})
    )
    assert checkpoint_inventory(tmp_path) == [("a.weight", [2, 3], "bfloat16"), ("b.bias", [4], "float32")]


def test_inventory_rejects_a_missing_shard(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "gone.safetensors"}}))
    with pytest.raises(FileNotFoundError):
        checkpoint_inventory(tmp_path)
