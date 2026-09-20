"""The trained Hero replay fixture must preserve stacked expert values."""

import json

from safetensors.torch import load_file, save_file
import torch

from tests.gpu.hero_replay_checkpoint import split_stacked_hero_checkpoint


def test_split_stacked_hero_checkpoint_preserves_values(tmp_path) -> None:
    source = tmp_path / "stacked"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"grugmoe_artifact_schema_version": 2, "num_experts": 3}))
    state = {
        f"model.layers.0.mlp.experts.{projection}.weight": torch.arange(12, dtype=torch.float32)
        .reshape(3, 2, 2)
        .to(torch.bfloat16)
        + offset
        for offset, projection in enumerate(("gate_proj", "up_proj", "down_proj"))
    }
    state["model.layers.0.mlp.router.weight"] = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    save_file(state, str(source / "model-00001-of-00001.safetensors"), metadata={"format": "pt"})
    (source / "model.safetensors.index.json").write_text(
        json.dumps(
            {"metadata": {"total_size": 96}, "weight_map": {name: "model-00001-of-00001.safetensors" for name in state}}
        )
    )

    destination = tmp_path / "split"
    assert split_stacked_hero_checkpoint(source, destination) == 9
    converted = load_file(str(destination / "model-00001-of-00001.safetensors"))
    index = json.loads((destination / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"]) == set(converted)
    assert len(converted) == 10
    torch.testing.assert_close(converted["model.layers.0.mlp.router.weight"], state["model.layers.0.mlp.router.weight"])
    for projection in ("gate_proj", "up_proj", "down_proj"):
        stacked = state[f"model.layers.0.mlp.experts.{projection}.weight"]
        for expert in range(3):
            torch.testing.assert_close(
                converted[f"model.layers.0.mlp.experts.{expert}.{projection}.weight"], stacked[expert], rtol=0, atol=0
            )
    assert set(load_file(str(source / "model-00001-of-00001.safetensors"))) == set(state)
