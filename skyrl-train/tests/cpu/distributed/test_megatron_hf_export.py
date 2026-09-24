import json
from pathlib import Path
from types import SimpleNamespace

from safetensors.torch import save_file
import torch

from marinskyrl.model_manifest import MODEL_MANIFEST_FILENAME, ModelManifest
from tests.cpu.util import stub_megatron_modules

stub_megatron_modules()

from skyrl_train.distributed.megatron import megatron_strategy  # noqa: E402


class _Bridge:
    def save_hf_weights(self, actor_module, output_dir) -> None:
        save_file({"weight": torch.ones(1)}, Path(output_dir) / "model.safetensors")


def _strategy(rank: int):
    strategy = megatron_strategy.MegatronStrategy.__new__(megatron_strategy.MegatronStrategy)
    strategy.hf_config = object()
    strategy.is_rank_0 = lambda: rank == 0
    strategy.log = lambda *message: None

    def save_hf_configs(config, output_dir, tokenizer) -> None:
        Path(output_dir, "config.json").write_text("{}")
        Path(output_dir, "tokenizer.json").write_text("{}")

    strategy.save_hf_configs = save_hf_configs
    return strategy


def test_megatron_local_hf_export_finalizes_after_nonzero_rank(monkeypatch, tmp_path: Path) -> None:
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    model = SimpleNamespace(actor_module=[])
    bridge = _Bridge()
    monkeypatch.setattr(megatron_strategy, "materialize_megatron_params", lambda actor_module: None)
    monkeypatch.setattr(megatron_strategy.dist, "barrier", lambda: None)

    _strategy(rank=1).save_hf_model(bridge, model, str(export_dir))

    assert (export_dir / "model.safetensors").is_file()
    assert not (export_dir / MODEL_MANIFEST_FILENAME).exists()

    _strategy(rank=0).save_hf_model(bridge, model, str(export_dir), tokenizer=object())

    manifest = ModelManifest.from_mapping(
        json.loads((export_dir / MODEL_MANIFEST_FILENAME).read_text()),
        str(export_dir),
    )
    assert {entry.path for entry in manifest.files} >= {
        "config.json",
        "model.safetensors",
        "model.safetensors.index.json",
        "tokenizer.json",
    }
