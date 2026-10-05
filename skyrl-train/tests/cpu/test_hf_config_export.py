"""HF export retains generation settings embedded in original model configs."""

import json

from transformers import AutoConfig

from skyrl_train.distributed.strategy import DistributedStrategy


def test_hf_export_keeps_legacy_generation_fields_and_current_model_values(tmp_path):
    original = tmp_path / "original"
    original.mkdir()
    (original / "config.json").write_text(
        json.dumps({"model_type": "gpt2", "n_layer": 2, "begin_suppress_tokens": [41, 42], "eos_token_id": 43})
    )
    config = AutoConfig.from_pretrained(original)
    assert "begin_suppress_tokens" not in config.to_dict()
    config.n_layer = 4
    config.eos_token_id = 44
    exported = tmp_path / "exported"

    DistributedStrategy.save_hf_configs(None, config, str(exported))

    saved = json.loads((exported / "config.json").read_text())
    assert saved["begin_suppress_tokens"] == [41, 42]
    assert saved["n_layer"] == 4
    assert saved["eos_token_id"] == 44
