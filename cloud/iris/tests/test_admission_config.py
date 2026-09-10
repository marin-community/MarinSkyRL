from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from hydra import compose, initialize_config_dir

from cloud.iris.rl_config_translation import build_skyrl_hydra_args, parse_rl_config


@pytest.mark.parametrize("delay", [0, 3])
def test_actual_async_parser_composes_bounded_admission(delay, tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "entrypoint": "fully_async",
                "trainer": {
                    "fully_async": {
                        "max_buffered_groups": 256,
                        "admission_order": "fifo",
                        "injected_delay_max_steps": delay,
                        "first_token_admission": True,
                    }
                },
                "context_budget": {"request_window_tokens": 2048, "max_new_tokens_per_turn": 1024, "max_turns": 1},
            }
        )
    )
    parsed = parse_rl_config(str(path), model_override="/tmp/model")
    args = build_skyrl_hydra_args(
        parsed, {"model_path": "/tmp/model", "num_nodes": 1}, SimpleNamespace(gpus_per_node=8, cpus_per_node=16)
    )
    location = Path(__file__).resolve().parents[3] / "skyrl-train/skyrl_train/config"
    with initialize_config_dir(config_dir=str(location), version_base=None):
        default = compose(config_name="ppo_base_config")
        config = compose(config_name="ppo_base_config", overrides=args)
    assert default.trainer.fully_async.max_buffered_groups is None
    assert config.trainer.fully_async.max_buffered_groups == 256
    assert config.trainer.fully_async.injected_delay_max_steps == delay
    assert config.trainer.fully_async.first_token_admission is True
