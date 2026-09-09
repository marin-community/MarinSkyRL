from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from hydra import compose, initialize_config_dir

from cloud.iris.rl_config_translation import (
    build_checkpoint_export_hydra_args,
    build_skyrl_hydra_args,
    parse_checkpoint_export_config,
    parse_rl_config,
)


@pytest.mark.parametrize("export", [False, True])
def test_guard_object_uri_survives_actual_parser_and_hydra(tmp_path, export):
    uri = "s3://marin-us-east-02a/marin/users/ahmad/diagnostics/measurement.json"
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "trainer": {"measurement_guard_uri": uri},
                "entrypoint": "standard",
                "context_budget": {"request_window_tokens": 2048, "max_new_tokens_per_turn": 1024, "max_turns": 1},
            }
        )
    )
    parse, build = (
        (parse_checkpoint_export_config, build_checkpoint_export_hydra_args)
        if export
        else (parse_rl_config, build_skyrl_hydra_args)
    )
    parsed = parse(str(path), model_override="/tmp/model")
    args = build(
        parsed, {"model_path": "/tmp/model", "num_nodes": 1}, SimpleNamespace(gpus_per_node=8, cpus_per_node=16)
    )
    location = Path(__file__).resolve().parents[3] / "skyrl-train/skyrl_train/config"
    with initialize_config_dir(config_dir=str(location), version_base=None):
        config = compose(config_name="ppo_base_config", overrides=args)
    assert config.trainer.measurement_guard_uri == uri
