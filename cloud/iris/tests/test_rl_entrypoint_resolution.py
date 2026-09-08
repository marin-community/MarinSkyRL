from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from cloud.iris.iris_backend import create_parser, normalize
from cloud.iris.rl_config_translation import build_skyrl_hydra_args, parse_rl_config
from skyrl_train.entrypoints.main_base import config_dir


_REPO_ROOT = Path(__file__).resolve().parents[3]


def test_external_rl_config_rejects_deleted_module_path_before_dry_run(tmp_path):
    config = tmp_path / "rl.yaml"
    config.write_text("entrypoint: examples.terminal_bench.entrypoints.main_tbench\n")
    args = create_parser().parse_args(["--rl_config", str(config), "--model_path", "Qwen/Qwen3-8B", "--dry-run"])

    with pytest.raises(SystemExit, match="examples.terminal_bench.entrypoints.main_tbench"):
        normalize(args)


def test_rl_config_resolves_named_terminal_bench_entrypoint(tmp_path):
    config = tmp_path / "rl.yaml"
    config.write_text(
        """\
entrypoint: terminal_bench
context_budget:
  request_window_tokens: 2
  max_new_tokens_per_turn: 1
  max_turns: 1
"""
    )

    parsed = parse_rl_config(str(config))

    assert parsed.entrypoint == "skyrl_train.entrypoints.terminal_bench"


def test_native_readback_output_uri_survives_launcher_translation(tmp_path):
    uri = "s3://marin-us-east-02a/diagnostics/native-readback"
    config = tmp_path / "readback.yaml"
    config.write_text(
        "entrypoint: weight_sync_readback\n"
        "context_budget:\n  request_window_tokens: 2\n  max_new_tokens_per_turn: 1\n  max_turns: 1\n"
        f"trainer:\n  weight_sync_readback_output: {uri}\n  weight_sync_nccl_diagnostics: true\n  logger: console\n"
    )
    parsed = parse_rl_config(str(config))
    generated = build_skyrl_hydra_args(parsed, {"num_nodes": 1}, SimpleNamespace(gpus_per_node=8))
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="ppo_base_config", overrides=generated)
    assert parsed.entrypoint == "skyrl_train.entrypoints.weight_sync_readback"
    assert cfg.trainer.weight_sync_readback_output == uri
    assert cfg.trainer.logger == "console"


@pytest.mark.parametrize("ignore_eos", [False, True])
def test_training_eos_control_composes_with_structured_native_config(tmp_path, ignore_eos):
    config = tmp_path / "rl.yaml"
    config.write_text(
        "entrypoint: fully_async\n"
        "context_budget:\n  request_window_tokens: 2048\n  max_new_tokens_per_turn: 1024\n  max_turns: 1\n"
        f"generator:\n  sampling_params:\n    ignore_eos: {str(ignore_eos).lower()}\n"
    )
    parsed = parse_rl_config(str(config))
    hydra_args = build_skyrl_hydra_args(parsed, {"num_nodes": 2}, SimpleNamespace(gpus_per_node=8))
    assert f"generator.sampling_params.ignore_eos={str(ignore_eos).lower()}" in hydra_args
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="ppo_base_config", overrides=hydra_args)
    assert OmegaConf.is_struct(cfg.generator.sampling_params)
    assert cfg.generator.sampling_params.ignore_eos is ignore_eos
    assert not cfg.generator.eval_sampling_params.get("ignore_eos", False)
    assert cfg.generator.eval_sampling_params.temperature == 0.0


def test_terminal_bench_config_group_is_packaged_with_the_trainer():
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="ppo_base_config", overrides=["+terminal_bench_config=terminal_bench"])

    assert cfg.get("terminal_bench_config") is not None


def test_terminal_bench_launcher_overrides_compose_with_packaged_group():
    parsed = parse_rl_config(str(_REPO_ROOT / "cloud/iris/configs/tasktrove_dq_sweep_30b.yaml"))
    expected_prm = {
        "name": "loop_penalty",
        "window_size": 7,
        "similarity_threshold": 0.6,
        "min_turns": 11,
        "check_interval": 4,
    }
    expected_trace_upload = {
        "enabled": True,
        "repo_org": "marin-community",
        "episodes": "all",
        "dataset_type": "RL",
    }
    parsed = replace(
        parsed,
        terminal_bench={**parsed.terminal_bench, "prm": expected_prm, "trace_upload": expected_trace_upload},
    )
    hydra_args = build_skyrl_hydra_args(parsed, {"num_nodes": 4}, SimpleNamespace(gpus_per_node=8))

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="ppo_base_config", overrides=hydra_args)

    assert OmegaConf.to_container(cfg.terminal_bench_config.prm) == expected_prm
    assert OmegaConf.to_container(cfg.terminal_bench_config.trace_upload) == expected_trace_upload


def test_trajectory_runner_settings_reach_skyrl_hydra_config(tmp_path):
    config = tmp_path / "rl.yaml"
    config.write_text(
        """\
entrypoint: terminal_bench
context_budget:
  request_window_tokens: 2
  max_new_tokens_per_turn: 1
  max_turns: 1
trajectory_runner:
  process_pool:
    num_coordinators: 4
    cpus_per_coordinator: 8
    rpc_timeout_seconds: 1200
"""
    )

    parsed = parse_rl_config(str(config))
    hydra_args = build_skyrl_hydra_args(parsed, {"num_nodes": 4}, SimpleNamespace(gpus_per_node=8))

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="ppo_base_config", overrides=hydra_args)

    process_pool = OmegaConf.to_container(cfg.trajectory_runner.process_pool)
    assert process_pool["num_coordinators"] == 4
    assert process_pool["cpus_per_coordinator"] == 8
    assert process_pool["rpc_timeout_seconds"] == 1200
    assert process_pool["executor_workers"] == 256
