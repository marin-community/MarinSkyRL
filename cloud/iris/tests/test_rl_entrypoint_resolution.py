from dataclasses import dataclass, replace
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from cloud.iris.iris_backend import create_parser, normalize
from cloud.iris.rl_config_translation import build_skyrl_hydra_args, parse_rl_config
from skyrl_train.entrypoints.main_base import config_dir


_REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class _HPCStub:
    gpus_per_node: int = 8


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
    hydra_args = build_skyrl_hydra_args(parsed, {"num_nodes": 4}, _HPCStub())

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="ppo_base_config", overrides=hydra_args)

    assert OmegaConf.to_container(cfg.terminal_bench_config.prm) == expected_prm
    assert OmegaConf.to_container(cfg.terminal_bench_config.trace_upload) == expected_trace_upload
