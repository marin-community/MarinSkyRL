import pytest
from hydra import compose, initialize_config_dir

from cloud.iris.iris_backend import create_parser, normalize
from cloud.iris.rl_config_translation import parse_rl_config
from skyrl_train.entrypoints.main_base import config_dir


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
