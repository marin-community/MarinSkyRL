import importlib
from pathlib import Path

import yaml
from hydra import compose, initialize_config_dir

from skyrl_train.entrypoints.main_base import config_dir


CONFIG_DIR = Path(__file__).parents[1] / "configs"


def test_every_iris_rl_entrypoint_imports_from_the_installed_package():
    entrypoints = {
        config["entrypoint"]
        for path in CONFIG_DIR.glob("*.yaml")
        if (config := yaml.safe_load(path.read_text())) and "entrypoint" in config
    }

    assert entrypoints == {
        "skyrl_train.entrypoints.main_base",
        "skyrl_train.entrypoints.terminal_bench",
    }
    for entrypoint in entrypoints:
        module = importlib.import_module(entrypoint)
        assert callable(module.main)


def test_terminal_bench_config_group_is_packaged_with_the_trainer():
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="ppo_base_config", overrides=["+terminal_bench_config=terminal_bench"])

    assert cfg.terminal_bench_config.prm.window_size == 6
