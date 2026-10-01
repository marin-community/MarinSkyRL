"""Contract checks for the asynchronous Snowball MOPD smoke against its synchronous baseline."""

from pathlib import Path
from types import SimpleNamespace

import yaml

from cloud.iris.rl_config_translation import compose_skyrl_config, parse_rl_config
from cloud.iris.role_plan import derive_role_plan
import skyrl_train.objective.losses  # noqa: F401  (registers policy losses for validate_cfg)
from skyrl_train.utils import validate_cfg

CONFIGS = Path(__file__).parents[3] / "cloud" / "iris" / "configs"
ASYNC_CONFIG = CONFIGS / "snowball_mopd_ultra_async_smoke.yaml"
SYNC_CONFIG = CONFIGS / "snowball_mopd_ultra_smoke.yaml"
STUDENT = "open-athena/Snowball-67B-A2B-10T-Mixed-RLVR-Sync-Step92"


def test_async_smoke_runs_the_in_process_async_trainer_and_passes_trainer_validation():
    parsed = parse_rl_config(str(ASYNC_CONFIG), model_override=STUDENT)
    compiled = compose_skyrl_config(
        parsed,
        {"job_name": "mopd-async-smoke-test", "experiments_dir": "/tmp/exp", "num_nodes": 8},
        SimpleNamespace(gpus_per_node=8),
    )

    assert compiled.entrypoint == "skyrl_train.entrypoints.fully_async_in_process"
    validate_cfg(compiled.config)


def test_async_smoke_matches_the_sync_baseline_geometry_and_prompt_count():
    async_config = yaml.safe_load(ASYNC_CONFIG.read_text())
    sync_config = yaml.safe_load(SYNC_CONFIG.read_text())

    assert derive_role_plan(async_config) == derive_role_plan(sync_config)
    assert async_config["teachers"] == sync_config["teachers"]
    assert async_config["context_budget"] == sync_config["context_budget"]
    # One optimizer update per step in both schedules, so equal steps mean equal prompts.
    for key in ("train_batch_size", "policy_mini_batch_size", "max_steps"):
        assert async_config["trainer"][key] == sync_config["trainer"][key]
