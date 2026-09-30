"""Contract checks for the asynchronous Snowball MOPD smoke against its synchronous baseline."""

from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf


from cloud.iris.rl_config_translation import compose_skyrl_config, load_rl_recipe, parse_rl_config
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

    assert compiled.entrypoint == "skyrl_train.entrypoints.main_base"
    validate_cfg(compiled.config)


def test_async_smoke_matches_the_sync_baseline_geometry_and_prompt_count():
    async_config = OmegaConf.to_container(load_rl_recipe(str(ASYNC_CONFIG)), resolve=True)
    sync_config = OmegaConf.to_container(load_rl_recipe(str(SYNC_CONFIG)), resolve=True)

    assert derive_role_plan(async_config) == derive_role_plan(sync_config)
    assert async_config["teachers"] == sync_config["teachers"]
    assert async_config["context_budget"] == sync_config["context_budget"]
    # One optimizer update per step in both schedules, so equal steps mean equal prompts.
    for key in ("train_batch_size", "policy_mini_batch_size", "max_steps"):
        assert async_config["trainer"][key] == sync_config["trainer"][key]


SWE_SYNC_CONFIG = CONFIGS / "snowball_mopd_ultra_32k_swe_smoke.yaml"
SWE_ASYNC_CONFIG = CONFIGS / "snowball_mopd_ultra_async_32k_swe_smoke.yaml"


def test_swe_smokes_route_every_row_to_harbor_and_pass_trainer_validation():
    for config_path in (SWE_SYNC_CONFIG, SWE_ASYNC_CONFIG):
        parsed = parse_rl_config(str(config_path), model_override=STUDENT)
        compiled = compose_skyrl_config(
            parsed,
            {"job_name": "mopd-swe-smoke-test", "experiments_dir": "/tmp/exp", "num_nodes": 7},
            SimpleNamespace(gpus_per_node=8),
        )

        # Validation imports flash_attn when the recipe enables it, and CPU CI has no GPU build.
        compiled.config.trainer.flash_attn = False
        validate_cfg(compiled.config)
        assert compiled.config.get("terminal_bench_config")
        assert set(compiled.config.teacher_routing.opd.routes) == {"swe"}


def test_async_swe_smoke_matches_the_sync_swe_smoke_geometry_and_prompt_count():
    async_config = yaml.safe_load(SWE_ASYNC_CONFIG.read_text())
    sync_config = yaml.safe_load(SWE_SYNC_CONFIG.read_text())

    assert derive_role_plan(async_config) == derive_role_plan(sync_config)
    assert async_config["teachers"] == sync_config["teachers"]
    assert async_config["terminal_bench"] == sync_config["terminal_bench"]
    for key in ("train_batch_size", "policy_mini_batch_size", "max_steps"):
        assert async_config["trainer"][key] == sync_config["trainer"][key]
