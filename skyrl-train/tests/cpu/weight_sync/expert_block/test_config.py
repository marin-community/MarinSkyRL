"""Config validation of the ``expert_block`` weight-sync transport."""

import subprocess
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from marinskyrl.inference_placement import validate_expert_block_transport
from marinskyrl.runtime_options import GDNBackend, R3Transport, WeightSyncTransport
from skyrl_train import objective
from skyrl_train.config.query_bias import GrugQueryBiasUpdateMode
from skyrl_train.config.weight_sync_pause import WeightSyncPauseMode
from skyrl_train.utils import advantage_estimators, utils as trainer_utils
from skyrl_train.utils.utils import validate_cfg
from tests.cpu.util import example_dummy_config


def expert_block_config():
    cfg = example_dummy_config()
    cfg.trainer.strategy = "megatron"
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.policy.megatron_config.tensor_model_parallel_size = 1
    cfg.trainer.policy.megatron_config.expert_model_parallel_size = 8
    cfg.generator.update(
        backend="vllm",
        run_engines_locally=True,
        weight_sync_backend="nccl",
        weight_sync_transport="expert_block",
        inference_engine_tensor_parallel_size=1,
        inference_engine_pipeline_parallel_size=1,
        inference_engine_data_parallel_size=8,
        inference_engine_expert_parallel_size=8,
    )
    return cfg


def test_pipeline_parallel_engines_on_the_mp_backend_are_refused():
    cfg = expert_block_config()
    cfg.generator.inference_engine_pipeline_parallel_size = 2
    validate_expert_block_transport(cfg)
    cfg.generator.inference_engine_mp_backend = True
    with pytest.raises(ValueError, match="Ray executor"):
        validate_expert_block_transport(cfg)


def test_validate_cfg_runs_the_transport_check(generated_recipe_schema):
    root = Path(__file__).resolve().parents[5]
    assert Path(trainer_utils.__file__).resolve() == root / "skyrl-train/skyrl_train/utils/utils.py"
    assert Path(objective.__file__).resolve() == root / "skyrl-train/skyrl_train/objective/__init__.py"
    assert (
        Path(advantage_estimators.__file__).resolve() == root / "skyrl-train/skyrl_train/utils/advantage_estimators.py"
    )
    print(f"runtime selector validator source: {trainer_utils.__file__}")
    recipe_type, _ = generated_recipe_schema
    recipe = recipe_type.from_document(
        {
            "trainer": {
                "train_batch_size": 4,
                "policy_mini_batch_size": 4,
                "micro_train_batch_size_per_gpu": 1,
                "flash_attn": False,
                "progress": {"mode": "logging"},
                "policy": {
                    "grug_query_bias_update_mode": GrugQueryBiasUpdateMode.INTERPOLATE.value,
                    "grug_query_bias_interpolation_weight": 0.25,
                },
            },
            "generator": {
                "weight_sync_transport": WeightSyncTransport.EXPERT_BLOCK.value,
                "r3_transport": R3Transport.RESIDENT.value,
                "gdn_backend": GDNBackend.FLASHQLA.value,
                "weight_sync_pause": {"mode": WeightSyncPauseMode.WAIT.value},
                "vllm_v1_disable_multiproc": False,
            },
        }
    )
    cfg = OmegaConf.merge(example_dummy_config(), recipe.to_skyrl())
    with pytest.raises(ValueError, match="weight_sync_transport=expert_block requires"):
        validate_cfg(cfg)
    changed = recipe.with_settings([f"generator.weight_sync_transport={WeightSyncTransport.BROADCAST.value}"])
    validate_cfg(OmegaConf.merge(example_dummy_config(), changed.to_skyrl()))
    incompatible = changed.with_settings(["generator.vllm_v1_disable_multiproc=true"])
    with pytest.raises(ValueError, match="mode=wait requires"):
        validate_cfg(OmegaConf.merge(example_dummy_config(), incompatible.to_skyrl()))
    compatible = incompatible.with_settings([f"generator.weight_sync_pause.mode={WeightSyncPauseMode.KEEP.value}"])
    validate_cfg(OmegaConf.merge(example_dummy_config(), compatible.to_skyrl()))
    for path in (
        "generator.weight_sync_transport",
        "generator.r3_transport",
        "generator.gdn_backend",
        "generator.weight_sync_pause.mode",
        "trainer.progress.mode",
        "trainer.policy.grug_query_bias_update_mode",
    ):
        with pytest.raises(ValueError):
            changed.with_settings([f"{path}=unknown-runtime-choice"])


def test_the_model_package_imports_before_the_trainer_utilities():
    # The frozen-runtime bootstrap imports the Grug model first. That import reaches
    # skyrl_train.utils and this validator, and must not be circular.
    subprocess.run(
        [sys.executable, "-c", "from skyrl_train.models.grug_moe import GRUG_MOE_ARCHITECTURE"],
        check=True,
        capture_output=True,
        text=True,
    )
