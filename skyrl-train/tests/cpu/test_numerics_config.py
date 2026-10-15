import pytest
from omegaconf import OmegaConf
from transformers import LlamaConfig

import skyrl_train.objective.losses  # noqa: F401
from skyrl_train.config.numerics import NumericsMode, NumericsResolution, read_numerics
from skyrl_train.config.utils import get_default_config
from skyrl_train.models.grug_moe import GrugMoeConfig
from skyrl_train.utils.utils import validate_cfg


def _config(**settings):
    cfg = get_default_config()
    cfg.trainer.logger = "console"
    for path, value in settings.items():
        # engine_init_kwargs holds arbitrary vLLM engine kwargs, so new keys may be added under it.
        OmegaConf.update(cfg, path, value, force_add=path == "generator.engine_init_kwargs")
    return cfg


def _policy(tmp_path, model_config) -> str:
    path = tmp_path / "policy"
    model_config.save_pretrained(path)
    return str(path)


def test_the_default_run_resolves_to_native_numerics_and_leaves_the_config_unchanged():
    cfg = _config()
    validate_cfg(cfg)
    before = OmegaConf.to_container(cfg, resolve=False)

    resolution = read_numerics(cfg)

    assert resolution.mode is NumericsMode.NATIVE
    assert OmegaConf.to_container(cfg, resolve=False) == before


def test_batch_invariant_numerics_resolve_for_local_vllm_engines():
    cfg = _config(**{"trainer.numerics.mode": "batch_invariant"})
    validate_cfg(cfg)

    assert read_numerics(cfg).mode is NumericsMode.BATCH_INVARIANT


# The Snowball settings an exact run keeps: unpacked sequences, FLASH_ATTN named in the config, Triton MoE kernels,
# TP 1 engines on their own GPUs.
EXACT_SETTINGS = {
    "trainer.numerics.mode": "exact",
    "trainer.use_sample_packing": False,
    "trainer.placement.colocate_all": False,
    "trainer.policy.megatron_config.ddp_config.overlap_param_gather": False,
    "generator.vllm_attention_backend": "FLASH_ATTN",
    "generator.inference_engine_tensor_parallel_size": 1,
    "generator.engine_init_kwargs": {"moe_backend": "triton"},
}


def test_exact_numerics_resolve_with_snowball_engine_settings(tmp_path):
    policy = _policy(tmp_path, GrugMoeConfig())
    cfg = _config(
        **EXACT_SETTINGS, **{"trainer.numerics.check.on_failure": "stop", "trainer.policy.model.path": policy}
    )
    validate_cfg(cfg)

    assert read_numerics(cfg) == NumericsResolution(NumericsMode.EXACT)


@pytest.mark.parametrize(
    ("model_config", "reason"),
    [
        (LlamaConfig(num_hidden_layers=1), "not Grug"),
        (GrugMoeConfig(num_key_value_heads=4), "num_key_value_heads"),
        (GrugMoeConfig(sconv=True, grugmoe_artifact_schema_version=2), "Hero"),
    ],
    ids=["another model family", "another Grug shape", "Hero"],
)
def test_exact_numerics_resolve_to_native_for_policies_the_kernels_do_not_cover(tmp_path, model_config, reason):
    cfg = _config(**EXACT_SETTINGS, **{"trainer.policy.model.path": _policy(tmp_path, model_config)})
    validate_cfg(cfg)

    resolution = read_numerics(cfg)

    assert resolution.mode is NumericsMode.NATIVE
    assert any(reason in recorded for recorded in resolution.fallback_reasons), resolution


@pytest.mark.parametrize(
    ("settings", "error"),
    [
        ({"trainer.numerics.mode": "bitwise"}, "trainer.numerics.mode must be one of"),
        (
            {"trainer.numerics.mode": "batch_invariant", "generator.run_engines_locally": False},
            "remote inference server",
        ),
        ({"trainer.numerics.check.every_weight_syncs": -1}, "every_weight_syncs must be at least 0"),
        ({"trainer.numerics.check.on_failure": "halt"}, "on_failure must be one of"),
        ({"trainer.numerics.check.on_failure": "stop"}, "on_failure=stop requires a numerics mode"),
    ],
    ids=[
        "unknown mode",
        "batch_invariant with remote engines",
        "negative check interval",
        "unknown failure action",
        "stop without a bit-identical mode",
    ],
)
def test_validation_rejects_numerics_settings_the_run_cannot_honor(settings, error):
    cfg = _config(**settings)

    with pytest.raises(ValueError, match=error):
        validate_cfg(cfg)
