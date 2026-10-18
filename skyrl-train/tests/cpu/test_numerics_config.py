import pytest
from omegaconf import OmegaConf
from transformers import LlamaConfig

from skyrl_train.config.numerics import NumericsMode, read_numerics
from skyrl_train.config.utils import get_default_config
from skyrl_train.models.grug_moe import GrugMoeConfig


def _config(**settings):
    cfg = get_default_config()
    for path, value in settings.items():
        OmegaConf.update(cfg, path, value)
    return cfg


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
    model_config.save_pretrained(tmp_path)
    cfg = _config(**{"trainer.numerics.mode": "exact", "trainer.policy.model.path": str(tmp_path)})

    resolution = read_numerics(cfg)

    assert resolution.mode is NumericsMode.NATIVE
    assert any(reason in recorded for recorded in resolution.fallback_reasons), resolution
