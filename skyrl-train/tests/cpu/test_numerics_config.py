import pytest
from omegaconf import OmegaConf

import skyrl_train.objective.losses  # noqa: F401
from skyrl_train.config.numerics import NumericsMode, resolve_numerics
from skyrl_train.config.utils import get_default_config
from skyrl_train.utils.utils import validate_cfg


def _config(**settings):
    cfg = get_default_config()
    cfg.trainer.logger = "console"
    for path, value in settings.items():
        OmegaConf.update(cfg, path, value)
    return cfg


def test_the_default_run_resolves_to_native_numerics_and_leaves_the_config_unchanged():
    cfg = _config()
    validate_cfg(cfg)
    before = OmegaConf.to_container(cfg, resolve=False)

    resolution = resolve_numerics(cfg)

    assert resolution.mode is NumericsMode.NATIVE
    assert OmegaConf.to_container(cfg, resolve=False) == before


def test_batch_invariant_numerics_resolve_for_local_vllm_engines():
    cfg = _config(**{"trainer.numerics.mode": "batch_invariant"})
    validate_cfg(cfg)

    assert resolve_numerics(cfg).mode is NumericsMode.BATCH_INVARIANT


@pytest.mark.parametrize(
    ("settings", "error"),
    [
        ({"trainer.numerics.mode": "exact"}, "trainer.numerics.mode must be one of"),
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
