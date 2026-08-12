import sys
import types

import pytest
from omegaconf import OmegaConf

from skyrl_train.batch_invariant import BATCH_INVARIANT_ENV, enable_trainer_batch_invariance
from skyrl_train.inference_engines.ray_wrapped_inference_engine import _build_inference_engine_runtime_env
from skyrl_train.utils.utils import prepare_runtime_environment, validate_cfg
from tests.cpu.util import example_dummy_config


def test_batch_invariant_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv(BATCH_INVARIANT_ENV, raising=False)
    cfg = example_dummy_config()

    assert cfg.trainer.algorithm.batch_invariant is False
    assert BATCH_INVARIANT_ENV not in prepare_runtime_environment(cfg)
    assert enable_trainer_batch_invariance(False) == ()


def test_batch_invariant_reaches_ray_and_nested_vllm_workers(monkeypatch):
    cfg = example_dummy_config()
    cfg.trainer.algorithm.batch_invariant = True

    ray_environment = prepare_runtime_environment(cfg)
    monkeypatch.setenv(BATCH_INVARIANT_ENV, ray_environment[BATCH_INVARIANT_ENV])

    assert ray_environment[BATCH_INVARIANT_ENV] == "1"
    assert _build_inference_engine_runtime_env()["env_vars"][BATCH_INVARIANT_ENV] == "1"


def test_trainer_enables_the_pinned_vllm_kernels(monkeypatch):
    activation = {"initialized": False}
    batch_invariant_module = types.ModuleType("vllm.model_executor.layers.batch_invariant")
    batch_invariant_module.init_batch_invariance = lambda: activation.update(initialized=True)
    platforms_module = types.ModuleType("vllm.platforms")
    platforms_module.current_platform = types.SimpleNamespace(
        is_device_capability_family=lambda capability: capability == 80
    )
    monkeypatch.setitem(sys.modules, batch_invariant_module.__name__, batch_invariant_module)
    monkeypatch.setitem(sys.modules, platforms_module.__name__, platforms_module)
    monkeypatch.setenv(BATCH_INVARIANT_ENV, "1")

    enabled_ops = enable_trainer_batch_invariance(True)

    assert activation["initialized"] is True
    assert set(enabled_ops) >= {"aten::mm", "aten::addmm", "aten::bmm"}


@pytest.mark.parametrize(
    ("config_path", "value", "message"),
    [
        ("generator.backend", "sglang", "requires generator.backend='vllm'"),
        ("generator.run_engines_locally", False, "cannot configure a remote inference server"),
    ],
)
def test_batch_invariant_rejects_uncontrolled_generator(config_path, value, message):
    cfg = example_dummy_config()
    cfg.trainer.algorithm.batch_invariant = True
    OmegaConf.update(cfg, config_path, value)

    with pytest.raises(ValueError, match=message):
        validate_cfg(cfg)
