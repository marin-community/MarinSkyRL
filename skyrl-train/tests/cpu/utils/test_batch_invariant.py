import sys
import types

import pytest
from omegaconf import OmegaConf

from skyrl_train.batch_invariant import enable_trainer_batch_invariance
from skyrl_train.env_vars import VLLM_BATCH_INVARIANT_ENV
from skyrl_train.inference_engines.ray_wrapped_inference_engine import _build_inference_engine_runtime_env
from skyrl_train.utils.utils import prepare_runtime_environment, validate_batch_invariant_config
from tests.cpu.util import example_dummy_config


def test_batch_invariant_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv(VLLM_BATCH_INVARIANT_ENV, raising=False)
    monkeypatch.setattr("skyrl_train.utils.utils.peer_access_supported", lambda **_: True)
    cfg = example_dummy_config()

    assert cfg.trainer.algorithm.batch_invariant is False
    assert VLLM_BATCH_INVARIANT_ENV not in prepare_runtime_environment(cfg)


def test_batch_invariant_reaches_ray_and_nested_vllm_workers(monkeypatch):
    monkeypatch.setattr("skyrl_train.utils.utils.peer_access_supported", lambda **_: True)
    cfg = example_dummy_config()
    cfg.trainer.algorithm.batch_invariant = True

    ray_environment = prepare_runtime_environment(cfg)
    monkeypatch.setenv(VLLM_BATCH_INVARIANT_ENV, ray_environment[VLLM_BATCH_INVARIANT_ENV])

    assert ray_environment[VLLM_BATCH_INVARIANT_ENV] == "1"
    assert _build_inference_engine_runtime_env()["env_vars"][VLLM_BATCH_INVARIANT_ENV] == "1"


def test_trainer_requires_registered_kernels_after_vllm_activation(monkeypatch):
    batch_invariant_module = types.ModuleType("vllm.model_executor.layers.batch_invariant")
    batch_invariant_module.init_batch_invariance = lambda: None
    monkeypatch.setitem(sys.modules, batch_invariant_module.__name__, batch_invariant_module)
    monkeypatch.setenv(VLLM_BATCH_INVARIANT_ENV, "1")

    with pytest.raises(RuntimeError, match="without exposing registered CUDA overrides"):
        enable_trainer_batch_invariance(True)


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
        validate_batch_invariant_config(cfg)
