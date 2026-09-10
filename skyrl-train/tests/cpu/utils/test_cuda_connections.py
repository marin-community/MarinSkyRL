"""Opt-in CUDA connections are qualified before the Ray/vLLM worker boundary."""

from omegaconf import OmegaConf
import pytest

from cloud.iris.env_vars import EnvVarManager, EnvVarScope
from skyrl_train.inference_engines.ray_wrapped_inference_engine import _build_inference_engine_runtime_env
from skyrl_train.utils.utils import prepare_runtime_environment
from tests.cpu.util import example_dummy_config


def experimental_config():
    cfg = example_dummy_config()
    cfg.trainer.strategy = "megatron"
    cfg.trainer.cuda_device_max_connections = 8
    cfg.generator.inference_engine_tensor_parallel_size = 1
    cfg.generator.weight_sync_timing_mode = "bucket"
    return cfg


def test_opt_in_reaches_policy_and_nested_engine_environment(monkeypatch):
    monkeypatch.setattr("skyrl_train.utils.utils.peer_access_supported", lambda **_: True)
    monkeypatch.setenv("VLLM_RAY_EXTRA_ENV_VARS_TO_COPY", "CUSTOM_INPUT")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    cfg = experimental_config()
    policy = prepare_runtime_environment(cfg)
    assert policy["CUDA_DEVICE_MAX_CONNECTIONS"] == "8"
    assert policy["NCCL_CUMEM_ENABLE"] == policy["VLLM_BATCH_INVARIANT"] == "0"
    assert set(policy["VLLM_RAY_EXTRA_ENV_VARS_TO_COPY"].split(",")) == {"CUSTOM_INPUT", "CUDA_DEVICE_MAX_CONNECTIONS"}
    for key, value in policy.items():
        monkeypatch.setenv(key, value)
    nested = _build_inference_engine_runtime_env()["env_vars"]
    assert all(
        nested[key] == policy[key]
        for key in (
            "CUDA_DEVICE_MAX_CONNECTIONS",
            "NCCL_CUMEM_ENABLE",
            "VLLM_BATCH_INVARIANT",
            "VLLM_RAY_EXTRA_ENV_VARS_TO_COPY",
        )
    )


@pytest.mark.parametrize("tp", [1, 2, 4])
def test_default_megatron_connections_remain_one(monkeypatch, tp):
    monkeypatch.setattr("skyrl_train.utils.utils.peer_access_supported", lambda **_: True)
    cfg = example_dummy_config()
    cfg.trainer.strategy = "megatron"
    cfg.trainer.policy.megatron_config.tensor_model_parallel_size = tp
    assert cfg.trainer.cuda_device_max_connections is None
    assert prepare_runtime_environment(cfg)["CUDA_DEVICE_MAX_CONNECTIONS"] == "1"


@pytest.mark.parametrize(
    "path,value",
    [
        ("trainer.cuda_device_max_connections", True),
        ("trainer.cuda_device_max_connections", 4),
        ("trainer.strategy", "fsdp2"),
        ("trainer.policy.megatron_config.tensor_model_parallel_size", 2),
        ("trainer.ref.megatron_config.tensor_model_parallel_size", 2),
        ("trainer.policy.megatron_config.expert_tensor_parallel_size", 2),
        ("trainer.policy.megatron_config.transformer_config_kwargs.sequence_parallel", True),
        ("trainer.policy.megatron_config.transformer_config_kwargs.tensor_model_parallel_size", 2),
        ("generator.inference_engine_tensor_parallel_size", 2),
        ("generator.backend", "sglang"),
        ("generator.run_engines_locally", False),
        ("generator.weight_sync_timing_mode", "reference"),
        ("trainer.algorithm.batch_invariant", True),
    ],
)
def test_unsupported_opt_in_rejects_before_worker_creation(path, value):
    cfg = experimental_config()
    OmegaConf.update(cfg, path, value, force_add=True)
    with pytest.raises(ValueError):
        EnvVarManager.from_config(cfg).environment_for(EnvVarScope.RAY_WORKER)
