from tests.cpu.util import example_dummy_config
from omegaconf import OmegaConf

from skyrl_train.inference_engines.ray_wrapped_inference_engine import (
    _NCCL_FR_ENV_PASSTHROUGH,
    _build_inference_engine_runtime_env,
)
from skyrl_train.utils.utils import prepare_runtime_environment
from skyrl_train.config.utils import get_default_config
from skyrl_train.utils.constants import validate_worker_collective_timeout_seconds


def test_worker_collective_timeout_fallback_matches_trainer_default():
    config = get_default_config()

    assert config.trainer.distributed.worker_collective_timeout_seconds == 300
    assert validate_worker_collective_timeout_seconds() == 300


def test_runtime_environment_does_not_enable_nonblocking_communicators(monkeypatch):
    monkeypatch.setattr("skyrl_train.utils.utils.peer_access_supported", lambda **_: False)
    monkeypatch.delenv("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", raising=False)
    env = prepare_runtime_environment(example_dummy_config())

    assert env["TORCH_NCCL_ENABLE_MONITORING"] == "1"
    assert env["TORCH_NCCL_ASYNC_ERROR_HANDLING"] == "1"
    assert env["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] == "300"
    assert "TORCH_NCCL_USE_COMM_NONBLOCKING" not in env
    assert "TORCH_NCCL_NONBLOCKING_TIMEOUT" not in env


def test_monitor_heartbeat_is_capped_by_collective_timeout(monkeypatch):
    monkeypatch.setattr("skyrl_train.utils.utils.peer_access_supported", lambda **_: False)
    monkeypatch.setenv("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "300")
    cfg = example_dummy_config()
    OmegaConf.update(cfg, "trainer.distributed.worker_collective_timeout_seconds", 30)
    env = prepare_runtime_environment(cfg)

    assert env["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] == "30"


def test_debug_preset_preserves_collective_timeout_cap(monkeypatch):
    monkeypatch.setattr("skyrl_train.utils.utils.peer_access_supported", lambda **_: False)
    cfg = example_dummy_config()
    OmegaConf.update(cfg, "trainer.debug_mode", "distributed")
    OmegaConf.update(cfg, "trainer.distributed.worker_collective_timeout_seconds", 30)

    env = prepare_runtime_environment(cfg)

    assert env["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] == "30"
    assert env["TORCH_NCCL_ENABLE_TIMING"] == "1"


def test_inference_engine_forwards_only_supported_nccl_diagnostics(monkeypatch):
    for variable in _NCCL_FR_ENV_PASSTHROUGH:
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("TORCH_NCCL_USE_COMM_NONBLOCKING", "1")
    monkeypatch.setenv("TORCH_NCCL_NONBLOCKING_TIMEOUT", "47")
    monkeypatch.setenv("NCCL_BLOCKING_WAIT", "1")
    monkeypatch.setenv("TORCH_NCCL_BLOCKING_WAIT_TIMEOUT_MS", "1800000")
    monkeypatch.setenv("TORCH_NCCL_ENABLE_MONITORING", "1")

    runtime_env = _build_inference_engine_runtime_env()

    assert runtime_env == {"env_vars": {"TORCH_NCCL_ENABLE_MONITORING": "1"}}
