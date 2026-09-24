from tests.cpu.util import example_dummy_config
from omegaconf import OmegaConf

from skyrl_train.inference_engines.ray_wrapped_inference_engine import (
    _NCCL_FR_ENV_PASSTHROUGH,
    _build_inference_engine_runtime_env,
)
from skyrl_train.env_vars import (
    DEFAULT_RUNAI_STREAMER_LOG_TO_STDERR,
    DEFAULT_RUNAI_STREAMER_S3_REQUEST_TIMEOUT_MS,
    RUNAI_STREAMER_LOG_TO_STDERR_ENV,
    RUNAI_STREAMER_S3_REQUEST_TIMEOUT_MS_ENV,
)
from skyrl_train.utils.utils import prepare_runtime_environment


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


def test_selected_id_teacher_forces_v1_runner_without_changing_default(monkeypatch):
    for variable in _NCCL_FR_ENV_PASSTHROUGH:
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)

    default_env = _build_inference_engine_runtime_env()
    selected_env = _build_inference_engine_runtime_env(require_v1_model_runner=True)

    assert default_env is None or "VLLM_USE_V2_MODEL_RUNNER" not in default_env["env_vars"]
    assert selected_env is not None
    assert selected_env["env_vars"]["VLLM_USE_V2_MODEL_RUNNER"] == "0"


def test_runai_streamer_gets_s3_stall_tolerance_and_diagnostics(monkeypatch):
    monkeypatch.delenv(RUNAI_STREAMER_S3_REQUEST_TIMEOUT_MS_ENV, raising=False)
    monkeypatch.delenv(RUNAI_STREAMER_LOG_TO_STDERR_ENV, raising=False)

    runtime_env = _build_inference_engine_runtime_env(runai_streamer_enabled=True)

    assert runtime_env is not None
    assert runtime_env["env_vars"][RUNAI_STREAMER_S3_REQUEST_TIMEOUT_MS_ENV] == (
        DEFAULT_RUNAI_STREAMER_S3_REQUEST_TIMEOUT_MS
    )
    assert runtime_env["env_vars"][RUNAI_STREAMER_LOG_TO_STDERR_ENV] == DEFAULT_RUNAI_STREAMER_LOG_TO_STDERR


def test_runai_streamer_preserves_explicit_s3_tuning(monkeypatch):
    monkeypatch.setenv(RUNAI_STREAMER_S3_REQUEST_TIMEOUT_MS_ENV, "30000")
    monkeypatch.setenv(RUNAI_STREAMER_LOG_TO_STDERR_ENV, "0")

    runtime_env = _build_inference_engine_runtime_env(runai_streamer_enabled=True)

    assert runtime_env is not None
    assert runtime_env["env_vars"][RUNAI_STREAMER_S3_REQUEST_TIMEOUT_MS_ENV] == "30000"
    assert runtime_env["env_vars"][RUNAI_STREAMER_LOG_TO_STDERR_ENV] == "0"
