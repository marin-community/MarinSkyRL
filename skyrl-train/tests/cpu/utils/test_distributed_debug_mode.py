import json
from pathlib import Path

from omegaconf import OmegaConf

from tests.cpu.util import example_dummy_config

from skyrl_train.distributed_debug import apply_distributed_debug_mode, distributed_debug_environment
from skyrl_train.env_vars import (
    DEBUG_ARTIFACT_DIR_ENV,
    DEBUG_MODE_ENV,
    DistributedDebugMode,
    FR_DUMP_TEMP_FILE_ENV,
    NCCL_DEBUG_INFO_TEMP_FILE_ENV,
    write_process_manifest,
)
from skyrl_train.utils.utils import prepare_runtime_environment


def _debug_config(*, checkpoint_path: str = "/gpfs/experiments/run/checkpoints"):
    cfg = example_dummy_config()
    OmegaConf.update(cfg, "trainer.debug_mode", DistributedDebugMode.DISTRIBUTED.value)
    OmegaConf.update(cfg, "trainer.ckpt_path", checkpoint_path)
    return cfg


def test_normal_mode_does_not_enable_expensive_diagnostics(monkeypatch):
    monkeypatch.delenv(DEBUG_MODE_ENV, raising=False)
    cfg = example_dummy_config()

    environment = distributed_debug_environment(cfg)

    assert DEBUG_MODE_ENV not in environment
    assert DEBUG_ARTIFACT_DIR_ENV not in environment
    assert "NCCL_DEBUG" not in environment


def test_distributed_mode_expands_complete_worker_contract(monkeypatch):
    monkeypatch.delenv(DEBUG_MODE_ENV, raising=False)
    monkeypatch.delenv(DEBUG_ARTIFACT_DIR_ENV, raising=False)

    environment = distributed_debug_environment(_debug_config())

    assert environment[DEBUG_MODE_ENV] == "distributed"
    assert environment[DEBUG_ARTIFACT_DIR_ENV] == "/gpfs/experiments/run/debug"
    assert environment["NCCL_DEBUG"] == "INFO"
    assert environment["NCCL_DEBUG_SUBSYS"] == "INIT,BOOTSTRAP,ENV,NET,GRAPH,TUNING"
    assert environment["SKYRL_COLLECTIVE_PHASE_DIAGNOSTICS"] == "1"
    assert environment["TORCH_CPP_LOG_LEVEL"] == "INFO"
    assert environment["TORCH_NCCL_DESYNC_DEBUG"] == "1"
    assert environment["TORCH_NCCL_ENABLE_TIMING"] == "1"
    assert environment["TORCH_NCCL_TRACE_CPP_STACK"] == "1"
    assert environment["TORCH_SHOW_CPP_STACKTRACES"] == "1"
    assert environment["TORCH_SYMBOLIZE_MODE"] == "fast"
    assert environment["PYTHONFAULTHANDLER"] == "1"
    assert environment["TORCH_FR_DUMP_TEMP_FILE"].startswith("/gpfs/experiments/run/debug/flight_recorder/")
    assert environment["NCCL_DEBUG_FILE"] == "/gpfs/experiments/run/debug/nccl/nccl.%h.%p.log"
    assert "CUDA_LAUNCH_BLOCKING" not in environment
    assert "TORCH_DISTRIBUTED_DEBUG" not in environment


def test_distributed_mode_stages_remote_runs_locally(monkeypatch):
    monkeypatch.delenv(DEBUG_MODE_ENV, raising=False)
    monkeypatch.delenv(DEBUG_ARTIFACT_DIR_ENV, raising=False)

    environment = distributed_debug_environment(_debug_config(checkpoint_path="s3://bucket/run/checkpoints"))

    assert environment[DEBUG_ARTIFACT_DIR_ENV] == "/tmp/skyrl-debug/test-run"


def test_config_mode_uses_launcher_owned_artifact_path(monkeypatch):
    monkeypatch.setenv(DEBUG_ARTIFACT_DIR_ENV, "/tmp/launcher-owned-debug")
    cfg = example_dummy_config()
    OmegaConf.update(cfg, "trainer.debug_mode", "distributed")

    environment = distributed_debug_environment(cfg)

    assert environment[DEBUG_ARTIFACT_DIR_ENV] == "/tmp/launcher-owned-debug"
    assert environment["TORCH_NCCL_ENABLE_TIMING"] == "1"


def test_apply_mode_writes_resolved_driver_manifest(tmp_path, monkeypatch):
    artifact_root = tmp_path / "artifacts"
    process_environment = {DEBUG_ARTIFACT_DIR_ENV: str(artifact_root)}

    environment = apply_distributed_debug_mode(_debug_config(), environ=process_environment)
    manifest_path = write_process_manifest("driver", environment=environment)
    manifest = json.loads(manifest_path.read_text())

    assert manifest_path.parent == artifact_root / "processes"
    assert manifest["role"] == "driver"
    assert manifest["debug_mode"] == "distributed"
    assert manifest["environment"]["TORCH_NCCL_ENABLE_TIMING"] == "1"
    assert manifest["environment"]["NCCL_DEBUG_SUBSYS"] == "INIT,BOOTSTRAP,ENV,NET,GRAPH,TUNING"
    assert Path(environment["TORCH_FR_DUMP_TEMP_FILE"]).parent.is_dir()
    assert Path(environment["NCCL_DEBUG_FILE"]).parent.is_dir()


def test_distributed_mode_overrides_legacy_dump_destination(monkeypatch):
    monkeypatch.setattr("skyrl_train.utils.utils.peer_access_supported", lambda **_: False)
    monkeypatch.delenv(DEBUG_MODE_ENV, raising=False)
    monkeypatch.delenv(DEBUG_ARTIFACT_DIR_ENV, raising=False)
    monkeypatch.setenv(FR_DUMP_TEMP_FILE_ENV, "/tmp/nccl_fr_rank")
    monkeypatch.setenv(NCCL_DEBUG_INFO_TEMP_FILE_ENV, "/tmp/nccl_fr_rank")

    environment = prepare_runtime_environment(_debug_config())

    expected_prefix = "/gpfs/experiments/run/debug/flight_recorder/nccl_fr_rank_"
    assert environment[FR_DUMP_TEMP_FILE_ENV] == expected_prefix
    assert environment[NCCL_DEBUG_INFO_TEMP_FILE_ENV] == expected_prefix
    assert environment["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] == "300"
