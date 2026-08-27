import json

import ray
from omegaconf import OmegaConf

from cloud.iris.env_vars import EXECUTION_UID_ENV, RUN_ID_ENV, TELEMETRY_ENDPOINT_ENV
from tests.cpu.util import example_dummy_config

from skyrl_train.distributed_debug import distributed_debug_environment
from skyrl_train.env_vars import (
    DEBUG_ARTIFACT_DIR_ENV,
    DEBUG_MODE_ENV,
    FR_DUMP_TEMP_FILE_ENV,
    NCCL_DEBUG_INFO_TEMP_FILE_ENV,
)
from skyrl_train.utils.utils import initialize_ray


def test_ray_initialization_persists_distributed_debug_contract(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "run" / "checkpoints"
    cfg = example_dummy_config()
    OmegaConf.update(cfg, "trainer.debug_mode", "distributed")
    OmegaConf.update(cfg, "trainer.ckpt_path", str(checkpoint_path))
    for name, value in distributed_debug_environment(cfg).items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv(DEBUG_MODE_ENV)
    monkeypatch.delenv(DEBUG_ARTIFACT_DIR_ENV)
    monkeypatch.setenv(FR_DUMP_TEMP_FILE_ENV, "/tmp/nccl_fr_rank")
    monkeypatch.setenv(NCCL_DEBUG_INFO_TEMP_FILE_ENV, "/tmp/nccl_fr_rank")
    monkeypatch.setattr("skyrl_train.utils.utils.peer_access_supported", lambda **_: False)
    monkeypatch.setattr("skyrl_train.utils.utils.sync_registries", lambda: None)
    ray_init: dict[str, object] = {}
    monkeypatch.setattr(ray, "init", lambda **kwargs: ray_init.update(kwargs))

    initialize_ray(cfg)

    artifact_root = checkpoint_path.parent / "debug"
    manifests = list((artifact_root / "processes").glob("driver.*.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text())
    assert manifest["role"] == "driver"
    expected_prefix = str(artifact_root / "flight_recorder" / "nccl_fr_rank_")
    runtime_environment = ray_init["runtime_env"]["env_vars"]
    assert runtime_environment[FR_DUMP_TEMP_FILE_ENV] == expected_prefix
    assert runtime_environment[NCCL_DEBUG_INFO_TEMP_FILE_ENV] == expected_prefix
    assert runtime_environment["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] == "300"


def test_ray_initialization_with_telemetry_does_not_require_debug_artifacts(monkeypatch):
    cfg = example_dummy_config()
    OmegaConf.update(cfg, "trainer.debug_mode", "off")
    monkeypatch.setenv(TELEMETRY_ENDPOINT_ENV, "http://telemetry.example/v1")
    monkeypatch.setenv(RUN_ID_ENV, "nightly-run")
    monkeypatch.setenv(EXECUTION_UID_ENV, "iris:attempt")
    monkeypatch.delenv(DEBUG_MODE_ENV, raising=False)
    monkeypatch.delenv(DEBUG_ARTIFACT_DIR_ENV, raising=False)
    monkeypatch.setattr("skyrl_train.utils.utils.peer_access_supported", lambda **_: False)
    monkeypatch.setattr("skyrl_train.utils.utils.sync_registries", lambda: None)
    ray_init: dict[str, object] = {}
    monkeypatch.setattr(ray, "init", lambda **kwargs: ray_init.update(kwargs))

    initialize_ray(cfg)

    runtime_environment = ray_init["runtime_env"]["env_vars"]
    assert runtime_environment[TELEMETRY_ENDPOINT_ENV] == "http://telemetry.example/v1"
    assert runtime_environment[RUN_ID_ENV] == "nightly-run"
    assert EXECUTION_UID_ENV not in runtime_environment
    assert DEBUG_MODE_ENV not in runtime_environment
