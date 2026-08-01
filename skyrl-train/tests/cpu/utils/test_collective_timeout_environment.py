from tests.cpu.util import example_dummy_config

from skyrl_train.utils.utils import prepare_runtime_environment


def test_runtime_environment_arms_independent_nccl_monitor(monkeypatch):
    monkeypatch.setattr("skyrl_train.utils.utils.peer_access_supported", lambda **_: False)
    monkeypatch.delenv("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", raising=False)
    monkeypatch.setenv("SKYRL_WORKER_NCCL_TIMEOUT_IN_S", "1800")

    env = prepare_runtime_environment(example_dummy_config())

    assert env["TORCH_NCCL_ENABLE_MONITORING"] == "1"
    assert env["TORCH_NCCL_ASYNC_ERROR_HANDLING"] == "1"
    assert env["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] == "300"


def test_monitor_heartbeat_is_capped_by_collective_timeout(monkeypatch):
    monkeypatch.setattr("skyrl_train.utils.utils.peer_access_supported", lambda **_: False)
    monkeypatch.setenv("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "300")
    monkeypatch.setenv("SKYRL_WORKER_NCCL_TIMEOUT_IN_S", "30")

    env = prepare_runtime_environment(example_dummy_config())

    assert env["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] == "30"
