from skyrl_train.telemetry import TelemetryConfig


def test_telemetry_config_resolves_the_execution_uid(monkeypatch) -> None:
    monkeypatch.setenv("IRIS_ATTEMPT_UID", "01JABCDEF0123456789")
    monkeypatch.delenv("SKYRL_EXECUTION_UID", raising=False)
    assert TelemetryConfig.from_environment().execution_uid == "iris:01JABCDEF0123456789"

    monkeypatch.setenv("SKYRL_EXECUTION_UID", "iris:/other/task/0:attempt:0")
    assert TelemetryConfig.from_environment().execution_uid == "iris:/other/task/0:attempt:0"

    monkeypatch.delenv("SKYRL_EXECUTION_UID")
    monkeypatch.delenv("IRIS_ATTEMPT_UID")
    assert TelemetryConfig.from_environment().execution_uid is None
