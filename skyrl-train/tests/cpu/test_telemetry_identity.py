from skyrl_train.telemetry import TRAINER_ROLE, TelemetryConfig, _resources


def test_telemetry_resources_use_run_id() -> None:
    config = TelemetryConfig(
        endpoint="http://finelog:8080/v1/telemetry",
        run_id="/atqamar/iceball-micro-0",
        execution_uid="iris:/atqamar/iceball-micro-0/1:attempt:2",
    )

    resources = _resources(config, TRAINER_ROLE)

    assert resources["run_id"] == "/atqamar/iceball-micro-0"
    assert "root_run_uid" not in resources


def test_telemetry_config_without_override_uses_iris_attempt_uid(monkeypatch) -> None:
    monkeypatch.setenv("IRIS_ATTEMPT_UID", "01JABCDEF0123456789")
    monkeypatch.delenv("SKYRL_EXECUTION_UID", raising=False)

    assert TelemetryConfig.from_environment().execution_uid == "iris:01JABCDEF0123456789"


def test_telemetry_config_execution_uid_override_takes_precedence(monkeypatch) -> None:
    monkeypatch.setenv("IRIS_ATTEMPT_UID", "01JABCDEF0123456789")
    monkeypatch.setenv("SKYRL_EXECUTION_UID", "iris:/other/task/0:attempt:0")

    assert TelemetryConfig.from_environment().execution_uid == "iris:/other/task/0:attempt:0"


def test_telemetry_config_without_attempt_uid_leaves_execution_uid_unset(monkeypatch) -> None:
    monkeypatch.delenv("IRIS_ATTEMPT_UID", raising=False)
    monkeypatch.delenv("SKYRL_EXECUTION_UID", raising=False)

    assert TelemetryConfig.from_environment().execution_uid is None
