"""Run identity on the resource every MarinSkyRL telemetry row carries.

Finelog promotes a resource attribute to the ``telemetry_v1`` column of the same name,
and every Grafana panel joins on ``run_id``. marin#8379 renamed the attribute from
``root_run_uid``, which has no column and reaches ``resource_attributes_json`` only.

Run:
    python -m pytest skyrl-train/tests/cpu/test_telemetry_identity.py -v
"""

from skyrl_train.telemetry import TRAINER_ROLE, TelemetryConfig, _resources


def test_the_resource_carries_the_run_id_finelog_promotes() -> None:
    config = TelemetryConfig(
        endpoint="http://finelog:8080/v1/telemetry",
        run_id="/atqamar/iceball-micro-0",
        execution_uid="iris:/atqamar/iceball-micro-0/1:attempt:2",
    )

    resources = _resources(config, TRAINER_ROLE)

    assert resources["run_id"] == "/atqamar/iceball-micro-0"
    assert "root_run_uid" not in resources


def test_each_process_derives_its_own_task_attempt(monkeypatch) -> None:
    monkeypatch.setenv("IRIS_TASK_ID", "/atqamar/iceball-micro-0/1:2")
    monkeypatch.delenv("SKYRL_EXECUTION_UID", raising=False)

    assert TelemetryConfig.from_environment().execution_uid == "iris:/atqamar/iceball-micro-0/1:attempt:2"


def test_an_explicit_execution_uid_wins(monkeypatch) -> None:
    monkeypatch.setenv("IRIS_TASK_ID", "/atqamar/iceball-micro-0/1:2")
    monkeypatch.setenv("SKYRL_EXECUTION_UID", "iris:/other/task/0:attempt:0")

    assert TelemetryConfig.from_environment().execution_uid == "iris:/other/task/0:attempt:0"


def test_no_iris_task_leaves_the_execution_uid_unset(monkeypatch) -> None:
    monkeypatch.delenv("IRIS_TASK_ID", raising=False)
    monkeypatch.delenv("SKYRL_EXECUTION_UID", raising=False)

    assert TelemetryConfig.from_environment().execution_uid is None
