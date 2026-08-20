"""The writer for the telemetry variables ``skyrl_train.telemetry`` reads.

The producers, their tests and their documentation all merged; nothing ever wrote
``SKYRL_TELEMETRY_ENDPOINT`` or ``SKYRL_RUN_ID``, so every launch path exported
nothing. These tests pin the writer and the scopes its values reach.

Run:
    python -m pytest cloud/iris/tests/test_telemetry_env.py -v
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cloud.iris import telemetry_env  # noqa: E402
from cloud.iris.env_vars import (  # noqa: E402
    EXECUTION_UID_ENV,
    RUN_ID_ENV,
    TELEMETRY_ENDPOINT_ENV,
    EnvVarManager,
    EnvVarScope,
)


@dataclass(frozen=True)
class _JobInfo:
    job_id: str = "/atqamar/iceball-micro-0"
    task_id: str = "/atqamar/iceball-micro-0/1"
    attempt_id: int = 2


class _Client:
    def resolve_endpoint(self, name: str) -> str:
        assert name == "/system/log-server"
        return "http://finelog.marin.svc.cluster.local:8080/"


@dataclass(frozen=True)
class _Context:
    client: _Client


def _in_cluster(monkeypatch) -> None:
    monkeypatch.setattr(telemetry_env, "get_job_info", _JobInfo)
    monkeypatch.setattr(telemetry_env, "get_iris_ctx", lambda: _Context(_Client()))


def test_resolves_the_endpoint_and_identity_from_the_task_context(monkeypatch) -> None:
    _in_cluster(monkeypatch)

    assert telemetry_env.telemetry_environment() == {
        TELEMETRY_ENDPOINT_ENV: "http://finelog.marin.svc.cluster.local:8080/v1/telemetry",
        RUN_ID_ENV: "/atqamar/iceball-micro-0",
        EXECUTION_UID_ENV: "iris:/atqamar/iceball-micro-0/1:attempt:2",
    }


def test_an_explicit_run_id_wins_over_the_job_id(monkeypatch) -> None:
    _in_cluster(monkeypatch)

    resolved = telemetry_env.telemetry_environment(run_id="snowball-e6-muonh-0")

    assert resolved[RUN_ID_ENV] == "snowball-e6-muonh-0"


def test_no_cluster_context_leaves_telemetry_inert(monkeypatch) -> None:
    monkeypatch.setattr(telemetry_env, "get_job_info", lambda: None)
    monkeypatch.setattr(telemetry_env, "get_iris_ctx", lambda: None)

    assert telemetry_env.telemetry_environment() == {}


def test_an_unreachable_controller_does_not_reach_the_caller(monkeypatch) -> None:
    def explode() -> None:
        raise ConnectionError("controller is unreachable")

    monkeypatch.setattr(telemetry_env, "get_job_info", _JobInfo)
    monkeypatch.setattr(telemetry_env, "get_iris_ctx", explode)

    assert telemetry_env.telemetry_environment() == {}


def test_the_endpoint_and_run_id_reach_ray_workers_and_the_execution_uid_does_not() -> None:
    ambient = {
        TELEMETRY_ENDPOINT_ENV: "http://finelog:8080/v1/telemetry",
        RUN_ID_ENV: "/atqamar/iceball-micro-0",
        EXECUTION_UID_ENV: "iris:/atqamar/iceball-micro-0/1:attempt:2",
    }
    manager = EnvVarManager.from_config({}, environ=ambient)

    worker = manager.environment_for(EnvVarScope.RAY_WORKER)
    assert worker[TELEMETRY_ENDPOINT_ENV] == ambient[TELEMETRY_ENDPOINT_ENV]
    assert worker[RUN_ID_ENV] == ambient[RUN_ID_ENV]
    # One attempt spans several pods. Broadcasting the driver's would misattribute every
    # actor on every other node to the driver's task attempt.
    assert EXECUTION_UID_ENV not in worker
    assert manager.environment_for(EnvVarScope.TASK_RUNTIME)[EXECUTION_UID_ENV] == ambient[EXECUTION_UID_ENV]
