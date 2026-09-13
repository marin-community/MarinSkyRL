import json
import signal

from marinskyrl.process_diagnostics import (
    ProcessOutcome,
    enable_fatal_stack_capture,
    install_live_stack_capture,
    write_process_outcome,
)
from marinskyrl.environment_contract import (
    DEBUG_ARTIFACT_DIR_ENV,
    DEBUG_MODE_ENV,
    PYTHONFAULTHANDLER_ENV,
    write_process_manifest,
)


def test_signal_outcome_preserves_abort_identity() -> None:
    outcome = ProcessOutcome.from_returncode(-signal.SIGABRT)

    assert outcome.kind == "signal"
    assert outcome.raw_returncode == -signal.SIGABRT
    assert outcome.signal == signal.SIGABRT
    assert outcome.signal_name == "SIGABRT"
    assert outcome.public_exit_code == 128 + signal.SIGABRT


def test_process_outcome_writes_atomic_secret_free_receipt(tmp_path) -> None:
    outcome, path = write_process_outcome(
        "skyrl entrypoint",
        -signal.SIGABRT,
        pid=123,
        environment={DEBUG_ARTIFACT_DIR_ENV: str(tmp_path), "OPENAI_API_KEY": "do-not-record"},
        metadata={"entrypoint": "skyrl_train.entrypoints.main_base"},
    )

    assert path is not None
    receipt = json.loads(path.read_text())
    assert receipt["kind"] == outcome.kind
    assert receipt["signal_name"] == "SIGABRT"
    assert receipt["pid"] == 123
    assert receipt["metadata"] == {"entrypoint": "skyrl_train.entrypoints.main_base"}
    assert "do-not-record" not in path.read_text()
    assert not list(path.parent.glob("*.tmp"))


def test_process_manifest_does_not_capture_managed_secrets(tmp_path) -> None:
    path = write_process_manifest(
        "worker",
        environment={
            DEBUG_ARTIFACT_DIR_ENV: str(tmp_path),
            "SKYRL_DEBUG_MODE": "light",
            "DAYTONA_API_KEY": "do-not-record",
        },
    )

    payload = path.read_text()
    assert "SKYRL_DEBUG_MODE" in payload
    assert "DAYTONA_API_KEY" not in payload
    assert "do-not-record" not in payload


def test_live_stack_capture_is_only_installed_when_configured(tmp_path, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        "marinskyrl.process_diagnostics.faulthandler.register",
        lambda signum, **kwargs: calls.append((signum, kwargs)),
    )

    assert install_live_stack_capture("worker", environment={}) is None
    path = install_live_stack_capture(
        "worker",
        environment={DEBUG_ARTIFACT_DIR_ENV: str(tmp_path), DEBUG_MODE_ENV: "distributed"},
    )

    assert path is not None
    assert path.parent == tmp_path / "stacks"
    assert calls[0][0] == signal.SIGUSR2
    assert calls[0][1]["all_threads"] is True
    assert calls[0][1]["chain"] is False
    assert calls[0][1]["file"].name == str(path)


def test_fatal_stack_capture_enables_an_active_interpreter(monkeypatch) -> None:
    enabled = []
    monkeypatch.setattr("marinskyrl.process_diagnostics.faulthandler.is_enabled", lambda: False)
    monkeypatch.setattr("marinskyrl.process_diagnostics.faulthandler.enable", lambda: enabled.append(True))

    assert enable_fatal_stack_capture(environment={}) is False
    assert enable_fatal_stack_capture(environment={PYTHONFAULTHANDLER_ENV: "1"}) is True
    assert enabled == [True]
