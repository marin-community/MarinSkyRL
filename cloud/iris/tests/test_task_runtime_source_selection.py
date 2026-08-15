"""Behavior tests for runtime source selection inside an Iris GPU task."""

import pytest

from cloud.iris import task_runtime


def test_training_driver_starts_from_the_immutable_runtime_checkout(monkeypatch) -> None:
    observed: dict[str, object] = {}
    process = object()

    def fake_popen(argv, **kwargs):
        observed["argv"] = argv
        observed.update(kwargs)
        return process

    monkeypatch.setattr(task_runtime.subprocess, "Popen", fake_popen)
    environment = {"SKYRL_HOME": "/app/marinskyrl", "PYTHONPATH": "/app/marinskyrl:/app"}

    launched = task_runtime.launch_training_driver(["python", "-m", "cloud.iris.training_driver"], environment)

    assert launched is process
    assert observed == {
        "argv": ["python", "-m", "cloud.iris.training_driver"],
        "env": environment,
        "cwd": "/app/marinskyrl",
        "start_new_session": True,
    }


def test_training_driver_requires_the_runtime_checkout() -> None:
    with pytest.raises(RuntimeError, match="SKYRL_HOME"):
        task_runtime.launch_training_driver(["python"], {})
