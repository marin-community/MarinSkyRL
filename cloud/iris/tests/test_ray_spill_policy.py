import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cloud.iris import start_rl_iris_controller as controller  # noqa: E402


def test_remote_spilling_requires_explicit_opt_in(monkeypatch):
    monkeypatch.delenv("OT_AGENT_RAY_SPILL_TO_R2", raising=False)

    assert controller._ray_spill_uri("s3://marin-us-east-02a/iris/rl-rdv/job") is None

    monkeypatch.setenv("OT_AGENT_RAY_SPILL_TO_R2", "1")
    assert controller._ray_spill_uri("s3://marin-us-east-02a/iris/rl-rdv/job") == (
        "s3://marin-us-east-02a/iris/rl-rdv/job/ray_spill"
    )


def test_remote_spilling_fails_when_backend_dependency_is_missing(monkeypatch):
    monkeypatch.setenv("OT_AGENT_RAY_SPILL_TO_R2", "1")
    monkeypatch.setitem(sys.modules, "boto3", None)

    with pytest.raises(RuntimeError, match="requires boto3"):
        controller._ray_spill_uri("s3://marin-us-east-02a/iris/rl-rdv/job")


def test_default_spill_backend_uses_launcher_owned_local_storage(monkeypatch):
    monkeypatch.delenv("OT_AGENT_RAY_SPILL_DIR", raising=False)
    local_spill_dir = controller._ray_local_spill_dir()

    assert controller._ray_spill_flags(None, local_spill_dir) == ["--object-spilling-directory=/tmp/skyrl-ray-spill"]


def test_worker_uses_launcher_owned_local_spill_directory(monkeypatch):
    commands = []
    monkeypatch.delenv("OT_AGENT_RAY_SPILL_DIR", raising=False)
    monkeypatch.setattr(controller.subprocess, "run", lambda command, check: commands.append(command))

    controller.ray_start_worker("10.0.0.1", 6379, "10.0.0.2")

    assert "--object-spilling-directory=/tmp/skyrl-ray-spill" in commands[0]
