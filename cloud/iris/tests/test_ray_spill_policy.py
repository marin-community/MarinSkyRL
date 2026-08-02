import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cloud.iris.start_rl_iris_controller import _ray_spill_flags, _ray_spill_uri


def test_remote_spilling_requires_explicit_opt_in(monkeypatch):
    monkeypatch.delenv("OT_AGENT_RAY_SPILL_TO_R2", raising=False)

    assert _ray_spill_uri("s3://marin-us-east-02a/iris/rl-rdv/job") is None

    monkeypatch.setenv("OT_AGENT_RAY_SPILL_TO_R2", "1")
    assert _ray_spill_uri("s3://marin-us-east-02a/iris/rl-rdv/job") == (
        "s3://marin-us-east-02a/iris/rl-rdv/job/ray_spill"
    )


def test_default_spill_backend_uses_launcher_owned_local_storage(monkeypatch):
    monkeypatch.delenv("OT_AGENT_RAY_SPILL_DIR", raising=False)

    assert _ray_spill_flags(None) == ["--object-spilling-directory=/tmp/skyrl-ray-spill"]
