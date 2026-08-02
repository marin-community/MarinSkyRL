import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cloud.iris import start_rl_iris_controller as controller  # noqa: E402
from cloud.iris.ray_storage import DEFAULT_RAY_SPILL_DIR, RaySpillBackend  # noqa: E402


def test_remote_spilling_requires_explicit_opt_in():
    assert controller._ray_spill_uri("s3://marin-us-east-02a/iris/rl-rdv/job", RaySpillBackend.LOCAL) is None

    assert controller._ray_spill_uri("s3://marin-us-east-02a/iris/rl-rdv/job", RaySpillBackend.R2) == (
        "s3://marin-us-east-02a/iris/rl-rdv/job/ray_spill"
    )


def test_remote_spilling_fails_when_backend_dependency_is_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "boto3", None)

    with pytest.raises(RuntimeError, match="requires boto3"):
        controller._ray_spill_uri("s3://marin-us-east-02a/iris/rl-rdv/job", RaySpillBackend.R2)


def test_remote_spilling_rejects_non_s3_rendezvous():
    with pytest.raises(ValueError, match="requires an s3:// rendezvous directory"):
        controller._ray_spill_uri("/shared/rendezvous/job", RaySpillBackend.R2)


def test_default_spill_backend_uses_launcher_owned_local_storage():
    assert controller._ray_spill_flags(None, DEFAULT_RAY_SPILL_DIR) == [
        "--object-spilling-directory=/tmp/skyrl-ray-spill"
    ]
