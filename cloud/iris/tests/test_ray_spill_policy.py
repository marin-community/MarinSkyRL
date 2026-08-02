import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cloud.iris import start_rl_iris_controller as controller  # noqa: E402
from cloud.iris.ray_storage import (  # noqa: E402
    DEFAULT_RAY_SPILL_DIR,
    LocalRaySpillTarget,
    R2RaySpillTarget,
    RaySpillBackend,
)


def test_remote_spilling_requires_explicit_opt_in():
    local = controller._ray_spill_target(
        "s3://marin-us-east-02a/iris/rl-rdv/job",
        RaySpillBackend.LOCAL,
        DEFAULT_RAY_SPILL_DIR,
    )

    remote = controller._ray_spill_target(
        "s3://marin-us-east-02a/iris/rl-rdv/job",
        RaySpillBackend.R2,
        DEFAULT_RAY_SPILL_DIR,
    )

    assert local == LocalRaySpillTarget(DEFAULT_RAY_SPILL_DIR)
    assert remote == R2RaySpillTarget("s3://marin-us-east-02a/iris/rl-rdv/job/ray_spill")


def test_remote_spilling_fails_when_backend_dependency_is_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "boto3", None)

    with pytest.raises(RuntimeError, match="requires boto3"):
        controller._ray_spill_target(
            "s3://marin-us-east-02a/iris/rl-rdv/job",
            RaySpillBackend.R2,
            DEFAULT_RAY_SPILL_DIR,
        )


def test_remote_spilling_rejects_non_s3_rendezvous():
    with pytest.raises(ValueError, match="requires an s3:// rendezvous directory"):
        controller._ray_spill_target("/shared/rendezvous/job", RaySpillBackend.R2, DEFAULT_RAY_SPILL_DIR)
