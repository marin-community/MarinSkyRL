import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cloud.iris import task_runtime as runtime  # noqa: E402
from cloud.iris.ray_storage import (  # noqa: E402
    DEFAULT_RAY_SPILL_DIR,
    LocalRaySpillTarget,
    R2RaySpillTarget,
    RaySpillBackend,
)


def test_remote_spilling_requires_explicit_opt_in():
    local = runtime._ray_spill_target(
        "s3://marin-us-east-02a/iris/rl-rdv/job",
        RaySpillBackend.LOCAL,
        DEFAULT_RAY_SPILL_DIR,
    )

    remote = runtime._ray_spill_target(
        "s3://marin-us-east-02a/iris/rl-rdv/job",
        RaySpillBackend.R2,
        DEFAULT_RAY_SPILL_DIR,
    )

    assert local == LocalRaySpillTarget(DEFAULT_RAY_SPILL_DIR)
    assert remote == R2RaySpillTarget("s3://marin-us-east-02a/iris/rl-rdv/job/ray_spill")


def test_remote_spilling_fails_when_backend_dependency_is_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "boto3", None)

    with pytest.raises(RuntimeError, match="requires boto3"):
        runtime._ray_spill_target(
            "s3://marin-us-east-02a/iris/rl-rdv/job",
            RaySpillBackend.R2,
            DEFAULT_RAY_SPILL_DIR,
        )


def test_remote_spilling_rejects_non_s3_rendezvous():
    with pytest.raises(ValueError, match="requires an s3:// rendezvous directory"):
        runtime._ray_spill_target("/shared/rendezvous/job", RaySpillBackend.R2, DEFAULT_RAY_SPILL_DIR)


def test_remote_spilling_rejects_local_directory_override():
    with pytest.raises(ValueError, match="only applies.*local"):
        runtime._ray_spill_target(
            "s3://marin-us-east-02a/iris/rl-rdv/job",
            RaySpillBackend.R2,
            "/local/nvme/ray-spill",
        )


def test_ray_worker_creates_its_local_spill_directory(tmp_path, monkeypatch):
    spill_dir = tmp_path / "ray-spill"
    commands = []
    monkeypatch.setattr(runtime.subprocess, "run", lambda command, **_kwargs: commands.append(command))

    runtime.ray_start_worker("10.0.0.1", 6379, "10.0.0.2", LocalRaySpillTarget(str(spill_dir)))

    assert spill_dir.is_dir()
    assert f"--object-spilling-directory={spill_dir}" in commands[0]
