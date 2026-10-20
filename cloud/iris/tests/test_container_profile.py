"""The launch config's Iris container profile reaches the job submission."""

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from iris.rpc import job_pb2

import cloud.iris.iris_backend as launcher
from cloud.iris.launch_config import load_launch_config
from cloud.iris.tests.test_launch_config import _raw_config


def _submitted_profile(tmp_path: Path, monkeypatch, profile: str | None) -> int:
    raw = _raw_config()
    raw["run"]["submission"] = "detach"
    if profile is not None:
        raw["iris"]["container_profile"] = profile
    path = tmp_path / "launch.yaml"
    path.write_text(yaml.safe_dump(raw))
    args = launcher._iris_submission_state(path, load_launch_config(path))
    submitted: dict = {}

    class Client:
        def submit(self, **kwargs):
            submitted.update(kwargs)
            return SimpleNamespace(job_id="/user/smoke")

    @contextmanager
    def ambient_client():
        yield Client()

    monkeypatch.setattr(launcher, "build_runtime_bundle", lambda _commit: tmp_path)
    monkeypatch.setattr(launcher, "build_config_task_command", lambda _args: ["true"])
    monkeypatch.setattr(launcher, "_ambient_in_cluster_client", lambda _workspace: ambient_client())
    outcome = launcher.launch(args, "a" * 40)
    assert outcome.job_state == "submitted"
    return submitted["container_profile"]


@pytest.mark.parametrize(
    ("profile", "expected"),
    [
        ("CONTAINER_PROFILE_PRIVILEGED", job_pb2.CONTAINER_PROFILE_PRIVILEGED),
        (None, job_pb2.CONTAINER_PROFILE_UNSPECIFIED),
    ],
)
def test_launch_config_container_profile_reaches_submission(tmp_path, monkeypatch, profile, expected):
    assert _submitted_profile(tmp_path, monkeypatch, profile) == expected


def test_unknown_container_profile_is_rejected_before_submission(tmp_path):
    raw = _raw_config()
    raw["iris"]["container_profile"] = "CONTAINER_PROFILE_ROOT"
    path = tmp_path / "launch.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="CONTAINER_PROFILE_ROOT"):
        launcher._iris_submission_state(path, load_launch_config(path))
