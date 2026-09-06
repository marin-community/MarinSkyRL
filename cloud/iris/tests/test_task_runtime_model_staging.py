import json
import sys

import pytest
from fsspec.implementations.local import LocalFileSystem

from cloud.iris import runtime_bundle, task_runtime
from cloud.iris.task_runtime import policy_chat_template_model


@pytest.mark.parametrize(
    ("prestage_model", "model_local_path", "expected"),
    [
        ("", "/tmp/materialized-model", "/tmp/materialized-model"),
        ("org/model", "/tmp/materialized-model", "org/model"),
    ],
)
def test_policy_chat_template_selects_materialized_model(
    prestage_model: str, model_local_path: str, expected: str
) -> None:
    assert policy_chat_template_model(prestage_model, model_local_path) == expected


def test_policy_chat_template_requires_a_materialized_model() -> None:
    with pytest.raises(ValueError, match="requires --prestage-model or --model-local-path"):
        policy_chat_template_model("", "")


@pytest.mark.parametrize("failure_mode", ["missing-marker", "missing-policy", "copy-failure"])
def test_invalid_checkpoint_staging_fails_before_any_ray_process_starts(tmp_path, monkeypatch, failure_mode):
    source = tmp_path / "source"
    source.mkdir()
    if failure_mode != "missing-marker":
        (source / "trainer_state.pt").write_bytes(b"completed marker")
    if failure_mode == "copy-failure":
        (source / "policy").mkdir()
        (source / "policy/__0_0.distcp").write_bytes(b"weights")

        def fail_download(self, source_path, local_path, **kwargs):
            raise OSError("fixture download interrupted")

        monkeypatch.setattr(LocalFileSystem, "get_file", fail_download)
    destination = tmp_path / "node-checkpoint"
    (tmp_path / runtime_bundle.BUNDLE_IDENTITY_FILE).write_text(
        json.dumps({"launcher_commit": "test-checkpoint-staging", "files": []})
    )
    monkeypatch.setattr(runtime_bundle, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "task_runtime.py",
            "--checkpoint-source-uri",
            source.as_uri(),
            "--checkpoint-local-path",
            str(destination),
            "--",
            "unused-driver",
        ],
    )
    process_marker = tmp_path / "process-started"

    def start_process(*args, **kwargs):
        process_marker.touch()
        raise AssertionError("Ray/bootstrap processes must not start for an invalid checkpoint")

    monkeypatch.setattr(task_runtime.subprocess, "Popen", start_process)
    error_type, expected_error = {
        "missing-marker": (ValueError, "Completed checkpoint marker"),
        "missing-policy": (ValueError, "trainer state and policy shards"),
        "copy-failure": (OSError, "fixture download interrupted"),
    }[failure_mode]
    with pytest.raises(error_type, match=expected_error):
        task_runtime.main()

    assert not process_marker.exists()
    assert not destination.exists()
    assert not list(tmp_path.glob(".node-checkpoint.staging-*"))
