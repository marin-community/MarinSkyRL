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


@pytest.mark.parametrize("failures", [1, 3])
def test_model_staging_retries_timed_out_file_without_recopying_completed_files(tmp_path, monkeypatch, failures):
    from pathlib import Path
    from fsspec.exceptions import FSTimeoutError
    from cloud.iris import artifacts

    source = tmp_path / "source"
    source.mkdir()
    (source / "a-config.json").write_bytes(b"configuration")
    (source / "b-weights").write_bytes(b"complete weights")
    destination = tmp_path / "materialized"
    original = LocalFileSystem.get_file
    calls = []

    def download(filesystem, source_path, local_path, **kwargs):
        name = Path(source_path).name
        calls.append(name)
        if name == "b-weights" and calls.count(name) <= failures:
            assert not Path(local_path).exists()
            Path(local_path).write_bytes(b"partial")
            raise FSTimeoutError("multipart read timed out")
        return original(filesystem, source_path, local_path, **kwargs)

    monkeypatch.setattr(LocalFileSystem, "get_file", download)
    waits = []
    monkeypatch.setattr(artifacts.time, "sleep", waits.append)
    request = artifacts.ArtifactSource(source.as_uri(), "immutable-test", str(destination))
    if failures == 3:
        with pytest.raises(FSTimeoutError):
            artifacts.materialize(request)
        assert not destination.exists()
        assert not list(tmp_path.glob(".materialized.staging-*"))
        assert waits == [2, 4]
    else:
        result = artifacts.materialize(request)
        assert (destination / "b-weights").read_bytes() == b"complete weights"
        assert len(result.files) == 2
        assert waits == [2]
        assert (destination / artifacts.SOURCE_MANIFEST_FILENAME).is_file()
    assert calls.count("a-config.json") == 1
    assert calls.count("b-weights") == min(failures + 1, 3)


def test_s3_staging_uses_longer_read_timeout_and_virtual_addressing(monkeypatch):
    from cloud.iris import artifacts

    sentinel = object()

    def resolve(uri, *, storage_options):
        assert uri == "s3://bucket/model"
        assert storage_options["config_kwargs"] == {"s3": {"addressing_style": "virtual"}, "read_timeout": 180}
        return sentinel, None, ["bucket/model"]

    monkeypatch.delenv(artifacts.S3_ADDRESSING_STYLE_ENV, raising=False)
    monkeypatch.setattr(artifacts.fsspec, "get_fs_token_paths", resolve)
    assert artifacts.fs_and_path("s3://bucket/model") == (sentinel, "bucket/model")
