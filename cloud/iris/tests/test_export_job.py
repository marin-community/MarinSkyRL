"""Independent conversion and retries through the durable training/export contract."""

import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from cloud.iris.export_job import execute_export
from cloud.iris.job import _write_json, execute_job
from cloud.iris.iris_backend import IrisLaunchOutcome
from cloud.iris.protocol import AttemptState, SkyRLExportSpec, SkyRLExportRequest, SkyRLExportPaths
from cloud.iris.tests.test_job import (
    FakeLaunchBackend, _spec, _write_terminal_training_outputs, _write_policy_export,
    runtime_checkout,  # noqa: F401 - pytest fixture
)
from marinskyrl.export_completion import ExportReceipt


class ConversionService:
    def __init__(self, *, fail_after_write=False):
        self.submissions = []
        self.fail_after_write = fail_after_write

    def export(self, request):
        self.submissions.append(request)
        assert request.global_step == 8
        _write_policy_export(request.export_root)
        _write_json(request.receipt_uri, ExportReceipt(
            request.request_fingerprint, request.attempt_id,
            f"{request.export_root}/global_step_8/policy", 8,
        ).to_dict())
        if self.fail_after_write:
            raise subprocess.CalledProcessError(1, ["iris-export"])


@pytest.fixture
def completed_training(tmp_path, runtime_checkout):
    training = _spec(tmp_path, runtime_checkout[1])
    _write_terminal_training_outputs(training)
    result = execute_job(training, backend=FakeLaunchBackend(IrisLaunchOutcome("training-job", "succeeded", 0)))
    assert result.state == AttemptState.SUCCEEDED
    export_root = tmp_path / "model"
    export = SkyRLExportSpec(SkyRLExportRequest(
        training.request.output.terminal_manifest_uri, "export-1", SkyRLExportPaths(
            str(export_root / "exports"), str(export_root / "attempts"), str(export_root / "terminal.json"),
        ),
    ), training.execution)
    return training, export


def test_export_uses_recorded_checkpoint_even_when_latest_advances(completed_training):
    training, export = completed_training
    root = Path(training.request.output.checkpoint_root.removeprefix("file://"))
    (root / "latest_ckpt_global_step.txt").write_text("9")
    service = ConversionService()
    result = execute_export(export, backend=service)
    assert result.state == AttemptState.SUCCEEDED
    assert result.model.global_step == 8
    assert result.training_iris_job_id == "training-job"
    assert json.loads(Path(export.request.output.terminal_manifest_uri).read_text())["response"]["model"]["global_step"] == 8


def test_export_failure_keeps_training_success_and_retry_reuses_verified_files(completed_training):
    training, export = completed_training
    service = ConversionService(fail_after_write=True)
    result = execute_export(export, backend=service)
    assert result.state == AttemptState.FAILED
    training_manifest = Path(training.request.output.terminal_manifest_uri.removeprefix("file://"))
    assert json.loads(training_manifest.read_text())["response"]["state"] == "succeeded"
    assert not Path(export.request.output.terminal_manifest_uri).exists()
    # A durable export remains reusable after the temporary native checkpoint expires.
    shutil.rmtree(Path(training.request.output.checkpoint_root.removeprefix("file://")))
    retry = replace(export, request=replace(export.request, attempt_id="export-2"))
    result = execute_export(retry, backend=service)
    assert result.state == AttemptState.SUCCEEDED
    assert result.reused_export
    assert len(service.submissions) == 1


def test_export_expired_checkpoint_fails_before_gpu_submission(completed_training):
    training, export = completed_training
    shutil.rmtree(Path(training.request.output.checkpoint_root.removeprefix("file://")))
    service = ConversionService()
    result = execute_export(export, backend=service)
    assert result.state == AttemptState.FAILED
    assert service.submissions == []
    assert not Path(export.request.output.terminal_manifest_uri).exists()


def test_export_does_not_adopt_unreceipted_hf_files(completed_training):
    training, export = completed_training
    _write_policy_export(export.request.output.export_root)
    shutil.rmtree(Path(training.request.output.checkpoint_root.removeprefix("file://")))
    service = ConversionService()
    result = execute_export(export, backend=service)
    assert result.state == AttemptState.FAILED
    assert service.submissions == []
