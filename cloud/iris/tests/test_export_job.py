"""Independent conversion and retries through the durable training/export contract."""

import json
import shutil
import subprocess
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from cloud.iris.export_job import execute_export
from cloud.iris.job import _write_json, execute_job
from cloud.iris.iris_backend import IrisLaunchOutcome
from cloud.iris.protocol import (
    AttemptState,
    RuntimeIdentity,
    SkyRLExportSpec,
    SkyRLExportRequest,
    SkyRLExportPaths,
    export_spec,
)
from cloud.iris.runtime_environment import RuntimeProfile
from cloud.iris.tests.test_job import (
    FakeLaunchBackend,
    _spec,
    _write_terminal_training_outputs,
    _write_policy_export,
    runtime_checkout as runtime_checkout,  # pytest fixture re-export
)
from marinskyrl.export_completion import ExportReceipt


class ConversionService:
    def __init__(self, *, fail_after_write=False):
        self.submissions = []
        self.configurations = []
        self.fail_after_write = fail_after_write

    def export(self, request):
        self.submissions.append(request)
        self.configurations.append(Path(request.config_path).read_bytes())
        assert request.global_step == 8
        _write_policy_export(request.export_root)
        _write_json(
            request.receipt_uri,
            ExportReceipt(
                request.request_fingerprint,
                request.attempt_id,
                f"{request.export_root}/global_step_8/policy",
                8,
            ).to_dict(),
        )
        if self.fail_after_write:
            raise subprocess.CalledProcessError(1, ["iris-export"])


@pytest.fixture
def completed_training(tmp_path, runtime_checkout):  # noqa: F811 - pytest injects the imported fixture
    training = _spec(tmp_path, runtime_checkout[1])
    _write_terminal_training_outputs(training)
    result = execute_job(training, backend=FakeLaunchBackend(IrisLaunchOutcome("training-job", "succeeded", 0)))
    assert result.state == AttemptState.SUCCEEDED
    export_root = tmp_path / "model"
    export = SkyRLExportSpec(
        SkyRLExportRequest(
            training.request.output.terminal_manifest_uri,
            "export-1",
            SkyRLExportPaths(
                str(export_root / "exports"),
                str(export_root / "attempts"),
                str(export_root / "terminal.json"),
            ),
        ),
        training.execution,
    )
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
    assert (
        json.loads(Path(export.request.output.terminal_manifest_uri).read_text())["response"]["model"]["global_step"]
        == 8
    )


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


def test_export_startup_failure_preserves_checkpoint_for_retry(completed_training):
    training, export = completed_training
    export = replace(export, execution=replace(export.execution, max_retries=1))
    terminal_path = Path(training.request.output.terminal_manifest_uri.removeprefix("file://"))
    original_training = terminal_path.read_bytes()

    class StartupFailure:
        def export(self, request):
            assert request.max_retries == 1
            raise subprocess.CalledProcessError(1, ["export-bootstrap"])

    failed = execute_export(export, backend=StartupFailure())
    assert failed.state == AttemptState.FAILED
    assert terminal_path.read_bytes() == original_training
    assert not Path(export.request.output.terminal_manifest_uri).exists()
    service = ConversionService()
    retry = replace(export, request=replace(export.request, attempt_id="export-after-startup-failure"))
    completed = execute_export(retry, backend=service)
    assert completed.state == AttemptState.SUCCEEDED and not completed.reused_export
    assert completed.model.global_step == 8
    assert terminal_path.read_bytes() == original_training
    assert service.submissions[0].max_retries == 1


def test_exporter_revision_is_bound_separately_from_training(completed_training, runtime_checkout):  # noqa: F811
    training, export = completed_training
    original = ConversionService()
    # Capture the production conversion configuration before advancing the exporter checkout.
    execute_export(export, backend=original)
    first_manifest = json.loads(Path(export.request.output.terminal_manifest_uri).read_text())
    checkout, training_commit = runtime_checkout
    marker = checkout / "cloud/iris/task_runtime.py"
    marker.write_text(marker.read_text() + "EXPORT_RETRY_REVISION = True\n")
    subprocess.run(["git", "add", "cloud/iris/task_runtime.py"], cwd=checkout, check=True)
    subprocess.run(["git", "commit", "-qm", "export-only fixture revision"], cwd=checkout, check=True)
    exporter_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
    assert exporter_commit != training_commit
    revised = replace(
        export,
        request=replace(
            export.request,
            attempt_id="new-exporter",
            exporter_runtime=RuntimeIdentity(exporter_commit, RuntimeProfile.FSDP_EXPORT),
            output=replace(
                export.request.output,
                terminal_manifest_uri=export.request.output.terminal_manifest_uri + ".new",
                export_root=export.request.output.export_root + ".new",
            ),
        ),
    )
    service = ConversionService()
    completed = execute_export(export_spec(json.loads(json.dumps(asdict(revised)))), backend=service)
    assert completed.runtime == training.request.runtime
    assert completed.exporter_runtime.commit == exporter_commit
    assert completed.state == AttemptState.SUCCEEDED and not completed.reused_export
    second_manifest = json.loads(Path(revised.request.output.terminal_manifest_uri).read_text())
    assert second_manifest["export_receipt_uri"] != first_manifest["export_receipt_uri"]
    assert service.submissions[0].global_step == original.submissions[0].global_step
    assert service.submissions[0].checkpoint_root == original.submissions[0].checkpoint_root
    assert service.configurations == original.configurations


def test_exporter_revision_mismatch_is_rejected_before_conversion(completed_training):
    _, export = completed_training
    request = replace(export.request, exporter_runtime=RuntimeIdentity("0" * 40, RuntimeProfile.FSDP_EXPORT))
    service = ConversionService()
    with pytest.raises(ValueError, match="does not match requested"):
        execute_export(replace(export, request=request), backend=service)
    assert service.submissions == []


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
