import json

import pytest
from omegaconf import OmegaConf

from cloud.iris import export_hf_checkpoint
from cloud.iris.export_hf_checkpoint import ExportJobSpec, argument_parser, build_command, manual_spec, request_spec
from skyrl_train.config.utils import get_default_config
from skyrl_train.hf_export import (
    protected_hf_export_steps,
    read_hf_export_request,
    write_hf_export_request,
)
from skyrl_train.hf_export_schema import HFExportRequest, HFExportStatus, HFUploadMode
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.utils.trainer_utils import cleanup_old_checkpoints
from skyrl_train.utils.utils import validate_hf_export_config


def _trainer_config(
    tmp_path,
    *,
    model_path: str = "org/model",
    model_source_uri: str | None = None,
    model_source_identity: str | None = None,
):
    return OmegaConf.create(
        {
            "trainer": {
                "ckpt_path": str(tmp_path / "checkpoints"),
                "export_path": str(tmp_path / "exports"),
                "placement": {"policy_num_nodes": 2, "policy_num_gpus_per_node": 4},
                "policy": {
                    "model": {
                        "path": model_path,
                        "source_uri": model_source_uri,
                        "source_identity": model_source_identity,
                    }
                },
                "hf_hub_repo_id": "org/exported-model",
            }
        }
    )


def _queue_export(tmp_path, **config_overrides):
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.cfg = _trainer_config(tmp_path, **config_overrides)
    trainer.all_timings = {}
    trainer.global_step = 10
    checkpoint = tmp_path / "checkpoints" / "global_step_10"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.pt").write_bytes(b"complete")
    trainer.handle_hf_export()
    return checkpoint


def _export_command(request):
    return build_command(
        ExportJobSpec(
            request=request,
            rl_config="config.yaml",
            cluster="cw-rno2a",
            priority="batch",
            job_name="export-step-10",
            timeout=7200,
            no_wait=False,
        )
    )


def _command_options(command):
    return {command[index]: command[index + 1] for index in range(len(command) - 1) if command[index].startswith("--")}


def test_normal_hf_save_records_rerunnable_request_without_live_export(tmp_path):
    checkpoint = _queue_export(tmp_path)

    request = read_hf_export_request(str(checkpoint))
    assert request is not None
    assert request.status is HFExportStatus.PENDING
    assert request.step == 10
    assert request.checkpoint_base_path == str(tmp_path / "checkpoints")
    assert request.checkpoint_path == str(checkpoint)
    assert request.export_path == str(tmp_path / "exports")
    assert request.hf_hub_repo_id == "org/exported-model"
    assert not (tmp_path / "exports").exists()


def test_export_request_preserves_durable_source_for_task_local_model(tmp_path):
    checkpoint = _queue_export(
        tmp_path,
        model_path="/tmp/materialized-model",
        model_source_uri="s3://models/policy",
        model_source_identity="policy@abc123",
    )

    request = read_hf_export_request(str(checkpoint))
    assert request is not None
    assert request.model_path == "/tmp/materialized-model"
    assert request.model_source_uri == "s3://models/policy"
    assert request.model_source_identity == "policy@abc123"


def test_export_request_rejects_task_local_model_without_durable_source():
    with pytest.raises(ValueError, match="task-local model_path"):
        HFExportRequest(
            step=10,
            checkpoint_base_path="s3://bucket/checkpoints",
            checkpoint_path="s3://bucket/checkpoints/global_step_10",
            export_path="s3://bucket/exports",
            model_path="/tmp/materialized-model",
            num_nodes=8,
            gpus_per_node=8,
        )


def test_checkpoint_cleanup_retains_pending_export_source(tmp_path):
    for step in (5, 10, 15):
        (tmp_path / f"global_step_{step}").mkdir()

    cleanup_old_checkpoints(str(tmp_path), max_checkpoints=1, protected_steps={5})

    assert sorted(path.name for path in tmp_path.iterdir()) == ["global_step_15", "global_step_5"]


def test_pending_export_request_protects_its_checkpoint(tmp_path):
    checkpoint = tmp_path / "global_step_5"
    checkpoint.mkdir()
    request = HFExportRequest(
        step=5,
        checkpoint_base_path=str(tmp_path),
        checkpoint_path=str(checkpoint),
        export_path=str(tmp_path / "exports"),
        model_path="org/model",
        num_nodes=2,
        gpus_per_node=4,
    )
    write_hf_export_request(request)

    assert protected_hf_export_steps(str(tmp_path)) == {5}

    write_hf_export_request(request.with_status(HFExportStatus.COMPLETE, last_exit_code=0))
    assert protected_hf_export_steps(str(tmp_path)) == set()


@pytest.mark.parametrize("request_contents", ["{", "{}"])
def test_corrupt_export_request_protects_its_checkpoint(tmp_path, request_contents):
    checkpoint = tmp_path / "global_step_5"
    checkpoint.mkdir()
    (checkpoint / "hf_export_request.json").write_text(request_contents)

    assert protected_hf_export_steps(str(tmp_path)) == {5}


def test_hf_export_interval_must_be_checkpoint_aligned():
    cfg = OmegaConf.create({"trainer": {"ckpt_interval": 3, "hf_save_interval": 5}})

    with pytest.raises(ValueError, match="multiple of trainer.ckpt_interval"):
        validate_hf_export_config(cfg)


def test_default_hf_export_interval_tracks_checkpoint_override():
    cfg = get_default_config()
    cfg.trainer.ckpt_interval = 10

    assert cfg.trainer.hf_save_interval == 10
    validate_hf_export_config(cfg)


def test_normal_training_rejects_in_band_hub_upload():
    cfg = OmegaConf.create(
        {
            "trainer": {
                "callbacks": [
                    {"type": "checkpoint", "save_steps": 5},
                    {"type": "hf_model_save", "save_steps": 5},
                    {"type": "hf_hub_upload", "upload_steps": 5, "repo_id": "org/model"},
                ],
            }
        }
    )

    with pytest.raises(ValueError, match="produced out of band"):
        validate_hf_export_config(cfg)


def test_export_job_owns_timeout_and_waits_for_completion():
    request = HFExportRequest(
        step=10,
        checkpoint_base_path="s3://bucket/run/checkpoints",
        checkpoint_path="s3://bucket/run/checkpoints/global_step_10",
        export_path="s3://bucket/run/exports",
        model_path="org/model",
        num_nodes=8,
        gpus_per_node=8,
        hf_hub_repo_id="org/exported-model",
        hf_hub_private=True,
        hf_hub_revision="main",
        hf_upload_mode=HFUploadMode.LATEST,
    )
    command = _export_command(request)
    options = _command_options(command)
    overrides = {
        value.split("=", 1)[0]: value.split("=", 1)[1]
        for index, value in enumerate(command)
        if index > 0 and command[index - 1] == "--skyrl_override"
    }

    assert options["--entrypoint"] == "skyrl_train.entrypoints.checkpoint_export"
    assert overrides["++checkpoint_export.step"] == "10"
    assert overrides["++checkpoint_export.checkpoint_path"] == request.checkpoint_path
    assert overrides["++checkpoint_export.export_root"] == request.export_path
    assert not any(key.startswith("++trainer.callbacks") for key in overrides)
    assert options["--timeout"] == "7200"
    assert "--no-wait" not in command
    assert overrides["++checkpoint_export.hf_hub_repo_id"] == "org/exported-model"


def test_export_job_materializes_request_model_source():
    request = HFExportRequest(
        step=10,
        checkpoint_base_path="s3://bucket/run/checkpoints",
        checkpoint_path="s3://bucket/run/checkpoints/global_step_10",
        export_path="s3://bucket/run/exports",
        model_path="/tmp/materialized-model",
        model_source_uri="s3://models/policy",
        model_source_identity="policy@abc123",
        num_nodes=8,
        gpus_per_node=8,
    )

    command = _export_command(request)
    options = _command_options(command)

    assert options["--model_path"] == "/tmp/materialized-model"
    assert options["--model-source-uri"] == "s3://models/policy"
    assert options["--model-source-identity"] == "policy@abc123"


def test_request_rejects_operator_override_instead_of_ignoring_it(tmp_path):
    checkpoint = tmp_path / "checkpoints" / "global_step_10"
    checkpoint.mkdir(parents=True)
    write_hf_export_request(
        HFExportRequest(
            step=10,
            checkpoint_base_path=str(checkpoint.parent),
            checkpoint_path=str(checkpoint),
            export_path=str(tmp_path / "exports"),
            model_path="org/model",
            num_nodes=2,
            gpus_per_node=4,
        )
    )
    parser = argument_parser()
    args = parser.parse_args(["--request", str(checkpoint), "--rl_config", "config.yaml", "--num-nodes", "8"])

    with pytest.raises(SystemExit):
        request_spec(args, parser)


def test_request_mode_rejects_task_local_model_without_source_before_submission(tmp_path):
    checkpoint = tmp_path / "checkpoints" / "global_step_10"
    checkpoint.mkdir(parents=True)
    (checkpoint / "hf_export_request.json").write_text(
        json.dumps(
            {
                "step": 10,
                "checkpoint_base_path": str(checkpoint.parent),
                "checkpoint_path": str(checkpoint),
                "export_path": str(tmp_path / "exports"),
                "model_path": "/tmp/materialized-model",
                "num_nodes": 8,
                "gpus_per_node": 8,
                "status": "pending",
                "hf_upload_mode": "latest",
            }
        )
    )
    parser = argument_parser()
    args = parser.parse_args(["--request", str(checkpoint), "--rl_config", "config.yaml"])

    with pytest.raises(SystemExit):
        request_spec(args, parser)


def test_manual_export_requires_explicit_checkpoint_geometry():
    parser = argument_parser()
    args = parser.parse_args(
        [
            "--ckpt_path",
            "/checkpoint",
            "--step",
            "10",
            "--model_path",
            "org/model",
            "--rl_config",
            "config.yaml",
        ]
    )

    with pytest.raises(SystemExit):
        manual_spec(args, parser)


@pytest.mark.parametrize(
    ("exit_code", "expected_status"),
    [(0, HFExportStatus.COMPLETE), (17, HFExportStatus.PENDING)],
)
def test_export_request_records_lifecycle_result(tmp_path, monkeypatch, exit_code, expected_status):
    checkpoint = tmp_path / "checkpoints" / "global_step_10"
    checkpoint.mkdir(parents=True)
    request = HFExportRequest(
        step=10,
        checkpoint_base_path=str(tmp_path / "checkpoints"),
        checkpoint_path=str(checkpoint),
        export_path=str(tmp_path / "exports"),
        model_path="org/model",
        num_nodes=2,
        gpus_per_node=4,
    )
    write_hf_export_request(request)
    monkeypatch.setattr(export_hf_checkpoint.subprocess, "call", lambda *args, **kwargs: exit_code)
    monkeypatch.setattr(
        "sys.argv",
        [
            "export_hf_checkpoint.py",
            "--request",
            str(checkpoint),
            "--rl_config",
            "config.yaml",
            "--timeout",
            "7200",
        ],
    )

    with pytest.raises(SystemExit) as result:
        export_hf_checkpoint.main()

    assert result.value.code == exit_code
    updated = read_hf_export_request(str(checkpoint))
    assert updated is not None
    assert updated.status is expected_status
    assert updated.attempts == 1
    assert updated.timeout == 7200
    assert updated.last_exit_code == exit_code
