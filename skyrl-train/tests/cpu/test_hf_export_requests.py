import pytest
from omegaconf import OmegaConf

from cloud.iris import export_hf_checkpoint
from cloud.iris.export_hf_checkpoint import ExportJobSpec, argument_parser, build_command, request_spec
from skyrl_train.config.utils import get_default_config
from skyrl_train.hf_export import (
    pending_hf_export_steps,
    read_hf_export_request,
    write_hf_export_request,
)
from skyrl_train.hf_export_schema import HFExportRequest, HFExportStatus, HFUploadMode
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.utils.trainer_utils import cleanup_old_checkpoints
from skyrl_train.utils.utils import validate_hf_export_config


def _trainer_config(tmp_path, *, execute: bool = False):
    return OmegaConf.create(
        {
            "trainer": {
                "ckpt_path": str(tmp_path / "checkpoints"),
                "export_path": str(tmp_path / "exports"),
                "hf_export_execution": execute,
                "placement": {"policy_num_nodes": 2, "policy_num_gpus_per_node": 4},
                "policy": {"model": {"path": "org/model"}},
                "hf_hub_repo_id": "org/exported-model",
            }
        }
    )


def test_normal_hf_save_records_rerunnable_request_without_live_export(tmp_path):
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.cfg = _trainer_config(tmp_path)
    trainer.all_timings = {}
    trainer.global_step = 10
    checkpoint = tmp_path / "checkpoints" / "global_step_10"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.pt").write_bytes(b"complete")
    trainer.handle_hf_model_save()

    request = read_hf_export_request(str(checkpoint))
    assert request is not None
    assert request.status is HFExportStatus.PENDING
    assert request.step == 10
    assert request.checkpoint_base_path == str(tmp_path / "checkpoints")
    assert request.checkpoint_path == str(checkpoint)
    assert request.export_path == str(tmp_path / "exports")
    assert request.hf_hub_repo_id == "org/exported-model"
    assert not (tmp_path / "exports").exists()


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

    assert pending_hf_export_steps(str(tmp_path)) == {5}

    write_hf_export_request(request.with_status(HFExportStatus.COMPLETE, last_exit_code=0))
    assert pending_hf_export_steps(str(tmp_path)) == set()


@pytest.mark.parametrize("request_contents", ["{", "{}"])
def test_corrupt_export_request_protects_its_checkpoint(tmp_path, request_contents):
    checkpoint = tmp_path / "global_step_5"
    checkpoint.mkdir()
    (checkpoint / "hf_export_request.json").write_text(request_contents)

    assert pending_hf_export_steps(str(tmp_path)) == {5}


def test_hf_export_interval_must_be_checkpoint_aligned():
    cfg = OmegaConf.create({"trainer": {"ckpt_interval": 3, "hf_save_interval": 5, "hf_export_execution": False}})

    with pytest.raises(ValueError, match="multiple of trainer.ckpt_interval"):
        validate_hf_export_config(cfg)


def test_default_hf_export_interval_tracks_checkpoint_override():
    cfg = get_default_config()
    cfg.trainer.ckpt_interval = 10

    assert cfg.trainer.hf_save_interval == 10
    validate_hf_export_config(cfg)


@pytest.mark.parametrize(
    "trainer_config",
    [
        {
            "callbacks": [
                {"type": "checkpoint", "save_steps": 5},
                {"type": "hf_model_save", "save_steps": 5},
                {"type": "hf_hub_upload", "upload_steps": 5, "repo_id": "org/model"},
            ]
        },
    ],
)
def test_normal_training_rejects_in_band_hub_upload(trainer_config):
    cfg = OmegaConf.create({"trainer": {**trainer_config, "hf_export_execution": False}})

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
    command = build_command(
        ExportJobSpec(
            request=request,
            rl_config="config.yaml",
            cluster="cw-rno2a",
            priority="batch",
            train_data='["dataset"]',
            job_name="export-step-10",
            timeout=7200,
            no_wait=False,
        )
    )

    options = {
        command[index]: command[index + 1] for index in range(len(command) - 1) if command[index].startswith("--")
    }
    overrides = {
        value.split("=", 1)[0]: value.split("=", 1)[1]
        for index, value in enumerate(command)
        if index > 0 and command[index - 1] == "--skyrl_override"
    }

    assert overrides["++trainer.hf_export_execution"] == "true"
    assert overrides["++trainer.callbacks"] == "[]"
    assert overrides["++trainer.ckpt_interval"] == "-1"
    assert options["--timeout"] == "7200"
    assert "--no-wait" not in command
    assert overrides["++trainer.hf_hub_repo_id"] == "org/exported-model"


def test_request_rejects_operator_override_instead_of_ignoring_it():
    parser = argument_parser()
    args = parser.parse_args(
        ["--request", "/checkpoint/global_step_10", "--rl_config", "config.yaml", "--num-nodes", "8"]
    )

    with pytest.raises(SystemExit):
        request_spec(args, parser)


@pytest.mark.parametrize(
    ("exit_code", "expected_status"),
    [(0, HFExportStatus.COMPLETE), (17, HFExportStatus.PENDING)],
)
def test_export_request_records_terminal_result(tmp_path, monkeypatch, exit_code, expected_status):
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
    assert updated.timeout_seconds == 7200
    assert updated.last_exit_code == exit_code
