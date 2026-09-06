import os
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from skyrl_train.checkpoint_exporter import (
    CheckpointExporter,
    CheckpointExportPlan,
    CheckpointExportResult,
    RayPolicyExportWorkers,
)
from skyrl_train.entrypoints import checkpoint_export
from skyrl_train.entrypoints.checkpoint_export import _completion_metadata, _write_completion_receipt
from skyrl_train.hf_export_schema import HFUploadMode
from skyrl_train.hf_model_io import verify_hf_model_export
from skyrl_train.hf_publisher import HuggingFacePublisher

from marinskyrl.export_completion import verify_export_receipt


class FakePolicyExportWorkers:
    def __init__(self):
        self.model_path = None
        self.checkpoint_path = None
        self.export_path = None
        self.tokenizer = None
        self.closed = False

    def initialize(self, model_path: str) -> None:
        self.model_path = model_path

    def load_model_checkpoint(self, checkpoint_path: str) -> None:
        self.checkpoint_path = checkpoint_path

    def save_hf_model(self, export_path: str, tokenizer: object) -> None:
        self.export_path = export_path
        self.tokenizer = tokenizer
        Path(export_path).mkdir(parents=True)
        Path(export_path, "model.safetensors").write_bytes(b"weights")

    def close(self) -> None:
        self.closed = True


class MissingExportWorkers(FakePolicyExportWorkers):
    def save_hf_model(self, export_path: str, tokenizer: object) -> None:
        self.export_path = export_path


class MetadataOnlyExportWorkers(FakePolicyExportWorkers):
    def save_hf_model(self, export_path: str, tokenizer: object) -> None:
        self.export_path = export_path
        Path(export_path).mkdir(parents=True)
        Path(export_path, "config.json").write_text("{}")
        Path(export_path, "model.safetensors.index.json").write_text(
            '{"weight_map": {"model.layers.0.weight": "model-00001-of-00001.safetensors"}}'
        )


class FakeRayActorGroup:
    def __init__(self):
        self.calls = []
        self.killed = False

    def async_run_ray_method(self, dispatch_type, method_name, *args, **kwargs):
        self.calls.append((dispatch_type, method_name, args, kwargs))
        return [f"{method_name}-ref"]

    def kill_actors(self):
        self.killed = True


class FakeHubApi:
    def __init__(self):
        self.repositories = []
        self.uploads = []

    def create_repo(self, **kwargs):
        self.repositories.append(kwargs)

    def upload_folder(self, **kwargs):
        self.uploads.append(kwargs)


class FakePublisher:
    def __init__(self):
        self.calls = []

    def publish(self, export_path: str, step: int) -> None:
        self.calls.append((export_path, step))


def _plan(tmp_path: Path, step: int = 12) -> CheckpointExportPlan:
    checkpoint_path = tmp_path / "checkpoints" / f"global_step_{step}"
    checkpoint_path.mkdir(parents=True)
    (checkpoint_path / "policy").mkdir()
    torch.save({"global_step": step}, checkpoint_path / "trainer_state.pt")
    return CheckpointExportPlan(
        step=step,
        checkpoint_path=str(checkpoint_path),
        export_root=str(tmp_path / "exports"),
        model_path="org/model",
    )


def test_checkpoint_exporter_converts_only_the_policy_model(tmp_path):
    plan = _plan(tmp_path)
    workers = FakePolicyExportWorkers()
    tokenizer = object()

    result = CheckpointExporter(plan, workers, tokenizer).run()

    assert result.step == 12
    assert result.export_path == str(tmp_path / "exports" / "global_step_12" / "policy")
    assert workers.model_path == "org/model"
    assert workers.checkpoint_path == str(tmp_path / "checkpoints" / "global_step_12" / "policy")
    assert workers.export_path == result.export_path
    assert workers.tokenizer is tokenizer
    assert workers.closed


def test_checkpoint_entrypoint_records_receipt_for_complete_export(tmp_path: Path) -> None:
    export_path = tmp_path / "policy"
    export_path.mkdir()
    (export_path / "config.json").write_text("{}")
    (export_path / "tokenizer.json").write_text("{}")
    (export_path / "model.safetensors").write_bytes(b"weights")
    receipt_path = tmp_path / "receipts" / "attempt-2.json"
    cfg = OmegaConf.create(
        {
            "checkpoint_export": {
                "completion_receipt_uri": str(receipt_path),
                "request_fingerprint": "request-2",
                "attempt_id": "attempt-2",
            }
        }
    )
    metadata = _completion_metadata(cfg)
    assert metadata is not None

    _write_completion_receipt(metadata, CheckpointExportResult(step=9, export_path=str(export_path)))

    assert verify_export_receipt(str(receipt_path), "request-2", str(export_path), 9).attempt_id == "attempt-2"


def test_checkpoint_entrypoint_rejects_partial_receipt_metadata() -> None:
    cfg = OmegaConf.create(
        {"checkpoint_export": {"completion_receipt_uri": "s3://bucket/receipt.json", "attempt_id": "attempt-1"}}
    )

    with pytest.raises(ValueError, match="provided together"):
        _completion_metadata(cfg)


def test_checkpoint_entrypoint_writes_receipt_after_ray_shutdown(monkeypatch) -> None:
    events = []
    cfg = OmegaConf.create(
        {
            "checkpoint_export": {
                "completion_receipt_uri": "receipt.json",
                "request_fingerprint": "request-2",
                "attempt_id": "attempt-2",
            }
        }
    )
    result = CheckpointExportResult(step=9, export_path="policy")

    @dataclass(frozen=True)
    class RemoteExport:
        def remote(self, remote_cfg):
            assert remote_cfg is cfg
            return "result-ref"

    monkeypatch.setattr(checkpoint_export, "initialize_ray", lambda _cfg: events.append("initialize"))
    monkeypatch.setattr(checkpoint_export, "run_checkpoint_export", RemoteExport())
    monkeypatch.setattr(checkpoint_export.ray, "get", lambda ref: result if ref == "result-ref" else None)
    monkeypatch.setattr(checkpoint_export.ray, "shutdown", lambda: events.append("shutdown"))
    monkeypatch.setattr(checkpoint_export.signal, "signal", lambda *_args: None)
    monkeypatch.setattr(
        checkpoint_export,
        "_write_completion_receipt",
        lambda metadata, completed: events.append(("receipt", metadata, completed)),
    )

    checkpoint_export.main.__wrapped__(cfg)

    assert events == [
        "initialize",
        "shutdown",
        ("receipt", ("receipt.json", "request-2", "attempt-2"), result),
    ]


def test_checkpoint_exporter_rejects_a_mismatched_checkpoint_marker(tmp_path):
    plan = _plan(tmp_path)
    torch.save({"global_step": 11}, Path(plan.checkpoint_path) / "trainer_state.pt")
    workers = FakePolicyExportWorkers()

    with pytest.raises(ValueError, match="checkpoint step mismatch"):
        CheckpointExporter(plan, workers, object()).run()

    assert workers.closed


@pytest.mark.parametrize(
    ("worker_type", "error"),
    [
        (MissingExportWorkers, "no safetensors weights"),
        (MetadataOnlyExportWorkers, "missing 1 referenced safetensors shard"),
    ],
)
def test_checkpoint_exporter_rejects_incomplete_conversion_result(tmp_path, worker_type, error):
    workers = worker_type()
    publisher = FakePublisher()

    with pytest.raises(RuntimeError, match=error):
        CheckpointExporter(_plan(tmp_path), workers, object(), publisher).run()

    assert workers.closed
    assert publisher.calls == []


def test_unsharded_cloud_export_verification_does_not_head_optional_index(monkeypatch):
    export_path = "s3://bucket/exports/global_step_1/policy"
    monkeypatch.setattr(
        "skyrl_train.hf_model_io.io.list_dir",
        lambda path: [f"{path}/config.json", f"{path}/model.safetensors"],
    )

    def reject_head(path):
        raise AssertionError(f"verification issued an object HEAD request: {path}")

    monkeypatch.setattr("skyrl_train.hf_model_io.io.exists", reject_head)

    verify_hf_model_export(export_path)


def test_ray_policy_export_workers_loads_model_state_without_training_state():
    group = FakeRayActorGroup()
    resolved_refs = []
    workers = RayPolicyExportWorkers(group, resolve=lambda refs: resolved_refs.append(refs))

    workers.initialize("org/model")
    workers.load_model_checkpoint("/checkpoints/global_step_12/policy")
    workers.save_hf_model("/exports/global_step_12/policy", tokenizer="tokenizer")
    workers.close()

    assert group.calls == [
        ("pass_through", "init_model_for_export", ("org/model",), {}),
        (
            "pass_through",
            "load_checkpoint",
            (),
            {
                "ckpt_dir": "/checkpoints/global_step_12/policy",
                "load_training_state": False,
            },
        ),
        (
            "pass_through",
            "save_hf_model",
            ("/exports/global_step_12/policy", "tokenizer"),
            {},
        ),
    ]
    assert resolved_refs == [
        ["init_model_for_export-ref"],
        ["load_checkpoint-ref"],
        ["save_hf_model-ref"],
    ]
    assert group.killed


def test_hf_publisher_publishes_root_and_requested_archive(tmp_path, monkeypatch):
    export_path = tmp_path / "exports" / "global_step_12" / "policy"
    export_path.mkdir(parents=True)
    (export_path / "model.safetensors").write_bytes(b"weights")
    api = FakeHubApi()
    publisher = HuggingFacePublisher(
        repo_id="org/model",
        private=True,
        revision="main",
        upload_mode=HFUploadMode.ALL,
        api=api,
    )
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    publisher.publish(str(export_path), step=12)

    assert api.repositories == [{"repo_id": "org/model", "repo_type": "model", "private": True, "exist_ok": True}]
    assert [upload["path_in_repo"] for upload in api.uploads] == ["", "checkpoints/step_12"]
    assert all(upload["folder_path"] == str(export_path) for upload in api.uploads)
    assert os.environ["HF_HUB_OFFLINE"] == "1"
