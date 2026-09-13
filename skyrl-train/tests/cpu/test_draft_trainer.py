"""Behavior tests for dedicated draft-training transport and lifecycle primitives."""

from __future__ import annotations

from pathlib import Path
import shutil

import pytest
from safetensors.torch import load_file
import torch

import skyrl_train.draft_trainer as draft_trainer_module
from skyrl_train.draft_trainer import (
    DraftTrainer,
)
from skyrl_train.inference_engines.vllm.online_eagle_trainer import OnlineEagleUpdateResult


def test_draft_trainer_materializes_direct_capture_transfer(tmp_path: Path, monkeypatch) -> None:
    initial = tmp_path / "initial"
    initial.mkdir()
    trainer = DraftTrainer(
        initial_draft_dir=str(initial),
        initial_draft_revision="draft-initial",
        process_id="process",
    )
    trainer._draft_transfer_group = object()
    received = {
        "window-000000.safetensors::input_ids": torch.tensor([1, 2, 3]),
        "target.safetensors::model.embed_tokens.weight": torch.ones(2, 2),
        "target.safetensors::lm_head.weight": torch.ones(2, 2),
    }
    monkeypatch.setattr(draft_trainer_module, "transfer_tensor_operations", lambda *_args, **_kwargs: received)
    plan = {
        "format": "marinskyrl-online-eagle-capture-transfer",
        "format_version": 1,
        "total_bytes": 56,
        "operations": [{"operation": "metadata-only"}],
        "files": [
            {
                "destination_path": "window-000000.safetensors",
                "tensors": [{"name": "input_ids"}],
            },
            {
                "destination_path": "target.safetensors",
                "tensors": [{"name": "lm_head.weight"}, {"name": "model.embed_tokens.weight"}],
            },
        ],
        "target_config_json": '{"model_type":"test"}',
        "capture_manifest": {
            "windows": [{"path": "window-000000.safetensors", "sha256": "source"}],
            "captured_rows": 3,
            "dropped_windows": 0,
            "oversized_windows": 0,
            "unselected_windows": 0,
            "target": {
                "weights_path": "target.safetensors",
                "weights_sha256": "source",
                "config_path": "target-config.json",
                "config_sha256": "source",
            },
        },
    }

    result = trainer.receive_capture(plan, str(tmp_path / "process" / "step-4" / "merged"))

    assert result["captured_rows"] == 3
    assert result["transfer_bytes"] == 56
    assert torch.equal(
        load_file(Path(result["capture_dir"]) / "window-000000.safetensors")["input_ids"],
        received["window-000000.safetensors::input_ids"],
    )


def _training_job(tmp_path: Path, *, step: int, parent_draft_revision: str) -> dict:
    candidate_dir = tmp_path / "process" / "candidates" / f"step-{step}"
    capture_dir = tmp_path / "process" / f"step-{step}" / "merged"
    capture_dir.mkdir(parents=True, exist_ok=True)
    return {
        "step": step,
        "capture_dir": str(capture_dir),
        "draft_model_dir": str(tmp_path / "ignored-driver-path"),
        "initial_draft_source_identity": "hf://draft@revision",
        "parent_draft_revision": parent_draft_revision,
        "target_revision": f"policy-step-{step - 1}",
        "target_weights_sha256": f"target-{step - 1}",
        "output_dir": str(candidate_dir),
        "failure_artifact_path": str(tmp_path / "failures" / f"step-{step}"),
        "num_speculative_tokens": 3,
        "seed": 42,
        "training": {},
    }


def test_draft_trainer_owns_served_lineage_across_commit_and_rollback(tmp_path: Path, monkeypatch) -> None:
    initial = tmp_path / "initial"
    initial.mkdir()
    runtime_initializations = []
    observed_draft_dirs = []

    class Runtime:
        def __init__(self, job, capture_dir):
            runtime_initializations.append((job.draft_model_dir, capture_dir))

        def update(self, job):
            observed_draft_dirs.append(job.draft_model_dir)
            candidate = Path(job.output_dir)
            candidate.mkdir(parents=True)
            (candidate / "model.safetensors").write_bytes(b"weights")
            return OnlineEagleUpdateResult(
                active=True,
                accepted=True,
                step=job.step,
                parent_draft_revision=job.parent_draft_revision,
                trained_against_target_revision=job.target_revision,
                trained_against_target_weights_sha256=job.target_weights_sha256,
                candidate_dir=job.output_dir,
                draft_revision=f"draft-step-{job.step}",
                weights_sha256=f"draft-digest-{job.step}",
            )

        def commit(self, _draft_revision):
            return None

        def rollback(self, _draft_revision):
            return None

    monkeypatch.setattr(draft_trainer_module, "OnlineEagleTrainerRuntime", Runtime)
    monkeypatch.setattr(
        draft_trainer_module,
        "validate_online_eagle_serving_candidate",
        lambda _path: {"weights_path": "model.safetensors"},
    )
    candidate_tensors = {"draft.weight": torch.arange(4, dtype=torch.bfloat16).reshape(2, 2)}
    monkeypatch.setattr(draft_trainer_module, "load_file", lambda _path: candidate_tensors)
    monkeypatch.setattr(
        draft_trainer_module,
        "remove_online_eagle_scratch",
        lambda path: shutil.rmtree(path, ignore_errors=True),
    )
    published = []

    def publish(source, destination, *, draft_revision, served_target_revision):
        published.append((source, destination, draft_revision, served_target_revision))
        return {"complete": True, "path": f"{destination}/manifest.json"}

    monkeypatch.setattr(draft_trainer_module, "publish_speculator_checkpoint", publish)

    trainer = DraftTrainer(
        initial_draft_dir=str(initial),
        initial_draft_revision="draft-initial",
        process_id="process",
    )
    first = trainer.update(_training_job(tmp_path, step=4, parent_draft_revision="draft-initial"))
    assert first["transfer_manifest"]["revision"] == "draft-step-4"
    assert first["transfer_manifest"]["total_bytes"] == 8
    assert "object_ref" not in repr(first["transfer_manifest"])
    trainer.commit("draft-step-4")
    trainer.publish(str(tmp_path / "published"), "draft-step-4", "policy-step-7")

    second = trainer.update(_training_job(tmp_path, step=8, parent_draft_revision="draft-step-4"))
    trainer.rollback("draft-step-8")

    assert observed_draft_dirs == [str(initial), str(tmp_path / "process" / "candidates" / "step-4")]
    assert runtime_initializations == [(str(initial), tmp_path / "process" / "step-4" / "merged")]
    assert published == [
        (
            str(tmp_path / "process" / "candidates" / "step-4"),
            str(tmp_path / "published"),
            "draft-step-4",
            "policy-step-7",
        )
    ]
    assert not Path(second["result"]["candidate_dir"]).exists()
    status = trainer.status()
    assert status["served_draft_dir"] == str(tmp_path / "process" / "candidates" / "step-4")
    assert status["served_draft_revision"] == "draft-step-4"
    assert status["pending_candidate_dir"] is None
    assert status["pending_draft_revision"] is None


def test_draft_trainer_rejects_parent_outside_its_served_lineage(tmp_path: Path) -> None:
    initial = tmp_path / "initial"
    initial.mkdir()
    trainer = DraftTrainer(
        initial_draft_dir=str(initial),
        initial_draft_revision="draft-initial",
        process_id="process",
    )

    with pytest.raises(RuntimeError, match="served lineage"):
        trainer.update(_training_job(tmp_path, step=4, parent_draft_revision="draft-other"))


def test_draft_trainer_reports_training_failure_and_preserves_its_artifact(tmp_path: Path, monkeypatch) -> None:
    initial = tmp_path / "initial"
    initial.mkdir()

    failure_dir = tmp_path / "preserved-failure"

    class FailingRuntime:
        def __init__(self, _job, _capture_dir):
            pass

        def update(self, _job):
            raise RuntimeError("training failed")

    monkeypatch.setattr(draft_trainer_module, "OnlineEagleTrainerRuntime", FailingRuntime)
    monkeypatch.setattr(
        draft_trainer_module,
        "preserve_online_eagle_failure",
        lambda _job, _error: str(failure_dir),
    )
    monkeypatch.setattr(
        draft_trainer_module,
        "publish_online_eagle_failure_bundle",
        lambda _source, destination: {"path": f"{destination}/manifest.json"},
    )
    monkeypatch.setattr(draft_trainer_module, "remove_online_eagle_scratch", lambda _path: None)
    trainer = DraftTrainer(
        initial_draft_dir=str(initial),
        initial_draft_revision="draft-initial",
        process_id="process",
    )

    update = trainer.update(_training_job(tmp_path, step=4, parent_draft_revision="draft-initial"))

    assert update["transfer_manifest"] is None
    assert update["result"] == {
        "active": True,
        "accepted": False,
        "step": 4,
        "error": "RuntimeError: training failed",
        "failure_dir": str(failure_dir),
        "failure_artifact_path": f"{tmp_path}/failures/step-4/manifest.json",
    }


def test_draft_trainer_cleanup_releases_its_process_scratch(tmp_path: Path, monkeypatch) -> None:
    scratch = tmp_path / "scratch"
    process_root = scratch / "process"
    process_root.mkdir(parents=True)
    (process_root / "capture.bin").write_bytes(b"capture")
    initial = tmp_path / "initial"
    initial.mkdir()
    monkeypatch.setattr(draft_trainer_module, "ONLINE_EAGLE_SCRATCH_ROOT", scratch)
    monkeypatch.setattr(
        draft_trainer_module,
        "remove_online_eagle_scratch",
        lambda path: shutil.rmtree(path, ignore_errors=True),
    )
    trainer = DraftTrainer(
        initial_draft_dir=str(initial),
        initial_draft_revision="draft-initial",
        process_id="process",
    )

    result = trainer.cleanup()

    assert result == {"path": str(process_root)}
    assert not process_root.exists()
