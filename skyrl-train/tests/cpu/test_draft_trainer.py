"""Behavior tests for dedicated draft-training transport and lifecycle primitives."""

from __future__ import annotations

from pathlib import Path
import shutil

import pytest

import skyrl_train.draft_trainer as draft_trainer_module
from skyrl_train.draft_trainer import (
    DIRECTORY_BUNDLE_FORMAT,
    DraftTrainer,
    bundle_directory_for_ray,
    materialize_ray_directory_bundle,
    validate_materialized_bundle,
)
from skyrl_train.inference_engines.vllm.online_eagle_trainer import OnlineEagleUpdateResult


def _identity(value):
    return value


def test_ray_directory_bundle_round_trips_in_bounded_chunks(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "manifest.json").write_text('{"complete":true}')
    (source / "nested").mkdir()
    (source / "nested" / "weights.bin").write_bytes(b"0123456789")

    bundle = bundle_directory_for_ray(source, put=_identity, chunk_bytes=4)
    destination = tmp_path / "destination"
    materialize_ray_directory_bundle(bundle, destination, get=_identity)

    assert bundle["format"] == DIRECTORY_BUNDLE_FORMAT
    assert [chunk["bytes"] for chunk in bundle["files"][1]["chunks"]] == [4, 4, 2]
    assert (destination / "manifest.json").read_text() == '{"complete":true}'
    assert (destination / "nested" / "weights.bin").read_bytes() == b"0123456789"
    validate_materialized_bundle(bundle, destination)


def test_ray_directory_bundle_rejects_path_traversal_atomically(tmp_path: Path) -> None:
    bundle = {
        "format": DIRECTORY_BUNDLE_FORMAT,
        "format_version": 1,
        "total_bytes": 1,
        "files": [
            {
                "path": "../escape",
                "bytes": 1,
                "sha256": "unused",
                "chunks": [{"bytes": 1, "sha256": "unused", "object_ref": b"x"}],
            }
        ],
    }
    destination = tmp_path / "destination"

    with pytest.raises(ValueError, match="normalized relative path"):
        materialize_ray_directory_bundle(bundle, destination, get=_identity)

    assert not destination.exists()
    assert not (tmp_path / "escape").exists()
    assert not list(tmp_path.glob(".destination.tmp-*"))


def test_ray_directory_bundle_rejects_corrupt_chunk_atomically(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "weights.bin").write_bytes(b"correct")
    bundle = bundle_directory_for_ray(source, put=_identity, chunk_bytes=4)
    bundle["files"][0]["chunks"][0]["object_ref"] = b"wrong"
    destination = tmp_path / "destination"

    with pytest.raises(ValueError, match="chunk digest mismatch"):
        materialize_ray_directory_bundle(bundle, destination, get=_identity)

    assert not destination.exists()
    assert not list(tmp_path.glob(".destination.tmp-*"))


def test_ray_directory_bundle_refuses_to_overwrite_existing_destination(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "weights.bin").write_bytes(b"weights")
    destination = tmp_path / "destination"
    destination.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        materialize_ray_directory_bundle(
            bundle_directory_for_ray(source, put=_identity),
            destination,
            get=_identity,
        )


def test_ray_directory_bundle_excludes_private_trainer_state(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.safetensors").write_bytes(b"served")
    (source / "trainer_state.pt").write_bytes(b"private")

    bundle = bundle_directory_for_ray(
        source,
        put=_identity,
        excluded_relative_paths={"trainer_state.pt"},
    )

    assert [item["path"] for item in bundle["files"]] == ["model.safetensors"]


def _training_job(tmp_path: Path, *, step: int, parent_draft_revision: str) -> dict:
    candidate_dir = tmp_path / "process" / "candidates" / f"step-{step}"
    return {
        "step": step,
        "capture_dir": str(tmp_path / "process" / f"step-{step}" / "merged"),
        "draft_model_dir": str(tmp_path / "ignored-driver-path"),
        "initial_draft_source_identity": "hf://draft@revision",
        "parent_draft_revision": parent_draft_revision,
        "target_revision": f"policy-step-{step - 1}",
        "target_weights_sha256": f"target-{step - 1}",
        "output_dir": str(candidate_dir),
        "result_path": f"{candidate_dir}.result.json",
        "failure_artifact_path": str(tmp_path / "failures" / f"step-{step}"),
        "num_speculative_tokens": 3,
        "seed": 42,
        "training": {},
    }


def test_draft_trainer_owns_served_lineage_across_commit_and_rollback(tmp_path: Path, monkeypatch) -> None:
    initial = tmp_path / "initial"
    initial.mkdir()
    observed_draft_dirs = []

    def materialize(_bundle, destination):
        path = Path(destination)
        path.mkdir(parents=True)
        (path / "capture.bin").write_bytes(b"capture")
        return path

    def train(job):
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

    monkeypatch.setattr(draft_trainer_module, "materialize_ray_directory_bundle", materialize)
    monkeypatch.setattr(draft_trainer_module, "validate_materialized_bundle", lambda _bundle, _path: None)
    monkeypatch.setattr(draft_trainer_module, "run_training_job", train)
    monkeypatch.setattr(
        draft_trainer_module,
        "bundle_directory_for_ray",
        lambda path, **_kwargs: {"path": str(path)},
    )
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
    first = trainer.update(_training_job(tmp_path, step=4, parent_draft_revision="draft-initial"), {})
    assert first["candidate_bundle"] == {"path": str(tmp_path / "process" / "candidates" / "step-4")}
    trainer.commit("draft-step-4")
    trainer.publish(str(tmp_path / "published"), "draft-step-4", "policy-step-7")

    second = trainer.update(_training_job(tmp_path, step=8, parent_draft_revision="draft-step-4"), {})
    trainer.rollback("draft-step-8")

    assert observed_draft_dirs == [str(initial), str(tmp_path / "process" / "candidates" / "step-4")]
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
        trainer.update(_training_job(tmp_path, step=4, parent_draft_revision="draft-other"), {})


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
