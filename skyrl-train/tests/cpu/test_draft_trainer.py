"""Behavior tests for the independent cloud-backed DraftTrainer."""

import json
from pathlib import Path
import shutil

import pytest

import skyrl_train.draft_trainer as draft_trainer_module
from marinskyrl.speculative_decoding import SpeculatorModelConfig, SpeculatorTrainingConfig
from skyrl_train.draft_trainer import (
    DraftCheckpoint,
    DraftTrainer,
    DraftUpdateRequest,
    latest_draft_checkpoint_uri,
    read_latest_draft_checkpoint,
)
from skyrl_train.inference_engines.vllm.online_eagle_trainer import OnlineEagleUpdateResult


_DRAFT_REVISION = "4bdb47c08e5b5190bea3c7a93c3e14470230e469"


class _CloudFixture:
    def __init__(self, root: Path):
        self.root = root
        self.events: list[tuple[str, str]] = []

    def path(self, uri: str) -> Path:
        assert uri.startswith("s3://bucket/")
        return self.root / uri.removeprefix("s3://bucket/")

    def exists(self, uri: str) -> bool:
        return self.path(uri).exists()

    def read_bytes(self, uri: str) -> bytes:
        return self.path(uri).read_bytes()

    def write_bytes_atomic(self, uri: str, value: bytes) -> None:
        path = self.path(uri)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
        self.events.append(("write", uri))

    def upload_directory(self, source: str, uri: str) -> None:
        destination = self.path(uri)
        shutil.copytree(source, destination)
        self.events.append(("upload", uri))

    def download_directory(self, uri: str, destination: str) -> None:
        shutil.copytree(self.path(uri), destination)
        self.events.append(("download", uri))

    def remove(self, uri: str) -> None:
        path = self.path(uri)
        shutil.rmtree(path) if path.is_dir() else path.unlink()
        self.events.append(("remove", uri))


@pytest.fixture
def cloud(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _CloudFixture:
    fixture = _CloudFixture(tmp_path / "cloud")
    for name in ("exists", "read_bytes", "write_bytes_atomic", "upload_directory", "download_directory", "remove"):
        monkeypatch.setattr(draft_trainer_module.io, name, getattr(fixture, name))
    return fixture


def _initial_model() -> SpeculatorModelConfig:
    return SpeculatorModelConfig(
        source_uri="hf://laion/snowball-64k-eagle3-draft-r2egym",
        source_identity=_DRAFT_REVISION,
    )


def _request(step: int = 4) -> DraftUpdateRequest:
    return DraftUpdateRequest(
        step=step,
        capture_uri=f"s3://bucket/run/drafts/captures/step-{step}",
        target_revision=f"policy-step-{step - 1}",
        num_speculative_tokens=3,
        seed=7,
        training=SpeculatorTrainingConfig(
            min_train_sequences=1,
            min_holdout_sequences=1,
        ),
    )


def _publish_capture(cloud: _CloudFixture, request: DraftUpdateRequest) -> None:
    capture = cloud.path(request.capture_uri)
    capture.mkdir(parents=True)
    (capture / "capture.txt").write_text("capture")


class _Runtime:
    instances: list["_Runtime"] = []

    def __init__(self, job, capture_dir):
        self.initial_job = job
        self.capture_dir = capture_dir
        self.restored_from = None
        self.instances.append(self)

    def restore(self, checkpoint_dir):
        self.restored_from = checkpoint_dir

    def update(self, job):
        candidate = Path(job.output_dir)
        candidate.mkdir()
        (candidate / "config.json").write_text("{}")
        (candidate / "model.safetensors").write_bytes(b"draft")
        (candidate / "trainer_state.pt").write_bytes(b"state")
        return OnlineEagleUpdateResult(
            accepted=True,
            step=job.step,
            draft_revision=f"draft-step-{job.step}",
        )


def _patch_training(monkeypatch: pytest.MonkeyPatch) -> None:
    _Runtime.instances.clear()
    monkeypatch.setattr(draft_trainer_module, "OnlineEagleTrainerRuntime", _Runtime)

    def merge(_capture_root, output_dir, **_kwargs):
        output_dir.mkdir()
        return {"active": True}

    monkeypatch.setattr(draft_trainer_module, "merge_online_eagle_captures", merge)


def test_draft_checkpoint_requires_cloud_uri() -> None:
    with pytest.raises(ValueError, match="cloud-backed"):
        DraftCheckpoint.from_mapping(
            {
                "step": 4,
                "revision": "draft-step-4",
                "uri": "/tmp/draft",
                "source_identity": _DRAFT_REVISION,
            }
        )


def test_draft_trainer_publishes_checkpoint_before_latest_pointer(
    cloud: _CloudFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_training(monkeypatch)
    request = _request()
    _publish_capture(cloud, request)
    checkpoint_root = "s3://bucket/run/drafts/checkpoints"
    trainer = DraftTrainer(initial_model=_initial_model(), checkpoint_root=checkpoint_root)

    result = trainer.update(request)

    candidate_uri = "s3://bucket/run/drafts/checkpoints/draft-step-4"
    assert result["accepted"] is True
    assert result["candidate_uri"] == candidate_uri
    assert cloud.events[-3:] == [
        ("upload", candidate_uri),
        ("write", latest_draft_checkpoint_uri(checkpoint_root)),
        ("remove", request.capture_uri),
    ]
    latest = read_latest_draft_checkpoint(checkpoint_root)
    assert latest == DraftCheckpoint(
        step=4,
        revision="draft-step-4",
        uri=candidate_uri,
        source_identity=_DRAFT_REVISION,
    )
    assert _Runtime.instances[0].initial_job.draft_model_source == _initial_model().source_uri


def test_draft_trainer_restores_latest_checkpoint_once(
    cloud: _CloudFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_training(monkeypatch)
    checkpoint_root = "s3://bucket/run/drafts/checkpoints"
    checkpoint_uri = f"{checkpoint_root}/draft-step-4"
    checkpoint_dir = cloud.path(checkpoint_uri)
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "trainer_state.pt").write_bytes(b"state")
    cloud.write_bytes_atomic(
        latest_draft_checkpoint_uri(checkpoint_root),
        json.dumps(
            {
                "step": 4,
                "revision": "draft-step-4",
                "uri": checkpoint_uri,
                "source_identity": _DRAFT_REVISION,
            }
        ).encode(),
    )
    request = _request(step=8)
    _publish_capture(cloud, request)

    trainer = DraftTrainer(initial_model=_initial_model(), checkpoint_root=checkpoint_root)
    result = trainer.update(request)

    assert result["accepted"] is True
    assert _Runtime.instances[0].initial_job.parent_draft_revision == "draft-step-4"
    assert _Runtime.instances[0].restored_from is not None


def test_latest_checkpoint_from_another_initial_draft_is_ignored(
    cloud: _CloudFixture,
) -> None:
    checkpoint_root = "s3://bucket/run/drafts/checkpoints"
    cloud.write_bytes_atomic(
        latest_draft_checkpoint_uri(checkpoint_root),
        json.dumps(
            {
                "step": 4,
                "revision": "draft-step-4",
                "uri": f"{checkpoint_root}/draft-step-4",
                "source_identity": "different-draft",
            }
        ).encode(),
    )

    assert (
        read_latest_draft_checkpoint(
            checkpoint_root,
            source_identity=_DRAFT_REVISION,
        )
        is None
    )


def test_draft_trainer_failure_is_nonfatal_and_consumes_capture(
    cloud: _CloudFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request()
    _publish_capture(cloud, request)
    monkeypatch.setattr(
        draft_trainer_module,
        "merge_online_eagle_captures",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("bad capture")),
    )
    trainer = DraftTrainer(initial_model=_initial_model(), checkpoint_root="s3://bucket/run/drafts/checkpoints")

    result = trainer.update(request)

    assert result == {
        "accepted": False,
        "step": 4,
        "error": "RuntimeError: bad capture",
    }
    assert not cloud.exists(request.capture_uri)
