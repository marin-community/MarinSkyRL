import os
from pathlib import Path

import huggingface_hub.constants
import pytest
from rigging.filesystem.distributed_lock import LeaseLostError

from cloud.iris import hf_model_cache
from cloud.iris.hf_model_cache import (
    CachedHuggingFaceModel,
    ensure_hugging_face_model_cache,
    stage_cached_hugging_face_model,
)


def test_hub_download_temporarily_enables_network_access(tmp_path: Path, monkeypatch) -> None:
    def snapshot_download(*_args, **_kwargs):
        assert not huggingface_hub.constants.is_offline_mode()
        assert "HF_HUB_OFFLINE" not in os.environ
        assert "TRANSFORMERS_OFFLINE" not in os.environ
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", True)
    monkeypatch.setattr(hf_model_cache, "snapshot_download", snapshot_download)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    resolved = hf_model_cache.download_hugging_face_snapshot(
        "laion/draft",
        revision="4bdb47c08e5b5190bea3c7a93c3e14470230e469",
        destination=tmp_path,
    )

    assert resolved == tmp_path
    assert huggingface_hub.constants.is_offline_mode()
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"


def test_lost_cache_lease_does_not_publish_completion_marker(tmp_path: Path, monkeypatch) -> None:
    class LostLease:
        def try_acquire(self) -> bool:
            return True

        def refresh(self) -> None:
            raise LeaseLostError("lost")

        def release(self) -> None:
            pass

    cache = tmp_path / "region-cache"

    def download_snapshot(_model_id: str, *, revision: str, destination: Path) -> Path:
        (destination / "config.json").write_text("{}")
        (destination / "model.safetensors").write_bytes(revision.encode())
        return destination

    monkeypatch.setattr(hf_model_cache, "marin_temp_bucket", lambda *_args, **_kwargs: str(cache))
    monkeypatch.setattr(hf_model_cache, "create_lock", lambda _path: LostLease())
    monkeypatch.setattr(hf_model_cache, "download_hugging_face_snapshot", download_snapshot)

    with pytest.raises(LeaseLostError):
        ensure_hugging_face_model_cache(
            "laion/draft",
            "4bdb47c08e5b5190bea3c7a93c3e14470230e469",
            ttl_days=14,
            source_prefix="s3://region/experiments/run",
        )

    assert not (cache / ".marinskyrl-cache.json").exists()


def test_repeated_draft_staging_uses_the_completed_region_cache(tmp_path: Path, monkeypatch) -> None:
    cache = tmp_path / "region-cache"
    local_model = tmp_path / "node" / "draft"
    downloads = []

    def download_snapshot(model_id: str, *, revision: str, destination: Path) -> Path:
        downloads.append((model_id, revision))
        (destination / "config.json").write_text("{}")
        (destination / "model.safetensors").write_bytes(b"weights")
        return destination

    monkeypatch.setattr(hf_model_cache, "marin_temp_bucket", lambda *_args, **_kwargs: str(cache))
    monkeypatch.setattr(hf_model_cache, "download_hugging_face_snapshot", download_snapshot)
    model = CachedHuggingFaceModel(
        model_id="laion/draft",
        revision="4bdb47c08e5b5190bea3c7a93c3e14470230e469",
        local_path=str(local_model),
    )

    stage_cached_hugging_face_model(model, ttl_days=14, source_prefix="s3://region/experiments/run")
    stage_cached_hugging_face_model(model, ttl_days=14, source_prefix="s3://region/experiments/next-run")

    assert downloads == [(model.model_id, model.revision)]
    assert (local_model / "config.json").read_text() == "{}"
    assert (local_model / "model.safetensors").read_bytes() == b"weights"
