import json
import os
from pathlib import Path

import huggingface_hub.constants
import numpy as np
import pytest
from safetensors.numpy import save_file

from cloud.iris import hf_model_cache
from cloud.iris.hf_model_cache import ensure_hugging_face_model_cache, stage_model_metadata


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


def test_repeated_draft_staging_uses_the_completed_region_cache(tmp_path: Path, monkeypatch) -> None:
    cache = tmp_path / "region-cache"
    local_model = tmp_path / "node" / "draft"
    downloads = []

    def download_snapshot(model_id: str, *, revision: str, destination: Path) -> Path:
        downloads.append((model_id, revision))
        (destination / "config.json").write_text("{}")
        (destination / "tokenizer.json").write_text("{}")
        save_file({"weight": np.arange(4, dtype=np.float32)}, destination / "model.safetensors")
        return destination

    monkeypatch.setattr(hf_model_cache, "marin_temp_bucket", lambda *_args, **_kwargs: str(cache))
    monkeypatch.setattr(hf_model_cache, "download_hugging_face_snapshot", download_snapshot)
    model_id = "laion/draft"
    revision = "4bdb47c08e5b5190bea3c7a93c3e14470230e469"

    cache_uri, manifest = ensure_hugging_face_model_cache(
        model_id, revision, ttl_days=14, source_prefix="s3://region/experiments/run"
    )
    stage_model_metadata(cache_uri, manifest, str(local_model))
    repeated_uri, repeated_manifest = ensure_hugging_face_model_cache(
        model_id, revision, ttl_days=14, source_prefix="s3://region/experiments/next-run"
    )
    (local_model / "stale.bin").write_bytes(b"stale weights")
    stage_model_metadata(cache_uri, manifest, str(local_model))

    assert downloads == [(model_id, revision)]
    assert repeated_uri == cache_uri
    assert repeated_manifest == manifest
    assert manifest.revision == revision
    assert manifest.identity.startswith("sha256:")
    assert {entry.path for entry in manifest.files} == {
        "config.json",
        "model.safetensors",
        "model.safetensors.index.json",
        "tokenizer.json",
    }
    assert (local_model / "config.json").read_text() == "{}"
    assert (local_model / "model.safetensors.index.json").is_file()
    assert not (local_model / "model.safetensors").exists()
    assert not (local_model / "stale.bin").exists()


def test_corrupt_completed_cache_fails_instead_of_remirroring(tmp_path: Path, monkeypatch) -> None:
    cache = tmp_path / "region-cache"
    cache.mkdir()
    (cache / ".marinskyrl-model-manifest.json").write_text("not json")
    monkeypatch.setattr(hf_model_cache, "marin_temp_bucket", lambda *_args, **_kwargs: str(cache))
    revision = "4bdb47c08e5b5190bea3c7a93c3e14470230e469"

    with pytest.raises(json.JSONDecodeError):
        ensure_hugging_face_model_cache("laion/draft", revision, ttl_days=14, source_prefix="s3://region/run")
