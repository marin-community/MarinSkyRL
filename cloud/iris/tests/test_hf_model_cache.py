import json
import os
from pathlib import Path

import huggingface_hub.constants
import numpy as np
import pytest
from safetensors.numpy import save_file

from cloud.iris import hf_model_cache
from cloud.iris.hf_model_cache import ensure_hugging_face_model_cache, ensure_model_manifest, stage_model_metadata
from marinskyrl.model_manifest import ModelManifest, snapshot_model_manifest


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


def test_corrupt_completed_cache_is_repaired_under_the_distributed_lock(tmp_path: Path, monkeypatch) -> None:
    cache = tmp_path / "region-cache"
    cache.mkdir()
    (cache / ".marinskyrl-model-manifest.json").write_text("not json")
    (cache / "stale.safetensors").write_bytes(b"stale")
    monkeypatch.setattr(hf_model_cache, "marin_temp_bucket", lambda *_args, **_kwargs: str(cache))
    revision = "4bdb47c08e5b5190bea3c7a93c3e14470230e469"

    def download_snapshot(_model_id: str, *, revision: str, destination: Path) -> Path:
        (destination / "config.json").write_text("{}")
        (destination / "tokenizer.json").write_text("{}")
        save_file({"weight": np.arange(4, dtype=np.float32)}, destination / "model.safetensors")
        return destination

    monkeypatch.setattr(hf_model_cache, "download_hugging_face_snapshot", download_snapshot)

    cache_uri, manifest = ensure_hugging_face_model_cache(
        "laion/draft", revision, ttl_days=14, source_prefix="s3://region/run"
    )

    assert cache_uri == str(cache)
    assert manifest == hf_model_cache.load_model_manifest(str(cache))
    assert not (cache / "stale.safetensors").exists()


def test_legacy_model_export_gets_a_manifest(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "tokenizer.json").write_text("{}")
    save_file({"weight": np.arange(4, dtype=np.float32)}, tmp_path / "model.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 16}, "weight_map": {"weight": "model.safetensors"}})
    )

    manifest = ensure_model_manifest(str(tmp_path))

    assert manifest == hf_model_cache.load_model_manifest(str(tmp_path))
    assert {entry.path for entry in manifest.files} == {
        "config.json",
        "model.safetensors",
        "model.safetensors.index.json",
        "tokenizer.json",
    }


def test_legacy_draft_export_gets_a_shared_tokenizer_manifest(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}")
    save_file({"weight": np.arange(4, dtype=np.float32)}, tmp_path / "model.safetensors")

    manifest = ensure_model_manifest(str(tmp_path), tokenizer_mode="policy")

    assert manifest.tokenizer_mode == "policy"
    assert manifest == hf_model_cache.load_model_manifest(str(tmp_path))
    assert json.loads((tmp_path / "model.safetensors.index.json").read_text())["weight_map"] == {
        "weight": "model.safetensors"
    }


def test_draft_manifest_can_share_the_policy_tokenizer(tmp_path: Path, monkeypatch) -> None:
    cache = tmp_path / "draft-cache"

    def download_snapshot(_model_id: str, *, revision: str, destination: Path) -> Path:
        assert revision == "4bdb47c08e5b5190bea3c7a93c3e14470230e469"
        (destination / "config.json").write_text("{}")
        save_file({"weight": np.arange(4, dtype=np.float32)}, destination / "model.safetensors")
        return destination

    monkeypatch.setattr(hf_model_cache, "marin_temp_bucket", lambda *_args, **_kwargs: str(cache))
    monkeypatch.setattr(hf_model_cache, "download_hugging_face_snapshot", download_snapshot)

    cache_uri, manifest = ensure_hugging_face_model_cache(
        "laion/draft",
        "4bdb47c08e5b5190bea3c7a93c3e14470230e469",
        ttl_days=14,
        source_prefix="s3://region/run",
        tokenizer_mode="policy",
    )

    assert cache_uri == str(cache)
    assert manifest.tokenizer_mode == "policy"
    assert not any(entry.path.startswith("tokenizer") for entry in manifest.files)
    assert (
        ModelManifest.from_mapping(json.loads((cache / ".marinskyrl-model-manifest.json").read_text()), str(cache))
        == manifest
    )


def test_model_manifest_rejects_metadata_paths_outside_the_model_root(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "tokenizer.json").write_text("{}")
    save_file({"weight": np.arange(4, dtype=np.float32)}, tmp_path / "model.safetensors")
    manifest = snapshot_model_manifest(tmp_path, model_id="laion/draft", revision="pinned")
    value = manifest.model_dump(mode="json")
    value["files"][0]["path"] = "../config.json"

    with pytest.raises(ValueError, match="relative and contained"):
        ModelManifest.from_mapping(value, "memory://models/draft")


def test_policy_manifest_without_tokenizer_mode_remains_compatible(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "tokenizer.json").write_text("{}")
    save_file({"weight": np.arange(4, dtype=np.float32)}, tmp_path / "model.safetensors")
    manifest = snapshot_model_manifest(tmp_path, model_id="snowball/policy", revision="pinned")
    value = manifest.model_dump(mode="json")
    value.pop("tokenizer_mode")

    assert ModelManifest.from_mapping(value, "memory://models/policy") == manifest
