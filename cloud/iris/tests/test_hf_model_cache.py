import json
import os
from pathlib import Path
import re

import huggingface_hub.constants
import numpy as np
import pytest
from safetensors.numpy import save_file

from cloud.iris import hf_model_cache
from cloud.iris.hf_model_cache import ensure_hugging_face_model_cache, stage_model_metadata
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


def test_repeated_draft_staging_uses_the_completed_region_cache(tmp_path: Path, monkeypatch, capsys) -> None:
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
    monkeypatch.setenv("HF_TOKEN", "sentinel-hf-token")
    monkeypatch.setenv("FSSPEC_S3", '{"endpoint_url":"http://cwlota.com","key":"sentinel-s3-key"}')
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://other.example/sentinel-url")
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
    output = capsys.readouterr().out
    assert "role=publisher" in output and "role=hit" in output
    assert "hf_token_present=True" in output
    assert "s3_endpoint_env_class=coreweave_in_cluster" in output
    for phase in ("download", "manifest", "publication"):
        assert f"phase={phase} started" in output
        assert re.search(rf"phase={phase} seconds=\d+\.\d+", output)
    assert re.search(r"role=publisher completed total_seconds=\d+\.\d+", output)
    assert re.search(r"files=4 bytes=\d+", output)
    assert not any(secret in output for secret in ("sentinel-hf-token", "sentinel-s3-key", "sentinel-url"))


@pytest.mark.parametrize(
    "fsspec_value, aws_value, expected_class",
    [
        (None, None, "unset"),
        (None, "https://cwobject.com", "other"),
        ('{"endpoint_url":"http://cwlota.com"}', None, "coreweave_in_cluster"),
        (
            '{"config_kwargs":{"s3":{"addressing_style":"virtual"}}}',
            "http://cwlota.com",
            "coreweave_in_cluster",
        ),
        ("not json", None, "unknown"),
    ],
)
def test_cold_cache_failure_reports_phase_and_safe_endpoint_class(
    tmp_path: Path, monkeypatch, capsys, fsspec_value, aws_value, expected_class
) -> None:
    cache = tmp_path / "region-cache"
    monkeypatch.setattr(hf_model_cache, "marin_temp_bucket", lambda *_args, **_kwargs: str(cache))
    if expected_class == "unset":
        monkeypatch.delenv("HF_TOKEN", raising=False)
    else:
        monkeypatch.setenv("HF_TOKEN", "sentinel-hf-token")
    for name, value in (("FSSPEC_S3", fsspec_value), ("AWS_ENDPOINT_URL", aws_value)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)

    def fail_download(*_args, **_kwargs) -> Path:
        raise RuntimeError("download failed")

    monkeypatch.setattr(hf_model_cache, "download_hugging_face_snapshot", fail_download)
    with pytest.raises(RuntimeError, match="download failed"):
        ensure_hugging_face_model_cache(
            "laion/draft", "4bdb47c08e5b5190bea3c7a93c3e14470230e469", ttl_days=14, source_prefix="s3://region/run"
        )

    output = capsys.readouterr().out
    assert f"s3_endpoint_env_class={expected_class}" in output
    assert f"hf_token_present={expected_class != 'unset'}" in output
    assert "phase=download started" in output
    assert "phase=download seconds=" not in output
    assert "role=publisher completed" not in output
    assert "sentinel-hf-token" not in output


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
