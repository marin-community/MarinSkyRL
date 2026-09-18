from pathlib import Path
from types import SimpleNamespace

from cloud.iris import hf_model_cache
from cloud.iris.hf_model_cache import CachedHuggingFaceModel, stage_cached_hugging_face_model


def test_hub_download_uses_an_online_child_process(tmp_path: Path, monkeypatch) -> None:
    observed_environment = {}

    def run(*_args, env, **_kwargs):
        observed_environment.update(env)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setattr(hf_model_cache.subprocess, "run", run)

    hf_model_cache._download_snapshot(
        "laion/draft",
        "4bdb47c08e5b5190bea3c7a93c3e14470230e469",
        tmp_path,
    )

    assert "HF_HUB_OFFLINE" not in observed_environment
    assert "TRANSFORMERS_OFFLINE" not in observed_environment


def test_repeated_draft_staging_uses_the_completed_region_cache(tmp_path: Path, monkeypatch) -> None:
    cache = tmp_path / "region-cache"
    local_model = tmp_path / "node" / "draft"
    downloads = []

    def download_snapshot(model_id: str, revision: str, local_dir: Path) -> None:
        downloads.append((model_id, revision))
        (local_dir / "config.json").write_text("{}")
        (local_dir / "model.safetensors").write_bytes(b"weights")

    monkeypatch.setattr(hf_model_cache, "marin_temp_bucket", lambda *_args, **_kwargs: str(cache))
    monkeypatch.setattr(hf_model_cache, "_download_snapshot", download_snapshot)
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
