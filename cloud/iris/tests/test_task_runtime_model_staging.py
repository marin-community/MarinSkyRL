from pathlib import Path

import pytest

from cloud.iris import task_runtime
from cloud.iris.task_runtime import (
    cached_hugging_face_models_from_json,
    policy_chat_template_model,
    stage_model,
    teacher_model_specs_from_json,
)


@pytest.fixture
def recorded_stage_model_downloads(monkeypatch):
    downloads = []

    def download(model_id, **kwargs):
        downloads.append((model_id, kwargs))
        return Path("/cache/model")

    monkeypatch.setattr(task_runtime, "download_hugging_face_snapshot", download)
    return downloads


@pytest.mark.parametrize(
    ("prestage_model", "model_local_path", "expected"),
    [
        ("", "/tmp/materialized-model", "/tmp/materialized-model"),
        ("org/model", "/tmp/materialized-model", "org/model"),
    ],
)
def test_policy_chat_template_selects_materialized_model(
    prestage_model: str, model_local_path: str, expected: str
) -> None:
    assert policy_chat_template_model(prestage_model, model_local_path) == expected


def test_policy_chat_template_requires_a_materialized_model() -> None:
    with pytest.raises(ValueError, match="requires --prestage-model or --model-local-path"):
        policy_chat_template_model("", "")


def test_stage_model_forwards_immutable_teacher_revision(recorded_stage_model_downloads) -> None:
    stage_model("Qwen/teacher", revision="abc123")

    assert recorded_stage_model_downloads[0][1]["revision"] == "abc123"


def test_stage_model_preserves_repository_chat_template(recorded_stage_model_downloads) -> None:
    task_runtime.stage_model("HuggingFaceTB/SmolLM3-3B")

    allow_patterns = recorded_stage_model_downloads[0][1]["allow_patterns"]
    assert "*.jinja" in allow_patterns


def test_teacher_model_staging_boundary_returns_typed_identities() -> None:
    models = teacher_model_specs_from_json('[{"path":"Qwen/teacher","revision":"abc123"}]')

    assert [(model.path, model.revision) for model in models] == [("Qwen/teacher", "abc123")]


def test_draft_model_staging_boundary_returns_local_materialization() -> None:
    revision = "4bdb47c08e5b5190bea3c7a93c3e14470230e469"
    models = cached_hugging_face_models_from_json(
        f'[{{"model_id":"laion/draft","revision":"{revision}","local_path":"/tmp/draft"}}]'
    )

    assert [(model.model_id, model.revision, model.local_path) for model in models] == [
        ("laion/draft", revision, "/tmp/draft")
    ]
