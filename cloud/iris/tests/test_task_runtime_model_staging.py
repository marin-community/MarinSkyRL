from types import SimpleNamespace

import pytest

from cloud.iris import task_runtime
from cloud.iris.task_runtime import policy_chat_template_model, stage_model, teacher_model_specs_from_json


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


def test_stage_model_forwards_immutable_teacher_revision(monkeypatch) -> None:
    commands = []

    def run(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout="PRESTAGE_LOCAL_DIR=/cache/model\n", stderr="")

    monkeypatch.setattr(task_runtime.subprocess, "run", run)

    stage_model("Qwen/teacher", revision="abc123")

    assert commands[0][-1] == "abc123"


def test_teacher_model_staging_boundary_returns_typed_identities() -> None:
    models = teacher_model_specs_from_json('[{"path":"Qwen/teacher","revision":"abc123"}]')

    assert [(model.path, model.revision) for model in models] == [("Qwen/teacher", "abc123")]
