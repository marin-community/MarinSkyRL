import pytest

from cloud.iris.hf_model_cache import CachedHuggingFaceModel
from cloud.iris.task_runtime import parse_args, policy_chat_template_model


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


def test_model_staging_cli_parses_launcher_payloads() -> None:
    revision = "4bdb47c08e5b5190bea3c7a93c3e14470230e469"
    args, train_argv = parse_args(
        [
            "--prestage-teacher-models-json",
            '[{"path":"Qwen/teacher","revision":"abc123"}]',
            "--prestage-draft-model",
            "laion/draft",
            revision,
            "/tmp/draft",
            "--",
            "python",
            "train.py",
        ]
    )

    assert [(model.path, model.revision) for model in args.prestage_teacher_models] == [("Qwen/teacher", "abc123")]
    assert args.prestage_draft_model == CachedHuggingFaceModel(
        model_id="laion/draft",
        revision=revision,
        local_path="/tmp/draft",
    )
    assert train_argv == ["python", "train.py"]
