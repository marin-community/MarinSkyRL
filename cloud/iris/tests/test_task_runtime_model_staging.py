from argparse import Namespace
from types import SimpleNamespace

import pytest

from cloud.iris import task_runtime
from cloud.iris.task_runtime import (
    apply_policy_model_to_command,
    parse_args,
    policy_chat_template_model,
    prepare_policy_model,
)


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
    args, train_argv = parse_args(
        [
            "--prestage-teacher-models-json",
            '[{"path":"Qwen/teacher","revision":"abc123"}]',
            "--",
            "python",
            "train.py",
        ]
    )

    assert [(model.path, model.revision) for model in args.prestage_teacher_models] == [("Qwen/teacher", "abc123")]
    assert train_argv == ["python", "train.py"]


def test_s3_policy_stages_metadata_and_rewrites_driver_without_materializing_weights(monkeypatch) -> None:
    identity = "sha256:" + "a" * 64
    manifest = SimpleNamespace(identity=identity)
    staged = []
    monkeypatch.setattr(task_runtime, "load_model_manifest", lambda _uri: manifest)
    monkeypatch.setattr(
        task_runtime, "stage_model_metadata", lambda uri, value, path: staged.append((uri, value, path))
    )
    args = Namespace(
        model_source_uri="s3://models/policy",
        model_source_identity=identity,
        prestage_model="",
        stream_model="",
        model_revision="",
        model_cache_ttl_days=None,
        model_cache_source_prefix="",
    )
    command = ["python", "-m", "cloud.iris.training_driver", "--model_path", "s3://models/policy"]

    policy_model = prepare_policy_model(args)

    assert policy_model is not None
    apply_policy_model_to_command(command, policy_model)
    assert staged == [("s3://models/policy", manifest, policy_model.metadata_path)]
    assert command[command.index("--model_path") + 1] == policy_model.metadata_path
    assert command[command.index("--model-source-uri") + 1] == "s3://models/policy"
    assert command[command.index("--model-source-identity") + 1] == identity
    assert "++generator.engine_init_kwargs.served_model_name=policy" in command
