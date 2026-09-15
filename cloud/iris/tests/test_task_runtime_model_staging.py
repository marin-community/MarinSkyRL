import pytest

from cloud.iris.task_runtime import _warm_sync_model_from_s3, policy_chat_template_model


def test_revision_rejects_unbound_flat_warm_mirror(capsys) -> None:
    revision = "6" * 40

    assert not _warm_sync_model_from_s3("org/model", "s3://models/org--model", revision)
    assert "no revision-bound content manifest" in capsys.readouterr().out


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
