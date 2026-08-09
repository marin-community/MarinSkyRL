import pytest

from cloud.iris.model_paths import model_source_cli_args
from cloud.iris.training_driver import LocalRLConfig


def test_model_source_cli_args_omit_absent_source() -> None:
    assert model_source_cli_args(None, None) == []


def test_training_driver_rejects_partial_model_source() -> None:
    with pytest.raises(ValueError, match="must be provided together"):
        LocalRLConfig(
            rl_config_path="config.yaml",
            job_name="invalid-model-source",
            model_path="/tmp/materialized-model",
            model_source_uri="s3://models/policy",
        )


def test_training_driver_rejects_source_for_hugging_face_repo_id() -> None:
    with pytest.raises(ValueError, match="requires a task-local model_path"):
        LocalRLConfig(
            rl_config_path="config.yaml",
            job_name="ambiguous-model-source",
            model_path="org/model",
            model_source_uri="s3://models/policy",
            model_source_identity="policy@abc123",
        )
