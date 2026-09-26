from pathlib import Path

import pytest
import yaml

from cloud.iris.model_paths import model_source_cli_args
from cloud.iris.tests.test_launch_config import _raw_config
from cloud.iris.training_driver import LocalRLConfig, main


def test_model_source_cli_args_omit_absent_source() -> None:
    assert model_source_cli_args(None, None) == []


def test_training_driver_rejects_partial_model_source() -> None:
    with pytest.raises(ValueError, match="must be provided together"):
        LocalRLConfig(
            job_name="invalid-model-source",
            model_path="/tmp/materialized-model",
            model_source_uri="s3://models/policy",
        )


def test_training_driver_rejects_source_for_hugging_face_repo_id() -> None:
    with pytest.raises(ValueError, match="requires a local metadata path"):
        LocalRLConfig(
            job_name="ambiguous-model-source",
            model_path="org/model",
            model_source_uri="s3://models/policy",
            model_source_identity="policy@abc123",
        )


def test_training_driver_omits_source_for_hugging_face_input(tmp_path: Path, monkeypatch) -> None:
    config = _raw_config()
    config["skyrl"] = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs/qwen_megatron_smoke.yaml").read_text()
    )
    config["inputs"]["model"] = {
        "uri": "Qwen/Qwen3-0.6B",
        "identity": "main",
        "local_path": "Qwen/Qwen3-0.6B",
        "tokenizer_uri": "Qwen/Qwen3-0.6B",
        "tokenizer_revision": "main",
    }
    config["inputs"]["data_kind"] = "parquet"
    path = tmp_path / "qwen-launch.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    seen = []

    class RecordingRunner:
        def __init__(self, run_config):
            seen.append(run_config)

        def setup(self):
            pass

        def run(self):
            return 0

    monkeypatch.setattr("cloud.iris.training_driver.LocalRLRunner", RecordingRunner)
    monkeypatch.setattr("sys.argv", ["training_driver", "--config", str(path)])

    with pytest.raises(SystemExit) as result:
        main()

    assert result.value.code == 0
    assert seen[0].model_path == "Qwen/Qwen3-0.6B"
    assert seen[0].model_source_uri is None
