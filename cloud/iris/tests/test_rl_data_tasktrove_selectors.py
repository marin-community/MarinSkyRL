import json

from omegaconf import OmegaConf

from cloud.iris import rl_data
from cloud.iris.training_driver import LocalRLConfig, LocalRLRunner
from marinskyrl.resource_locator import HFDatasetSelector, is_hugging_face_repo_id, parse_hf_dataset_selector


def test_dataset_selector_is_distinct_from_plain_repo_id():
    value = "fixture-org/fixture-trove@rev-1::a/nested"

    assert not is_hugging_face_repo_id(value)
    assert parse_hf_dataset_selector(value) == HFDatasetSelector(
        repo_id="fixture-org/fixture-trove",
        revision="rev-1",
        subdir="a/nested",
    )


def test_runner_resolves_and_records_training_and_validation_selectors(monkeypatch, tmp_path):
    resolved = {
        "fixture-org/fixture-trove::train": rl_data.ResolvedRLData(
            paths=("/tasks/train",),
            sources=("fixture-org/fixture-trove@immutable-sha::train",),
        ),
        "fixture-org/fixture-trove::validation": rl_data.ResolvedRLData(
            paths=("/tasks/validation",),
            sources=("fixture-org/fixture-trove@immutable-sha::validation",),
        ),
    }

    def resolve(values):
        return resolved[values[0]]

    monkeypatch.setattr("cloud.iris.training_driver.resolve_rl_train_data", resolve)
    launch_config = OmegaConf.create(
        {
            "run": {"mode": "train"},
            "inputs": {},
            "skyrl": {
                "data": {"train_data": [], "val_data": []},
            },
        }
    )
    runner = LocalRLRunner(
        LocalRLConfig(
            job_name="job",
            model_path="org/model",
            train_data=["fixture-org/fixture-trove::train"],
            val_data=["fixture-org/fixture-trove::validation"],
            experiments_dir=str(tmp_path / "experiments"),
            resolved_config_uri=(tmp_path / "resolved.json").as_uri(),
            launch_config=launch_config,
        )
    )
    monkeypatch.setattr(runner, "_setup_environment", lambda: None)
    monkeypatch.setattr(runner, "_run_skyrl", lambda _config: 0)

    assert runner.run() == 0

    recorded = json.loads((tmp_path / "resolved.json").read_text())
    assert recorded["train_data_sources"] == ["fixture-org/fixture-trove@immutable-sha::train"]
    assert recorded["val_data_sources"] == ["fixture-org/fixture-trove@immutable-sha::validation"]
    assert recorded["config"]["skyrl"]["data"]["train_data"] == ["/tasks/train"]
    assert recorded["config"]["skyrl"]["data"]["val_data"] == ["/tasks/validation"]
