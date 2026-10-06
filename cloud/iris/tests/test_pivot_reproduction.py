"""Recorded conditions compose as fresh, isolated runs on the current launcher."""

import json
from pathlib import Path

from omegaconf import OmegaConf
import pytest
import yaml

from cloud.iris.launch_config import load_launch_config
from scripts.pivotrl.prepare_run import EXPERIMENTS, prepare_run


EXPERIMENTS_BY_KEY = {
    item["key"]: item for item in json.loads((EXPERIMENTS / "manifest.json").read_text())["experiments"]
}


@pytest.mark.parametrize("key", sorted(EXPERIMENTS_BY_KEY))
def test_fresh_run_preserves_frozen_inputs_and_isolates_outputs(tmp_path: Path, key: str):
    experiment = EXPERIMENTS_BY_KEY[key]
    recorded = yaml.safe_load((EXPERIMENTS / experiment["config"]).read_text())
    output = tmp_path / "fresh.yaml"
    prepare_run(key, "repro-" + key, "s3://marin-us-east-02a/reproduction/" + key, tmp_path / "cluster.yaml", output)
    fresh = load_launch_config(output)
    assert fresh.run.id != recorded["run"]["id"]
    assert fresh.artifacts.checkpoint_root != recorded["artifacts"]["checkpoint_root"]
    assert OmegaConf.to_container(fresh.inputs.train_data, resolve=True) == recorded["inputs"]["train_data"]
    assert OmegaConf.to_container(fresh.inputs.validation_data, resolve=True) == recorded["inputs"]["validation_data"]
    for field, value in recorded["inputs"]["model"].items():
        assert fresh.inputs.model[field] == value
    assert fresh.skyrl.trainer.resume_path is None
    assert fresh.skyrl.trainer.resume_mode == "none"
    assert fresh.skyrl.trainer.policy.model.revision == recorded["inputs"]["model"]["identity"]
    if key.endswith("rl-sync"):
        assert fresh.skyrl.trainer.rollout_buffer.max_staleness_steps == 0
        assert fresh.skyrl.generator.sampling_params.max_generate_length == 2048
        assert fresh.skyrl.generator.n_samples_per_prompt == 8
        assert fresh.skyrl.trainer.algorithm.kl_estimator_type == "forward"
    else:
        assert fresh.skyrl.generator.reference_actions
        assert fresh.skyrl.trainer.algorithm.policy_loss_type == "sft"
        assert fresh.skyrl.trainer.loss_token_budget == 1000000
