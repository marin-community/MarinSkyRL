"""Pivot/SFT configuration preserves the tested physical model placement."""

from pathlib import Path

import pytest
import yaml

from cloud.iris.rl_config_translation import parse_rl_config, compose_skyrl_config
from skyrl_train.utils.utils import validate_batch_sizes
from infra.rl_data.pivot_recipe import recipe, STUDENTS

CONFIGS = Path(__file__).parents[1] / "configs"


@pytest.mark.parametrize("mode,loss,samples", [("pivotrl", "regular", 16), ("sft", "sft", 1), ("sft_random", "sft", 1)])
def test_pivot_mode_compiles_same_data_and_geometry(tmp_path, mode, loss, samples):
    baseline = yaml.safe_load((CONFIGS / "snowball_ultra_rlvr1_split64.yaml").read_text())
    raw = yaml.safe_load((CONFIGS / "snowball_pivotrl_split64.yaml").read_text())
    raw["pivot"]["mode"] = mode
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(raw))
    parsed = parse_rl_config(str(path))

    class HPC:
        gpus_per_node = 8

    cfg = compose_skyrl_config(
        parsed, {"job_name": "test", "experiments_dir": str(tmp_path), "num_nodes": 8}, HPC()
    ).config
    validate_batch_sizes(cfg)
    assert cfg.trainer.algorithm.policy_loss_type == loss
    assert cfg.generator.reference_actions == (mode in {"sft", "sft_random"})
    assert cfg.generator.n_samples_per_prompt == samples
    assert cfg.trainer.train_batch_size * samples == (1024 if mode == "pivotrl" else 64)
    assert list(cfg.data.train_data) == raw["data"]["train_data"]
    assert list(cfg.data.val_data) == raw["data"]["val_data"]
    assert cfg.trainer.algorithm.use_kl_loss == (mode == "pivotrl")
    baseline_cfg = compose_skyrl_config(
        parse_rl_config(str(CONFIGS / "snowball_ultra_rlvr1_split64.yaml")),
        {"job_name": "baseline", "experiments_dir": str(tmp_path), "num_nodes": 8},
        HPC(),
    ).config
    for role in ("policy", "ref"):
        assert cfg.trainer[role].megatron_config == baseline_cfg.trainer[role].megatron_config
    for key, value in baseline["trainer"]["placement"].items():
        assert cfg.trainer.placement[key] == value


@pytest.mark.parametrize("student", STUDENTS)
def test_profile_uses_pinned_student_over_every_candidate(tmp_path, student):
    raw = recipe(student, "profile", "candidates.parquet", ["validation.parquet"], 0.001)
    path = tmp_path / "profile.yaml"
    path.write_text(yaml.safe_dump(raw))
    parsed = parse_rl_config(str(path))

    class HPC:
        gpus_per_node = 8

    cfg = compose_skyrl_config(
        parsed, {"job_name": "profile", "experiments_dir": str(tmp_path), "num_nodes": 8}, HPC()
    ).config
    assert parsed.entrypoint == "skyrl_train.entrypoints.main_generate"
    assert cfg.generator.pivot_profiling
    assert cfg.generator.eval_n_samples_per_prompt == 8
    assert cfg.generator.eval_sampling_params.max_generate_length == 65535
    assert cfg.data.prompt_length_policy == "keep"
    assert list(cfg.data.val_data) == ["candidates.parquet"]
    assert cfg.trainer.policy.model.path == STUDENTS[student][0]
    assert cfg.generator.trajectory_retention.model_source_identity == STUDENTS[student][1]
    assert cfg.generator.trajectory_retention.required


def test_remaining_context_rejects_multiple_turns(tmp_path):
    raw = yaml.safe_load((CONFIGS / "snowball_pivotrl_split64.yaml").read_text())
    raw.pop("pivot")
    raw["context_budget"]["max_turns"] = 2
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="single-turn"):
        parse_rl_config(str(path))
