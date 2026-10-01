"""Contract checks for the native full Open-MOPD training entrypoint."""

import hashlib
import json
import shutil
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from hydra import compose, initialize_config_dir
from skyrl_train.config.trajectory_runner_capabilities import (
    TrajectoryRunnerMode,
    validate_trajectory_runner_capabilities,
)
from skyrl_train.inference_engines.utils import get_vllm_sampling_params
from skyrl_train.utils.utils import validate_cfg

SCRIPT = Path(__file__).parents[3] / "ci" / "opd" / "open_mopd_native_full.py"
CONFIG_ROOT = Path(__file__).parents[3] / "skyrl_train" / "config"
SPEC = spec_from_file_location("open_mopd_native_full", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = module_from_spec(SPEC)
sys.modules["open_mopd_native_full"] = MODULE
SPEC.loader.exec_module(MODULE)


def test_full_schedule_preserves_released_objective_and_every_checkpoint():
    arguments = MODULE.hydra_arguments(
        Path("/data/schedule.parquet"),
        Path("/data/aime24.parquet"),
        "s3://bucket/users/operator/checkpoints",
        "s3://bucket/users/operator/exports",
    )

    # Keep the released objective while avoiding the GPU-only FlashAttention import in CPU CI.
    with initialize_config_dir(config_dir=str(CONFIG_ROOT), version_base=None):
        config = compose(config_name="ppo_base_config", overrides=[*arguments, "trainer.flash_attn=false"])
    validate_cfg(config)
    sampling = get_vllm_sampling_params(config.generator.sampling_params)
    assert sampling["top_p"] == 0.99
    assert not config.generator.engine_init_kwargs.get("validate_rollout_logprob_sampling", False)
    assert config.data.val_data == ["/data/aime24.parquet"]
    assert config.trainer.ckpt_path == "s3://bucket/users/operator/checkpoints"
    assert config.trainer.export_path == "s3://bucket/users/operator/exports"
    assert config.trainer.ckpt_interval == config.trainer.hf_save_interval
    assert config.trainer.max_ckpts_to_keep == -1
    assert config.trainer.algorithm.distillation.objective == "student_topk_policy_surrogate"
    assert dict(config.trainer.algorithm.distillation.domain_gradient_balance.target_shares) == {
        "math": 1 / 3,
        "code": 1 / 3,
        "if": 1 / 3,
    }
    assert set(config.teachers) == {"math", "code", "if"}
    assert config.trainer.resume_mode is None
    validate_trajectory_runner_capabilities(config, TrajectoryRunnerMode.SKYRL_GYM)

    config.generator.use_conversation_multi_turn = True
    config.generator.chat_template.name_or_path = "qwen3_without_thinking"
    with pytest.raises(ValueError, match="exact sampled completion token IDs"):
        validate_trajectory_runner_capabilities(config, TrajectoryRunnerMode.SKYRL_GYM)


def test_schedule_rejects_changed_bytes_before_training(monkeypatch, tmp_path):
    destination = tmp_path / "schedule.parquet"

    def changed_download(_source, target):
        Path(target).write_bytes(b"not the pinned schedule")

    monkeypatch.setattr(MODULE.io, "download_file", changed_download)
    with pytest.raises(ValueError, match="digest mismatch"):
        MODULE.stage_schedule("s3://bucket/schedule.parquet", destination)


def test_resume_requires_the_same_run_and_a_durable_checkpoint(monkeypatch, tmp_path):
    filesystem = fsspec.filesystem("memory")
    prefix = "s3://bucket/users/open-mopd-resume-test"
    dataset = f"{prefix}/schedule.parquet"
    validation = f"{prefix}/aime24.parquet"
    checkpoints = f"{prefix}/checkpoints"
    exports = f"{prefix}/exports"
    manifest = f"{prefix}/manifest.json"
    monkeypatch.setattr(MODULE, "fs_and_path", lambda uri: (filesystem, uri.removeprefix("s3://")))
    schedule_file = tmp_path / "schedule.parquet"
    validation_file = tmp_path / "aime24.parquet"
    pq.write_table(pa.table({"prompt": ["first", "second"]}), schedule_file, row_group_size=1)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "prompt": [{"role": "user", "content": "What is 1 + 0?"}],
                    "env_class": "aime",
                    "reward_model": {"ground_truth": "1"},
                }
            ]
            * 30
        ),
        validation_file,
    )
    monkeypatch.setattr(MODULE, "SCHEDULE_SHA256", hashlib.sha256(schedule_file.read_bytes()).hexdigest())
    monkeypatch.setattr(MODULE, "SCHEDULE_ROWS", 2)
    monkeypatch.setattr(MODULE, "SCHEDULE_STEPS", 2)
    validation_sha256 = hashlib.sha256(validation_file.read_bytes()).hexdigest()
    sources = {dataset: schedule_file, validation: validation_file}
    monkeypatch.setattr(MODULE.io, "download_file", lambda source, target: shutil.copyfile(sources[source], target))
    commands = []

    def train(command, *, check):
        assert check is False
        commands.append(command)
        return SimpleNamespace(returncode=1 if len(commands) == 1 else 0)

    monkeypatch.setattr(MODULE.subprocess, "run", train)
    assert MODULE.run(dataset, validation, validation_sha256, checkpoints, exports, manifest, "pinned-commit") == 1
    with pytest.raises(FileNotFoundError, match="durable checkpoint marker"):
        MODULE.run(dataset, validation, validation_sha256, checkpoints, exports, manifest, "pinned-commit", resume=True)
    filesystem.pipe_file(f"{checkpoints.removeprefix('s3://')}/latest_ckpt_global_step.txt", b"2")
    with pytest.raises(ValueError, match="identity differs"):
        MODULE.run(
            dataset, validation, validation_sha256, checkpoints, exports, manifest, "different-commit", resume=True
        )
    assert (
        MODULE.run(dataset, validation, validation_sha256, checkpoints, exports, manifest, "pinned-commit", resume=True)
        == 0
    )
    with initialize_config_dir(config_dir=str(CONFIG_ROOT), version_base=None):
        initial = compose(config_name="ppo_base_config", overrides=commands[0][3:])
        resumed = compose(config_name="ppo_base_config", overrides=commands[1][3:])
    assert initial.trainer.resume_mode is None
    assert resumed.trainer.resume_mode == "latest"
    assert resumed.trainer.eval_interval == 2
    assert json.loads(filesystem.cat_file(manifest.removeprefix("s3://")))["status"] == "complete"
