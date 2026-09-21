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
from skyrl_train.utils.utils import validate_cfg

from cloud.iris.rl_config_translation import build_checkpoint_export_hydra_args, parse_checkpoint_export_config

SCRIPT = Path(__file__).parents[3] / "ci" / "opd" / "open_mopd_native_full.py"
CONFIG_ROOT = Path(__file__).parents[3] / "skyrl_train" / "config"
EXPORT_CONFIG = Path(__file__).parents[3] / "ci" / "opd" / "open_mopd_native_export.yaml"
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

    with initialize_config_dir(config_dir=str(CONFIG_ROOT), version_base=None):
        config = compose(config_name="ppo_base_config", overrides=list(arguments))
    validate_cfg(config)
    assert config.trainer.max_steps == 200
    assert config.trainer.train_batch_size == 1024
    assert config.trainer.policy_mini_batch_size == 256
    assert config.trainer.ckpt_interval == config.trainer.hf_save_interval == 2
    assert config.trainer.eval_interval == 2
    assert config.trainer.dump_eval_results is True
    assert config.data.val_data == ["/data/aime24.parquet"]
    assert config.trainer.max_ckpts_to_keep == -1
    assert config.trainer.ckpt_path == "s3://bucket/users/operator/checkpoints"
    assert config.trainer.export_path == "s3://bucket/users/operator/exports"
    assert config.data.shuffle is False
    assert config.trainer.algorithm.distillation.objective == "student_topk_policy_surrogate"
    assert config.trainer.algorithm.distillation.domain_gradient_balance.gap_scale_alpha == 1.0
    assert dict(config.trainer.algorithm.distillation.domain_gradient_balance.target_shares) == {
        "math": 1 / 3,
        "code": 1 / 3,
        "if": 1 / 3,
    }
    assert config.generator.sampling_params.logprobs == 16
    assert config.generator.sampling_params.max_generate_length == 16384
    assert config.generator.eval_sampling_params.max_generate_length == 16384
    assert config.environment.skyrl_gym.aime.strict_box_verify is True
    assert set(config.teachers) == {"math", "code", "if"}
    assert config.trainer.resume_mode is None


def test_schedule_rejects_changed_bytes_before_training(monkeypatch, tmp_path):
    destination = tmp_path / "schedule.parquet"

    def changed_download(_source, target):
        Path(target).write_bytes(b"not the pinned schedule")

    monkeypatch.setattr(MODULE.io, "download_file", changed_download)
    try:
        MODULE.stage_schedule("s3://bucket/schedule.parquet", destination)
    except ValueError as error:
        assert "digest mismatch" in str(error)
    else:
        raise AssertionError("Changed schedule bytes were accepted")


def test_export_uses_four_checkpoint_ranks_on_reserved_eight_gpu_node():
    parsed = parse_checkpoint_export_config(
        str(EXPORT_CONFIG), model_override="BytedTsinghua-SIA/Open-MOPD-SmolLM3-3B-MixSFT"
    )
    arguments = build_checkpoint_export_hydra_args(
        parsed,
        {"num_nodes": 1, "gpus_per_node": 8, "model_path": "BytedTsinghua-SIA/Open-MOPD-SmolLM3-3B-MixSFT"},
        SimpleNamespace(gpus_per_node=8),
    )
    with initialize_config_dir(config_dir=str(CONFIG_ROOT), version_base=None):
        config = compose(config_name="ppo_base_config", overrides=arguments)

    assert config.trainer.placement.policy_num_nodes == 1
    assert config.trainer.placement.policy_num_gpus_per_node == 4


@pytest.fixture
def native_run(monkeypatch, tmp_path):
    """Stage a two-step schedule and a 30-row AIME set on a memory filesystem with training stubbed out."""
    filesystem = fsspec.filesystem("memory")
    prefix = f"s3://bucket/users/{tmp_path.name}"
    dataset = f"{prefix}/schedule.parquet"
    validation = f"{prefix}/aime24.parquet"
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
    sources = {dataset: schedule_file, validation: validation_file}
    monkeypatch.setattr(MODULE.io, "download_file", lambda source, target: shutil.copyfile(sources[source], target))
    commands = []

    def train(command, *, check):
        assert check is False
        commands.append(command)
        return SimpleNamespace(returncode=1 if len(commands) == 1 else 0)

    monkeypatch.setattr(MODULE.subprocess, "run", train)
    return SimpleNamespace(
        filesystem=filesystem,
        commands=commands,
        checkpoints=f"{prefix}/checkpoints",
        manifest=f"{prefix}/manifest.json",
        arguments=dict(
            dataset_uri=dataset,
            validation_uri=validation,
            validation_sha256=hashlib.sha256(validation_file.read_bytes()).hexdigest(),
            checkpoint_uri=f"{prefix}/checkpoints",
            export_uri=f"{prefix}/exports",
            manifest_uri=f"{prefix}/manifest.json",
            source_commit="pinned-commit",
        ),
    )


def test_resume_requires_the_same_run_and_a_durable_checkpoint(native_run):
    run = native_run.arguments
    assert MODULE.run(**run) == 1
    with pytest.raises(FileNotFoundError, match="durable checkpoint marker"):
        MODULE.run(**run, resume=True)
    native_run.filesystem.pipe_file(f"{native_run.checkpoints.removeprefix('s3://')}/latest_ckpt_global_step.txt", b"2")
    with pytest.raises(ValueError, match="identity differs"):
        MODULE.run(**{**run, "source_commit": "different-commit"}, resume=True)
    assert MODULE.run(**run, resume=True) == 0
    with initialize_config_dir(config_dir=str(CONFIG_ROOT), version_base=None):
        initial = compose(config_name="ppo_base_config", overrides=native_run.commands[0][3:])
        resumed = compose(config_name="ppo_base_config", overrides=native_run.commands[1][3:])
    assert initial.trainer.resume_mode is None
    assert resumed.trainer.resume_mode == "latest"
    assert resumed.trainer.eval_interval == 2
    assert json.loads(native_run.filesystem.cat_file(native_run.manifest.removeprefix("s3://")))["status"] == "complete"


def _compose(arguments):
    with initialize_config_dir(config_dir=str(CONFIG_ROOT), version_base=None):
        return compose(config_name="ppo_base_config", overrides=list(arguments))


def test_fully_async_schedule_keeps_the_per_prompt_cadence_of_the_synchronous_run():
    common = (
        Path("/data/schedule.parquet"),
        Path("/data/aime24.parquet"),
        "s3://bucket/users/operator/checkpoints",
        "s3://bucket/users/operator/exports",
    )
    sync_arguments = MODULE.hydra_arguments(*common)
    async_arguments = MODULE.hydra_arguments(*common, schedule=MODULE.Schedule.FULLY_ASYNC)

    assert sync_arguments == MODULE.hydra_arguments(*common, schedule=MODULE.Schedule.SYNC)
    keys = [override.split("=", 1)[0] for override in async_arguments]
    assert len(keys) == len(set(keys))

    sync_config = _compose(sync_arguments)
    config = _compose(async_arguments)
    validate_cfg(config)
    updates_per_rollout_batch = sync_config.trainer.train_batch_size // sync_config.trainer.policy_mini_batch_size
    assert config.trainer.train_batch_size == config.trainer.policy_mini_batch_size == 256
    assert config.trainer.max_steps == sync_config.trainer.max_steps * updates_per_rollout_batch
    assert (
        config.trainer.ckpt_interval
        == config.trainer.hf_save_interval
        == config.trainer.eval_interval
        == sync_config.trainer.ckpt_interval * updates_per_rollout_batch
    )
    assert config.trainer.fully_async.max_staleness_steps == 1
    assert config.trainer.fully_async.num_parallel_generation_workers >= config.trainer.policy_mini_batch_size
    assert config.generator.batched is False
    assert config.generator.async_engine is True
    assert config.trainer.placement.colocate_all is False
    assert config.generator.sampling_params.top_p == sync_config.generator.sampling_params.top_p == 0.99
    assert config.trainer.algorithm.use_tis is False
    assert config.trainer.algorithm.distillation.objective == sync_config.trainer.algorithm.distillation.objective
    assert config.trainer.max_ckpts_to_keep == -1


def test_fully_async_schedule_needs_the_in_process_runner_for_student_selected_evidence():
    config = _compose(
        MODULE.hydra_arguments(
            Path("/data/schedule.parquet"),
            Path("/data/aime24.parquet"),
            "s3://bucket/users/operator/checkpoints",
            "s3://bucket/users/operator/exports",
            schedule=MODULE.Schedule.FULLY_ASYNC,
        )
    )
    validate_cfg(config)

    validate_trajectory_runner_capabilities(config, TrajectoryRunnerMode.SKYRL_GYM)
    with pytest.raises(ValueError, match="student-selected top-k teacher evidence"):
        validate_trajectory_runner_capabilities(config, TrajectoryRunnerMode.FULLY_ASYNC_SKYRL_GYM)
    assert MODULE.ENTRYPOINT_MODULES[MODULE.Schedule.FULLY_ASYNC] == "skyrl_train.entrypoints.fully_async_in_process"


def test_resume_rejects_a_different_schedule(native_run):
    run = native_run.arguments
    assert MODULE.run(**run, schedule=MODULE.Schedule.FULLY_ASYNC) == 1
    native_run.filesystem.pipe_file(f"{native_run.checkpoints.removeprefix('s3://')}/latest_ckpt_global_step.txt", b"8")
    with pytest.raises(ValueError, match="identity differs"):
        MODULE.run(**run, resume=True)
    assert MODULE.run(**run, resume=True, schedule=MODULE.Schedule.FULLY_ASYNC) == 0
    assert [command[2] for command in native_run.commands] == ["skyrl_train.entrypoints.fully_async_in_process"] * 2
    assert (
        json.loads(native_run.filesystem.cat_file(native_run.manifest.removeprefix("s3://")))["schedule"]
        == "fully_async"
    )
