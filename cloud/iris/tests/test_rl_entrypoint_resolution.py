import json
import logging
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from cloud.iris.iris_backend import create_parser, normalize
from cloud.iris.rl_config_translation import build_skyrl_hydra_args, parse_rl_config
from marinskyrl.distillation import TeacherSource
from cloud.iris.training_driver import LocalRLConfig, LocalRLRunner, parse_list_arg
from skyrl_train.entrypoints.main_base import config_dir


_REPO_ROOT = Path(__file__).resolve().parents[3]
FULLY_ASYNC_MODULE = "skyrl_train.entrypoints.fully_async"
GENERATE_MODULE = "skyrl_train.entrypoints.main_generate"
SYNC_MODULE = "skyrl_train.entrypoints.main_base"
SYNCHRONOUS_CONFIG = _REPO_ROOT / "cloud" / "iris" / "configs" / "delphi_math_rl.yaml"
TRANSLATION_LOGGER = "cloud.iris.rl_config_translation"


def _config_with(tmp_path: Path, entrypoint: str | None = "sync", **sections) -> Path:
    raw = yaml.safe_load(SYNCHRONOUS_CONFIG.read_text())
    if entrypoint is None:
        raw.pop("entrypoint")
    else:
        raw["entrypoint"] = entrypoint
    for section, values in sections.items():
        raw.setdefault(section, {}).update(values)
    config = tmp_path / "rl.yaml"
    config.write_text(yaml.safe_dump(raw, sort_keys=False))
    return config


def _translation_warnings(caplog) -> list[logging.LogRecord]:
    return [
        record for record in caplog.records if record.name == TRANSLATION_LOGGER and record.levelno >= logging.WARNING
    ]


@pytest.mark.parametrize(
    "entrypoint,trainer,warned",
    [
        ("sync", {"fully_async": {"max_staleness_steps": 4}}, True),
        ("sync", {}, False),
        ("fully_async", {"fully_async": {"max_staleness_steps": 4}}, False),
        ("terminal_bench", {"fully_async": {"max_staleness_steps": 4}, "placement": {"colocate_all": False}}, False),
        ("terminal_bench", {"fully_async": {"max_staleness_steps": 4}, "placement": {"colocate_all": True}}, True),
    ],
)
def test_fully_async_settings_are_reported_inert_only_when_the_fully_async_trainer_never_runs(
    tmp_path, caplog, entrypoint, trainer, warned
):
    config = _config_with(tmp_path, entrypoint, trainer=trainer)

    parse_rl_config(str(config))

    assert bool(_translation_warnings(caplog)) is warned


def test_the_old_entrypoint_name_still_resolves_and_warns_once(tmp_path, caplog):
    config = _config_with(tmp_path, "standard")

    parsed = parse_rl_config(str(config))

    assert parsed.entrypoint == SYNC_MODULE
    assert len(_translation_warnings(caplog)) == 1


def _dry_run(tmp_path: Path, config: Path, entrypoint: str | None) -> str:
    resolved = tmp_path / "resolved.json"
    runner = LocalRLRunner(
        LocalRLConfig(
            rl_config_path=str(config),
            job_name="guard",
            model_path="Qwen/Qwen3-8B",
            entrypoint=entrypoint,
            experiments_dir=str(tmp_path / "experiments"),
            resolved_config_uri=str(resolved),
            dry_run=True,
        )
    )
    assert runner.run() == 0
    return json.loads(resolved.read_text())["entrypoint"]


def test_a_launcher_entrypoint_matching_the_config_launches(tmp_path):
    assert _dry_run(tmp_path, _config_with(tmp_path), SYNC_MODULE) == SYNC_MODULE


def test_a_non_training_launcher_entrypoint_overrides_a_sync_config(tmp_path):
    assert _dry_run(tmp_path, _config_with(tmp_path), GENERATE_MODULE) == GENERATE_MODULE


def test_a_launcher_entrypoint_naming_another_training_loop_fails(tmp_path):
    with pytest.raises(ValueError, match="contradicts the RL config's entrypoint"):
        _dry_run(tmp_path, _config_with(tmp_path), FULLY_ASYNC_MODULE)


def test_external_rl_config_rejects_deleted_module_path_before_dry_run(tmp_path):
    config = tmp_path / "rl.yaml"
    config.write_text("entrypoint: examples.terminal_bench.entrypoints.main_tbench\n")
    args = create_parser().parse_args(["--rl_config", str(config), "--model_path", "Qwen/Qwen3-8B", "--dry-run"])

    with pytest.raises(SystemExit, match="examples.terminal_bench.entrypoints.main_tbench"):
        normalize(args)


def test_rl_config_resolves_named_terminal_bench_entrypoint(tmp_path):
    config = tmp_path / "rl.yaml"
    config.write_text(
        """\
entrypoint: terminal_bench
context_budget:
  request_window_tokens: 2
  max_new_tokens_per_turn: 1
  max_turns: 1
"""
    )

    parsed = parse_rl_config(str(config))

    assert parsed.entrypoint == "skyrl_train.entrypoints.terminal_bench"


def test_rl_config_rejects_removed_opd_entrypoint(tmp_path):
    config = tmp_path / "rl.yaml"
    config.write_text(
        """\
entrypoint: terminal_bench_teacher_logits
context_budget:
  request_window_tokens: 2
  max_new_tokens_per_turn: 1
  max_turns: 1
"""
    )

    with pytest.raises(ValueError, match="terminal_bench_teacher_logits"):
        parse_rl_config(str(config))


def test_rl_config_rejects_teacher_configuration(tmp_path):
    config = tmp_path / "rl.yaml"
    config.write_text(
        """\
entrypoint: terminal_bench
context_budget:
  request_window_tokens: 2
  max_new_tokens_per_turn: 1
  max_turns: 1
teacher:
  model_path: Qwen/Qwen3-4B
"""
    )

    with pytest.raises(ValueError, match="legacy teacher configuration is not supported"):
        parse_rl_config(str(config))


def test_rl_config_translates_distillation_only_replace_mode(tmp_path):
    config = tmp_path / "rl.yaml"
    config.write_text(
        """\
entrypoint: sync
context_budget:
  request_window_tokens: 2
  max_new_tokens_per_turn: 1
  max_turns: 1
trainer:
  algorithm:
    distillation:
      objective: sampled_reverse_kl
      routing_plan: opd
      coefficient: 1.0
      reward_mode: replace
teachers:
  primary:
    source: openai_compatible
    placement: external
    model:
      path: Qwen/teacher
      revision: teacher-revision
    endpoints:
      - url: https://teacher.example/v1
        max_concurrency: 8
    tokenizer_fingerprint: sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    max_sequence_length: 32768
    request_timeout_seconds: 120
    evidence: chosen_token
teacher_routing:
  opd:
    revision: route-revision
    routes:
      default:
        teacher: primary
        weight: 1.0
"""
    )

    parsed = parse_rl_config(str(config))

    assert parsed.distillation_plan is not None
    assert parsed.distillation_plan.teachers[0].source is TeacherSource.OPENAI_COMPATIBLE
    hydra_args = build_skyrl_hydra_args(parsed, {"num_nodes": 1}, SimpleNamespace(gpus_per_node=8))

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="ppo_base_config", overrides=hydra_args)

    assert cfg.trainer.algorithm.distillation.reward_mode == "replace"


def test_supported_local_teacher_plan_crosses_cli_and_hydra_boundaries(tmp_path):
    config = tmp_path / "rl.yaml"
    config.write_text(
        """\
entrypoint: sync
context_budget:
  request_window_tokens: 2
  max_new_tokens_per_turn: 1
  max_turns: 1
trainer:
  algorithm:
    distillation:
      objective: sampled_reverse_kl
      routing_plan: opd
      coefficient: 0.25
      reward_mode: add
teachers:
  primary:
    source: local_inference
    placement: pinned
    model:
      path: Qwen/teacher
      revision: teacher-revision
    backend: vllm
    evidence: chosen_token
    resources:
      num_nodes: 1
      gpus_per_node: 8
      tensor_parallel_size: 8
      colocation_group: teacher
teacher_routing:
  opd:
    revision: route-revision
    routes:
      default:
        teacher: primary
        weight: 1.0
"""
    )

    parsed = parse_rl_config(str(config))
    hydra_args = build_skyrl_hydra_args(parsed, {"num_nodes": 2}, SimpleNamespace(gpus_per_node=8))

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="ppo_base_config", overrides=hydra_args)

    assert cfg.trainer.algorithm.distillation.objective == "sampled_reverse_kl"
    assert cfg.teachers.primary.model.revision == "teacher-revision"
    assert cfg.teacher_routing.opd.routes.default.teacher == "primary"


def test_terminal_bench_config_group_is_packaged_with_the_trainer():
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="ppo_base_config", overrides=["+terminal_bench_config=terminal_bench"])

    assert cfg.get("terminal_bench_config") is not None


def test_terminal_bench_launcher_overrides_compose_with_packaged_group():
    parsed = parse_rl_config(str(_REPO_ROOT / "cloud/iris/configs/tasktrove_dq_sweep_30b.yaml"))
    expected_prm = {
        "name": "loop_penalty",
        "window_size": 7,
        "similarity_threshold": 0.6,
        "min_turns": 11,
        "check_interval": 4,
    }
    expected_trace_upload = {
        "enabled": True,
        "repo_org": "marin-community",
        "episodes": "all",
        "dataset_type": "RL",
    }
    parsed = replace(
        parsed,
        terminal_bench={**parsed.terminal_bench, "prm": expected_prm, "trace_upload": expected_trace_upload},
    )
    hydra_args = build_skyrl_hydra_args(parsed, {"num_nodes": 4}, SimpleNamespace(gpus_per_node=8))

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="ppo_base_config", overrides=hydra_args)

    assert OmegaConf.to_container(cfg.terminal_bench_config.prm) == expected_prm
    assert OmegaConf.to_container(cfg.terminal_bench_config.trace_upload) == expected_trace_upload


def test_packed_task_source_crosses_cli_and_hydra_boundaries():
    parsed = parse_rl_config(str(_REPO_ROOT / "cloud/iris/configs/tasktrove_dq_sweep_30b.yaml"))
    source = {
        "kind": "tasktrove_parquet",
        "uri": "s3://tasktrove/clean/part-00000.parquet",
        "local_path": "/tmp/tasktrove/clean",
        "relative_path": "part-00000.parquet",
        "selection": {
            "sources": ["DCAgent2__nl2bash"],
            "tags": ["bash", "terminal"],
            "modes": ["script"],
            "tag_match": "all",
            "limit": None,
            "seed": 17,
        },
    }
    train_data = parse_list_arg(json.dumps([source]))

    hydra_args = build_skyrl_hydra_args(
        parsed,
        {"num_nodes": 4, "train_data": train_data},
        SimpleNamespace(gpus_per_node=8),
    )

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="ppo_base_config", overrides=hydra_args)

    assert OmegaConf.to_container(cfg.data.train_data) == [source]


def test_trajectory_runner_settings_reach_skyrl_hydra_config(tmp_path):
    config = tmp_path / "rl.yaml"
    config.write_text(
        """\
entrypoint: terminal_bench
context_budget:
  request_window_tokens: 2
  max_new_tokens_per_turn: 1
  max_turns: 1
trajectory_runner:
  process_pool:
    num_coordinators: 4
    cpus_per_coordinator: 8
    rpc_timeout_seconds: 1200
"""
    )

    parsed = parse_rl_config(str(config))
    hydra_args = build_skyrl_hydra_args(parsed, {"num_nodes": 4}, SimpleNamespace(gpus_per_node=8))

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="ppo_base_config", overrides=hydra_args)

    process_pool = OmegaConf.to_container(cfg.trajectory_runner.process_pool)
    assert process_pool["num_coordinators"] == 4
    assert process_pool["cpus_per_coordinator"] == 8
    assert process_pool["rpc_timeout_seconds"] == 1200
    assert process_pool["executor_workers"] == 256


def test_nemotron_ultra_judge_secret_reference_composes():
    parsed = parse_rl_config(str(_REPO_ROOT / "cloud/iris/configs/snowball_ultra_rlvr1_split64.yaml"))
    hydra_args = build_skyrl_hydra_args(parsed, {"num_nodes": 8}, SimpleNamespace(gpus_per_node=8))
    source_config_dir = str(_REPO_ROOT / "skyrl-train/skyrl_train/config")

    with initialize_config_dir(config_dir=source_config_dir, version_base=None):
        cfg = compose(config_name="ppo_base_config", overrides=hydra_args)

    ultra = cfg.environment.skyrl_gym.nemotron_ultra
    assert ultra.judges.general.api_key_env == "TOGETHER_API_KEY"
    assert ultra.judges.general.reasoning_effort == "low"
    assert ultra.genrm.judge.api_key_env == "TOGETHER_API_KEY"
    assert ultra.genrm.judge.response_transport == "chat_completions"
