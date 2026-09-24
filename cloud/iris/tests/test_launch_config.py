"""Behavior tests for the structured SkyRL launch configuration."""

from __future__ import annotations

import base64
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import yaml
from omegaconf import OmegaConf
from omegaconf.errors import ConfigKeyError

from cloud.iris.launch_config import compose_launch_config, load_launch_config, validate_launch_config
from cloud.iris.rl_config_translation import (
    RL_CONFIG_PAYLOAD_ENV,
    RL_ENTRYPOINTS,
    RLEntrypoint,
    materialize_launch_config,
    parse_rl_config,
)


def _raw_config() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run": {
            "id": "smoke",
            "attempt_id": "attempt-1",
            "seed": 42,
            "mode": "train",
            "submission": "wait",
            "export_hf": True,
        },
        "runtime": {
            "launcher_commit": "a" * 40,
            "profile": "fsdp",
            "entrypoint": "skyrl_train.entrypoints.fully_async",
        },
        "iris": {
            "cluster": "cw-us-east-08a",
            "cluster_config": "configs/cw-us-east-08a.yaml",
            "job_name": "smoke",
            "wandb_entity": None,
            "allocation": {
                "num_nodes": 1,
                "gpus_per_node": 8,
                "gpu_variant": "H100",
                "cpu": 48,
                "memory": "700GB",
                "disk": "4TB",
            },
        },
        "ray": {
            "port": 6379,
            "spill_backend": "local",
            "spill_dir": "/tmp/skyrl-ray-spill",
            "rendezvous_dir": "s3://runs/smoke/rendezvous",
            "log_dir": "s3://runs/smoke/ray-logs",
        },
        "artifacts": {
            "checkpoint_root": "s3://runs/smoke/checkpoints",
            "export_root": "s3://runs/smoke/exports",
            "attempts_root": "s3://runs/smoke/attempts",
            "resolved_config_uri": "s3://runs/smoke/resolved.yaml",
            "terminal_manifest_uri": "s3://runs/smoke/terminal.json",
        },
        "inputs": {
            "model": {
                "uri": "s3://models/qwen",
                "identity": "sha256:model",
                "local_path": "/tmp/models/qwen",
                "tokenizer_uri": "s3://models/qwen-tokenizer",
                "tokenizer_revision": "a" * 40,
            },
            "data_kind": "tasks",
            "train_data": [],
            "validation_data": [],
        },
        "skyrl": {
            "entrypoint": "fully_async",
            "context_budget": {
                "request_window_tokens": 1024,
                "max_new_tokens_per_turn": 256,
                "max_turns": 1,
            },
            "model_num_attention_heads": 8,
            "trainer": {
                "seed": 42,
                "strategy": "fsdp2",
                "algorithm": {"use_kl_loss": False},
                "placement": {
                    "colocate_all": True,
                    "policy_num_nodes": 1,
                    "policy_num_gpus_per_node": 8,
                },
                "train_batch_size": 8,
                "policy_mini_batch_size": 8,
                "micro_train_batch_size_per_gpu": 1,
            },
            "generator": {
                "backend": "vllm",
                "run_engines_locally": True,
                "num_inference_engines": 8,
                "inference_engine_tensor_parallel_size": 1,
                "n_samples_per_prompt": 1,
            },
        },
    }


def test_launch_config_composes_and_loads_as_structured_hydra(tmp_path: Path) -> None:
    path = tmp_path / "resolved-launch.yaml"
    path.write_text(yaml.safe_dump(_raw_config(), sort_keys=False))

    config = load_launch_config(path)

    assert config.skyrl.trainer.train_batch_size == 8
    assert validate_launch_config(config).num_nodes == 1


@pytest.mark.parametrize(
    ("entrypoint", "colocate_all", "num_nodes", "expected"),
    [
        ("fully_async", True, 1, "async"),
        ("sync", True, 1, "sync"),
        ("terminal_bench", True, 1, "sync"),
        ("terminal_bench", False, 2, "async"),
        ("terminal_bench", None, 1, "sync"),
        ("generate", True, 1, None),
    ],
)
def test_composed_launch_records_the_trainer_its_entrypoint_runs(
    tmp_path: Path, entrypoint: str, colocate_all: bool | None, num_nodes: int, expected: str | None
) -> None:
    raw = _raw_config()
    raw["skyrl"]["entrypoint"] = entrypoint
    raw["skyrl"]["trainer"]["placement"]["colocate_all"] = colocate_all
    raw["iris"]["allocation"]["num_nodes"] = num_nodes
    path = tmp_path / "launch.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))

    config = load_launch_config(path)
    assert config.runtime.training_type == expected
    if colocate_all is None:
        # Composition fills a null colocate_all, but a resolved document can carry one. The role
        # plan then places the roles on separate nodes while the entrypoint still trains synchronously.
        config.skyrl.trainer.placement.colocate_all = None
        config.iris.allocation.num_nodes = 2
        resolved = tmp_path / "resolved-launch.yaml"
        OmegaConf.save(config, resolved)
        assert load_launch_config(resolved).runtime.training_type == expected


def test_launch_config_rejects_unknown_root_fields() -> None:
    raw = _raw_config()
    raw["unexpected"] = True

    with pytest.raises(ConfigKeyError):
        compose_launch_config(raw)


def test_launch_config_rejects_allocation_smaller_than_role_plan() -> None:
    raw = _raw_config()
    raw["iris"]["allocation"]["num_nodes"] = 0

    with pytest.raises(ValueError, match="num_nodes"):
        validate_launch_config(compose_launch_config(raw))


def test_fully_async_launch_requires_equal_training_batches() -> None:
    raw = deepcopy(_raw_config())
    raw["skyrl"]["trainer"]["policy_mini_batch_size"] = 4

    with pytest.raises(ValueError, match="train_batch_size == trainer.policy_mini_batch_size"):
        validate_launch_config(compose_launch_config(raw))


def test_colocated_rollout_parallelism_must_fit_the_allocated_bundle() -> None:
    raw = deepcopy(_raw_config())
    raw["skyrl"]["generator"]["inference_engine_data_parallel_size"] = 2

    with pytest.raises(ValueError, match="colocated rollout geometry"):
        validate_launch_config(compose_launch_config(raw))


def test_task_materializes_the_forwarded_launch_document(tmp_path: Path) -> None:
    destination = tmp_path / "launch.yaml"
    contents = yaml.safe_dump(_raw_config()).encode()

    path = materialize_launch_config(
        str(destination),
        {RL_CONFIG_PAYLOAD_ENV: base64.b64encode(contents).decode("ascii")},
    )

    assert path == str(destination)
    assert destination.read_bytes() == contents


TRANSLATION_LOGGER = "cloud.iris.rl_config_translation"


def _translation_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == TRANSLATION_LOGGER and record.levelno >= logging.WARNING
    ]


def _launch_document(tmp_path: Path, entrypoint: str, trainer: dict[str, Any], *, recipe_only: bool = False) -> Path:
    raw = _raw_config()
    raw["skyrl"]["entrypoint"] = entrypoint
    raw["skyrl"]["trainer"].update(deepcopy(trainer))
    path = tmp_path / ("recipe.yaml" if recipe_only else "launch.yaml")
    path.write_text(yaml.safe_dump(raw["skyrl"] if recipe_only else raw, sort_keys=False))
    return path


@pytest.mark.parametrize(
    ("entrypoint", "trainer", "warned"),
    [
        ("sync", {"fully_async": {"max_staleness_steps": 4}}, True),
        ("sync", {}, False),
        ("fully_async", {"fully_async": {"max_staleness_steps": 2}}, False),
        ("terminal_bench", {"fully_async": {"max_staleness_steps": 4}}, True),
    ],
)
def test_a_source_recipe_reports_fully_async_settings_its_trainer_never_reads(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, entrypoint: str, trainer: dict[str, Any], warned: bool
) -> None:
    with caplog.at_level(logging.WARNING, logger=TRANSLATION_LOGGER):
        parse_rl_config(str(_launch_document(tmp_path, entrypoint, trainer, recipe_only=True)))

    warnings = [message for message in _translation_warnings(caplog) if "never reads" in message]
    assert bool(warnings) is warned


def test_a_composed_document_reports_only_non_default_fully_async_settings(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    config = load_launch_config(_launch_document(tmp_path, "sync", {}))
    config.skyrl.trainer.fully_async.pause_mode = "keep"
    caplog.clear()

    with caplog.at_level(logging.WARNING, logger=TRANSLATION_LOGGER):
        validate_launch_config(config)

    assert _translation_warnings(caplog) == [
        "launch config smoke: entrypoint skyrl_train.entrypoints.main_base never runs the fully async trainer "
        "and never reads trainer.fully_async.pause_mode"
    ]


def test_the_old_sync_entrypoint_name_resolves_with_a_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger=TRANSLATION_LOGGER):
        parsed = parse_rl_config(str(_launch_document(tmp_path, "standard", {}, recipe_only=True)))

    assert parsed.entrypoint == RL_ENTRYPOINTS[RLEntrypoint.SYNC]
    assert any("old name for 'sync'" in message for message in _translation_warnings(caplog))


def test_fully_async_and_generator_knobs_compose_into_the_launch_document(tmp_path: Path) -> None:
    fully_async = {
        "first_token_admission": True,
        "pause_mode": "keep",
        "clear_kv_cache_on_weight_sync": False,
        "max_buffered_groups": 16,
    }
    raw = _raw_config()
    raw["skyrl"]["trainer"]["fully_async"] = fully_async
    raw["skyrl"]["generator"]["weight_sync_transport"] = "auto"
    path = tmp_path / "launch.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))

    config = load_launch_config(path)

    for key, value in fully_async.items():
        assert config.skyrl.trainer.fully_async[key] == value
    assert config.skyrl.generator.weight_sync_transport == "auto"
