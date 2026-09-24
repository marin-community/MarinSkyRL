"""Behavior tests for the structured SkyRL launch configuration."""

from __future__ import annotations

import base64
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import yaml
from omegaconf.errors import ConfigKeyError

from cloud.iris.launch_config import compose_launch_config, load_launch_config, validate_launch_config
from cloud.iris.rl_config_translation import RL_CONFIG_PAYLOAD_ENV, materialize_launch_config, parse_rl_config


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


def test_taskcompendium_source_recipe_selects_its_entrypoint(tmp_path: Path) -> None:
    path = tmp_path / "taskcompendium.yaml"
    path.write_text(
        "entrypoint: taskcompendium\n"
        "context_budget:\n"
        "  request_window_tokens: 2\n"
        "  max_new_tokens_per_turn: 1\n"
        "  max_turns: 1\n"
    )

    parsed = parse_rl_config(str(path))

    assert parsed.entrypoint == "skyrl_train.entrypoints.taskcompendium"


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
