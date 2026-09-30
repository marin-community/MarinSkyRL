"""Behavior tests for the structured SkyRL launch configuration."""

from __future__ import annotations

import base64
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import yaml

from cloud.iris.launch_config import compose_launch_config, load_launch_config, validate_launch_config
from cloud.iris.rl_config_translation import RL_CONFIG_PAYLOAD_ENV, materialize_launch_config


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
            "profile": "megatron",
            "entrypoint": "skyrl_train.entrypoints.main_base",
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
            "entrypoint": "standard",
            "context_budget": {
                "request_window_tokens": 1024,
                "max_new_tokens_per_turn": 256,
                "max_turns": 1,
            },
            "model_num_attention_heads": 8,
            "trainer": {
                "seed": 42,
                "strategy": "megatron",
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


def test_launch_preserves_unbounded_retention_at_existing_history_path(tmp_path: Path) -> None:
    raw = _raw_config()
    history_path = "s3://runs/original-profile/trajectories"
    raw["skyrl"]["generator"]["pivot_profiling_resume"] = True
    raw["skyrl"]["generator"]["trajectory_retention"] = {
        "enabled": True,
        "output_path": history_path,
        "required": True,
        "sample_fraction": 1.0,
        "phases": ["eval"],
        "max_bytes_per_step": None,
        "max_bytes_per_run": None,
    }
    path = tmp_path / "resume.yaml"
    path.write_text(yaml.safe_dump(raw))

    config = load_launch_config(path)

    assert config.skyrl.generator.trajectory_retention.output_path == history_path
    assert config.skyrl.generator.trajectory_retention.max_bytes_per_step is None
    assert config.skyrl.generator.trajectory_retention.max_bytes_per_run is None


@pytest.mark.parametrize(
    ("entrypoint", "max_staleness_steps", "expected"),
    [
        ("standard", 0, "sync"),
        ("standard", 2, "async"),
        ("terminal_bench", 1, "async"),
        ("generate", 0, None),
    ],
)
def test_composed_launch_records_whether_training_runs_ahead_of_its_updates(
    tmp_path: Path, entrypoint: str, max_staleness_steps: int, expected: str | None
) -> None:
    raw = _raw_config()
    raw["skyrl"]["entrypoint"] = entrypoint
    raw["skyrl"]["trainer"]["placement"]["colocate_all"] = False
    raw["skyrl"]["trainer"]["rollout_buffer"] = {"max_staleness_steps": max_staleness_steps}
    raw["iris"]["allocation"]["num_nodes"] = 1 if entrypoint == "generate" else 2
    path = tmp_path / "launch.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))

    assert load_launch_config(path).runtime.training_type == expected


def test_object_store_uris_in_skyrl_config_reach_the_task_unchanged(tmp_path: Path) -> None:
    raw = _raw_config()
    raw["skyrl"]["trainer"]["rollout_buffer"] = {"object_store_root": "s3://runs/smoke/rollouts"}
    path = tmp_path / "launch.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))

    config = load_launch_config(path)

    assert config.skyrl.trainer.rollout_buffer.object_store_root == "s3://runs/smoke/rollouts"


def test_qwen_smoke_accepts_hugging_face_model_input(tmp_path: Path) -> None:
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

    resolved = load_launch_config(path)

    assert resolved.skyrl.trainer.policy.model.path == "Qwen/Qwen3-0.6B"
    assert resolved.skyrl.trainer.policy.model.source_uri is None
    assert resolved.runtime.entrypoint == "skyrl_train.entrypoints.main_base"


def test_launch_config_rejects_allocation_smaller_than_role_plan() -> None:
    raw = _raw_config()
    raw["iris"]["allocation"]["num_nodes"] = 0

    with pytest.raises(ValueError, match="num_nodes"):
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


@pytest.mark.parametrize("entrypoint", ["generate", "terminal_bench_generate"])
def test_generate_only_reserves_inference_nodes_without_training_roles(tmp_path, entrypoint):
    raw = _raw_config()
    raw["skyrl"]["entrypoint"] = entrypoint
    raw["skyrl"]["trainer"]["placement"].update(colocate_all=False, policy_num_nodes=10)
    raw["skyrl"]["generator"].update(num_inference_engines=12, inference_engine_data_parallel_size=4)
    raw["iris"]["allocation"]["num_nodes"] = 6
    path = tmp_path / "generate-only.yaml"
    path.write_text(yaml.safe_dump(raw))
    config = load_launch_config(path)
    allocation = validate_launch_config(config)
    assert allocation.num_nodes * allocation.gpus_per_node == 48
