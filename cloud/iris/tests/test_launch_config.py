"""Behavior tests for the structured SkyRL launch configuration."""

from __future__ import annotations

import base64
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

from cloud.iris.launch_config import compose_launch_config, load_launch_config, validate_launch_config
from cloud.iris.rl_config_translation import RL_CONFIG_PAYLOAD_ENV, materialize_launch_config
from skyrl_train.objective.objective import build_objective_micro_batch, compute_policy_objective
from skyrl_train.objective.reduction import step_counts
from skyrl_train.objective.teacher import teacher_advantages
from skyrl_train.utils.advantage_estimators import compute_advantages_and_returns
from skyrl_train.utils.algorithm_registry import PolicyLossRegistry
from skyrl_train.utils import validate_cfg


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


@pytest.mark.parametrize(("loss", "reduction"), [("regular", "token_mean"), ("gspo", "sequence_mean")])
def test_launch_config_composes_and_loads_as_structured_hydra(tmp_path: Path, loss: str, reduction: str) -> None:
    path = tmp_path / "resolved-launch.yaml"
    raw = _raw_config()
    raw["skyrl"]["trainer"]["algorithm"].update(policy_loss_type=loss, loss_reduction=reduction)
    path.write_text(yaml.safe_dump(raw, sort_keys=False))

    config = load_launch_config(path)

    assert config.skyrl.trainer.train_batch_size == 8
    assert validate_launch_config(config).num_nodes == 1
    if loss == "gspo":
        config.skyrl.trainer.algorithm.loss_reduction = "token_mean"
        with pytest.raises(ValueError, match="gspo requires trainer.algorithm.loss_reduction=sequence_mean"):
            validate_launch_config(config)
        raw["skyrl"]["trainer"]["algorithm"]["loss_reduction"] = "token_mean"
        path.write_text(yaml.safe_dump(raw, sort_keys=False))
        with pytest.raises(ValueError, match="gspo requires trainer.algorithm.loss_reduction=sequence_mean"):
            load_launch_config(path)


@pytest.mark.parametrize(("key", "value"), [("use_tis", True), ("tis_imp_ratio_cap", 2.0)])
def test_composed_launch_rejects_tis_selectors_at_launch_and_startup(tmp_path: Path, key: str, value) -> None:
    path = tmp_path / "launch.yaml"
    path.write_text(yaml.safe_dump(_raw_config()))
    config = load_launch_config(path)
    OmegaConf.update(config.skyrl.trainer.algorithm, key, value, force_add=True)
    OmegaConf.save(config, path)

    with pytest.raises(ValueError, match="off_policy_correction"):
        load_launch_config(path)
    with pytest.raises(ValueError, match="off_policy_correction"):
        validate_cfg(config.skyrl)


@pytest.mark.parametrize("recipe", ["grpo", "dapo", "dr_grpo", "gspo", "cispo", "opd", "mopd"])
def test_algorithm_recipe_launch_drives_policy_value_and_gradient(tmp_path: Path, recipe: str):
    raw = _raw_config()
    raw["skyrl"]["config_groups"] = {"algorithm_recipe": recipe}
    raw["skyrl"]["generator"]["n_samples_per_prompt"] = 2
    teacher_recipe = recipe in {"opd", "mopd"}
    if teacher_recipe:
        raw["skyrl"]["trainer"]["algorithm"]["distillation"] = {"routing_plan": "expert", "coefficient": 1.0}
        raw["skyrl"]["teachers"] = {
            "expert": dict(
                source="openai_compatible",
                placement="external",
                evidence="chosen_token",
                model=dict(path="teacher", revision="teacher-revision"),
                endpoints=[dict(url="https://teacher.example/v1", max_concurrency=1)],
                tokenizer_fingerprint=f"sha256:{'a' * 64}",
                max_sequence_length=1024,
                request_timeout_seconds=30,
            )
        }
        raw["skyrl"]["teacher_routing"] = {
            "expert": dict(revision="route-revision", routes=dict(default=dict(teacher="expert", weight=1.0)))
        }
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(raw))
    config = load_launch_config(path).skyrl.trainer.algorithm

    mask = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    old = torch.full_like(mask, -2.0)
    current = (old + torch.tensor([[1.1, 1.1], [1.25, 1.25]]).log()).requires_grad_()
    if teacher_recipe:
        advantages, _ = teacher_advantages(
            old + torch.tensor([[-1.0, 0.0], [1.0, 1.0]]),
            old,
            mask.bool(),
            torch.ones_like(mask),
            None,
        )
    else:
        advantages, _ = compute_advantages_and_returns(
            token_level_rewards=torch.tensor([[0.0, 0.0], [0.0, 2.0]]),
            response_mask=mask,
            index=np.array(["prompt", "prompt"]),
            adv_estimator=config.advantage_estimator,
            config=config,
            grpo_norm_by_std=config.grpo_norm_by_std,
        )
    batch = build_objective_micro_batch(
        action_log_probs=current,
        old_action_log_probs=old,
        base_action_log_probs=None,
        advantages=advantages,
        loss_mask=mask,
        rollout_logprobs=None,
        response_span_tags=None,
        token_entropy=torch.zeros_like(mask),
        think_token_weight=1,
        teacher=None,
    )
    counts = step_counts([mask], [mask], [], [advantages], 8, lambda value: value)
    result = compute_policy_objective(
        batch,
        loss=PolicyLossRegistry.get(config.policy_loss_type),
        counts=counts,
        config=config,
        loss_scale=1,
        report_scale=1,
    )
    scale = 1.0 if teacher_recipe or recipe == "dr_grpo" else 1 / (math.sqrt(2) + 1e-6)
    denominator = {"grpo": 2, "dapo": 3, "dr_grpo": 16, "gspo": 2, "cispo": 3, "opd": 3, "mopd": 2}[recipe]
    upper = {"grpo": 1.2, "dr_grpo": 1.2, "gspo": 1.0004}.get(recipe, 1.25)
    second_weight = 1.0 if recipe in {"grpo", "gspo", "mopd"} else 2.0
    expected_value = (1.1 - second_weight * upper) * scale / denominator
    if recipe == "cispo":
        expected_value = (1.1 * (-2 + math.log(1.1)) - 2.5 * (-2 + math.log(1.25))) * scale / 3
    torch.testing.assert_close(result.optimization_loss, torch.tensor(expected_value), rtol=1e-5, atol=1e-7)
    result.optimization_loss.backward()
    positive_gradient = -1.25 * scale / denominator if upper == 1.25 else 0.0
    if recipe == "mopd":
        positive_gradient /= 2
    expected_gradient = torch.tensor([[1.1 * scale / denominator, 0], [positive_gradient, positive_gradient]])
    torch.testing.assert_close(current.grad, expected_gradient, rtol=1e-5, atol=1e-7)


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
    raw["iris"]["allocation"]["num_nodes"] = 2
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
