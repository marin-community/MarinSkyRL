# SPDX-FileCopyrightText: 2026 NovaSkyAI
# SPDX-License-Identifier: Apache-2.0

"""Numerical and durable-state checks for the optional Levanter learner."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("levanter")

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
from haliax import Axis
from levanter.models.snowball import SnowballConfig, SnowballLMHeadModel
from skyrl_train.learner import (
    LearnerBatch,
    LearnerConfig,
    LossNormalization,
    PolicyLoss,
    UpdateRequest,
)
from skyrl_train.learners.levanter_config import LevanterSnowballRuntimeConfig
from skyrl_train.learners.levanter_snowball import (
    LevanterSnowballLearner,
    _parameter_probe,
    _regular_grpo_loss,
    prepare_snowball_batch,
)
from skyrl_train.models.grug_moe import GrugMoeConfig, GrugMoeForCausalLM
from transformers import AutoConfig

_MODEL_VALUES = {
    "vocab_size": 24,
    "hidden_dim": 12,
    "intermediate_dim": 16,
    "shared_expert_intermediate_dim": 12,
    "num_experts": 4,
    "num_experts_per_token": 2,
    "num_layers": 2,
    "num_heads": 2,
    "num_kv_heads": 1,
    "head_dim": 8,
    "max_seq_len": 16,
    "sliding_window": 4,
    "qk_mult": 1.37,
    "layer_norm_eps": 1e-5,
    "initializer_std": 0.02,
}


def test_import_preserves_msrl_grug_transformers_registration():
    assert type(AutoConfig.for_model("grug_moe")) is GrugMoeConfig


def _snowball_config() -> SnowballConfig:
    return SnowballConfig(**_MODEL_VALUES, attention_implementation="reference")


def _runtime(log_dir: Path) -> LevanterSnowballRuntimeConfig:
    return LevanterSnowballRuntimeConfig(
        model_path="unused-test-model",
        seed=7,
        training_gpus=1,
        inference_world_size=1,
        train_batch_size=2,
        micro_train_batch_size_per_gpu=1,
        micro_forward_batch_size_per_gpu=1,
        num_train_steps=4,
        learning_rate=1e-5,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        weight_decay=0.01,
        max_grad_norm=0.5,
        parameter_dtype="float32",
        compute_dtype="float32",
        output_dtype="float32",
        attention_implementation="reference",
        moe_implementation="ring",
        publication_backend="gloo",
        publication_max_chunk_bytes=1 << 20,
        publication_timeout_seconds=10,
        generator_dtype="float32",
        require_accelerator=False,
        log_dir=str(log_dir),
    )


def _learner_config() -> LearnerConfig:
    return LearnerConfig(
        policy_loss=PolicyLoss.REGULAR,
        loss_normalization=LossNormalization.TOKEN_MEAN,
        requires_reference_log_probs=False,
        clip_low=0.2,
        clip_high=0.2,
        dual_clip_ratio=3.0,
        reference_kl_coefficient=None,
        kl_estimator_type="k3",
        use_absolute_kl=False,
        use_rollout_importance_sampling=False,
        rollout_importance_ratio_cap=-1.0,
        update_epochs=1,
        logprob_temperature=1.0,
        max_sequence_length=16,
    )


def _batch() -> LearnerBatch:
    return LearnerBatch(
        sequences=np.asarray(
            [
                [0, 0, 3, 4, 11, 12, 0],
                [0, 7, 8, 9, 13, 14, 15],
            ],
            dtype=np.int32,
        ),
        attention_mask=np.asarray(
            [
                [0, 0, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 1],
            ],
            dtype=np.int32,
        ),
        response_mask=np.asarray([[1, 1, 0], [1, 1, 1]], dtype=np.int32),
        loss_mask=np.asarray([[1, 1, 0], [1, 1, 1]], dtype=np.float32),
        rollout_log_probs=None,
        behavior_policy_versions=np.asarray([0, 0], dtype=np.int64),
    )


def _make_learner(log_dir: Path) -> LevanterSnowballLearner:
    model_config = _snowball_config()
    learner = LevanterSnowballLearner(
        _runtime(log_dir),
        model_factory=lambda: SnowballLMHeadModel.init(
            Axis("vocab", model_config.vocab_size),
            model_config,
            key=jax.random.key(7),
        ),
    )
    learner.initialize(_learner_config())
    return learner


def test_publication_keeps_grug_experts_stacked_for_vllm(tmp_path):
    learner = _make_learner(tmp_path / "publication-logs")
    published = dict(learner._iter_publication_tensors())
    stacked_name = "model.layers.0.mlp.experts.gate_proj.weight"

    assert stacked_name in published
    assert not any(".mlp.experts.0." in name for name in published)
    np.testing.assert_array_equal(
        published[stacked_name].numpy(),
        np.asarray(learner.model.to_state_dict()[stacked_name]),
    )
    learner.close()


def _sharded_probe_worker() -> None:
    devices = np.asarray(jax.devices())
    assert devices.size == 2
    mesh = jax.sharding.Mesh(devices, ("data",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))
    parameter = jax.device_put(jnp.arange(12, dtype=jnp.float32), sharding)
    np.testing.assert_array_equal(_parameter_probe({"parameter": parameter}, size=3), np.arange(3))


def test_parameter_probe_gathers_a_bounded_slice_from_two_devices(tmp_path):
    result = subprocess.run(
        [sys.executable, __file__, "--sharded-probe-worker"],
        cwd=Path(__file__).parents[3],
        env={
            **os.environ,
            "JAX_PLATFORMS": "cpu",
            "XLA_FLAGS": "--xla_force_host_platform_device_count=2",
        },
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"


def _dense_channels(batch: LearnerBatch, old_log_probs: np.ndarray, advantages: np.ndarray):
    prepared = prepare_snowball_batch(batch, max_sequence_length=16)
    return (
        prepared,
        prepared.dense_response_values(old_log_probs),
        prepared.dense_response_values(advantages),
        prepared.dense_response_values(batch.loss_mask),
    )


def _torch_model(model: SnowballLMHeadModel) -> GrugMoeForCausalLM:
    torch_model = GrugMoeForCausalLM(GrugMoeConfig(**_MODEL_VALUES)).float()
    state_dict = {name: torch.from_numpy(np.asarray(value).copy()) for name, value in model.to_state_dict().items()}
    torch_model.load_state_dict(state_dict, strict=True)
    return torch_model


def _torch_loss(torch_model, tokens, old_log_probs, advantages, loss_mask):
    token_ids = torch.as_tensor(tokens, dtype=torch.long)
    logits = torch_model(token_ids).logits.float()
    targets = token_ids[:, 1:]
    log_probs = torch.log_softmax(logits[:, :-1], dim=-1).gather(-1, targets[..., None])[..., 0]
    old = torch.as_tensor(old_log_probs)
    advantage = torch.as_tensor(advantages)
    mask = torch.as_tensor(loss_mask)
    ratio = torch.exp(torch.clamp(log_probs - old, -20.0, 20.0))
    surrogate = ratio * advantage
    clipped = torch.clamp(ratio, 0.8, 1.2) * advantage
    masked_loss = -torch.minimum(surrogate, clipped) * mask
    loss = (masked_loss.sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)).mean()
    return loss, log_probs


def test_padding_compaction_preserves_each_response_predictor_position():
    prepared = prepare_snowball_batch(_batch(), max_sequence_length=16)

    np.testing.assert_array_equal(
        prepared.tokens,
        [[3, 4, 11, 12, 0, 0, 0], [7, 8, 9, 13, 14, 15, 0]],
    )
    np.testing.assert_array_equal(prepared.response_positions, [[1, 2, 0], [2, 3, 4]])
    np.testing.assert_array_equal(prepared.response_mask, [[True, True, False], [True, True, True]])


def test_logprob_loss_gradient_and_first_adamw_update_match_torch(tmp_path):
    learner = _make_learner(tmp_path / "logs")
    batch = _batch()
    prepared = prepare_snowball_batch(batch, max_sequence_length=16)
    levanter_log_probs = learner.compute_log_probs(batch).policy_log_probs
    torch_model = _torch_model(learner.model)

    _, torch_dense_log_probs = _torch_loss(
        torch_model,
        prepared.tokens,
        np.zeros_like(prepared.tokens[:, 1:], dtype=np.float32),
        np.zeros_like(prepared.tokens[:, 1:], dtype=np.float32),
        np.ones_like(prepared.tokens[:, 1:], dtype=np.float32),
    )
    torch_log_probs = prepared.response_values(torch_dense_log_probs.detach().numpy())
    selected = batch.response_mask.astype(bool)
    logprob_deviation = np.abs(levanter_log_probs[selected] - torch_log_probs[selected])

    offsets = np.asarray([[0.25, -0.3, 0.0], [-0.5, 0.2, -0.1]], dtype=np.float32)
    old_log_probs = levanter_log_probs + offsets
    advantages = np.asarray([[1.0, -0.5, 0.0], [0.25, -1.0, 0.5]], dtype=np.float32)
    _, dense_old, dense_advantages, dense_mask = _dense_channels(batch, old_log_probs, advantages)

    torch_model.zero_grad(set_to_none=True)
    torch_loss, _ = _torch_loss(torch_model, prepared.tokens, dense_old, dense_advantages, dense_mask)
    torch_loss.backward()

    Batch = Axis("batch", 2)
    Pos = Axis("position", prepared.tokens.shape[1])
    Prediction = Axis("prediction", prepared.tokens.shape[1] - 1)

    def objective(model):
        return _regular_grpo_loss(
            model,
            hax.named(jnp.asarray(prepared.tokens), (Batch, Pos)),
            hax.named(jnp.asarray(dense_old), (Batch, Prediction)),
            hax.named(jnp.asarray(dense_advantages), (Batch, Prediction)),
            hax.named(jnp.asarray(dense_mask), (Batch, Prediction)),
            key=jax.random.key(0),
            temperature=1.0,
            clip_low=0.2,
            clip_high=0.2,
        )

    with learner._trainer_config.use_device_mesh():
        (levanter_loss, _), levanter_grad = eqx.filter_value_and_grad(objective, has_aux=True)(learner.model)

    global_token_mean = (
        -torch.minimum(
            torch.exp(torch.clamp(torch_dense_log_probs - torch.as_tensor(dense_old), -20.0, 20.0))
            * torch.as_tensor(dense_advantages),
            torch.clamp(
                torch.exp(torch.clamp(torch_dense_log_probs - torch.as_tensor(dense_old), -20.0, 20.0)),
                0.8,
                1.2,
            )
            * torch.as_tensor(dense_advantages),
        )
        * torch.as_tensor(dense_mask)
    ).sum() / torch.as_tensor(dense_mask).sum()
    assert float(torch_loss.detach()) != pytest.approx(float(global_token_mean.detach()), abs=1e-6)

    levanter_grad_state = {name: np.asarray(value) for name, value in levanter_grad.to_state_dict().items()}
    gradient_deviations = []
    for name, parameter in torch_model.named_parameters():
        gradient_deviations.append(np.abs(parameter.grad.numpy() - levanter_grad_state[name]).reshape(-1))
    gradient_deviation = np.concatenate(gradient_deviations)

    for name, parameter in torch_model.named_parameters():
        if name.endswith(".mlp.router.bias"):
            parameter.requires_grad_(False)
    trainable_parameters = [parameter for parameter in torch_model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=1e-5,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )
    torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=0.5)
    optimizer.step()

    update = learner.update(
        UpdateRequest(
            batch=batch,
            advantages=advantages,
            old_policy_log_probs=old_log_probs,
            old_policy_version=0,
            reference_log_probs=None,
            global_step=0,
            global_loss_denominator=None,
        )
    )
    updated_levanter_state = {name: np.asarray(value) for name, value in learner.model.to_state_dict().items()}
    parameter_deviations = []
    for name, parameter in torch_model.named_parameters():
        parameter_deviations.append(np.abs(parameter.detach().numpy() - updated_levanter_state[name]).reshape(-1))
    parameter_deviation = np.concatenate(parameter_deviations)

    evidence = {
        "logprob_mean_abs_diff": float(logprob_deviation.mean()),
        "logprob_max_abs_diff": float(logprob_deviation.max()),
        "loss_abs_diff": abs(float(levanter_loss) - float(torch_loss.detach())),
        "gradient_mean_abs_diff": float(gradient_deviation.mean()),
        "gradient_max_abs_diff": float(gradient_deviation.max()),
        "first_update_mean_abs_diff": float(parameter_deviation.mean()),
        "first_update_max_abs_diff": float(parameter_deviation.max()),
    }
    print(json.dumps(evidence, sort_keys=True))

    assert evidence["logprob_max_abs_diff"] < 1e-6
    assert evidence["loss_abs_diff"] < 1e-7
    assert evidence["gradient_max_abs_diff"] < 2e-6
    assert evidence["first_update_max_abs_diff"] < 2e-6
    assert update.metrics["preupdate_logprob_max_abs_diff"] == pytest.approx(0.5, abs=1e-6)
    assert update.metrics["parameter_probe_delta_l2"] > 0
    assert update.metrics["router_bias_max_delta"] == 0
    learner.close()


def _checkpoint_worker(checkpoint_path: str, request_path: str, result_path: str, log_dir: str) -> None:
    request = np.load(request_path)
    learner = _make_learner(Path(log_dir))

    async def load_inside_msrl_loop() -> None:
        learner.load_checkpoint(checkpoint_path)

    asyncio.run(load_inside_msrl_loop())
    restored_checkpoint = {f"checkpoint::{name}": value for name, value in _learner_state_arrays(learner).items()}
    batch = _batch()
    restored_log_probs = learner.compute_log_probs(batch).policy_log_probs
    update = learner.update(
        UpdateRequest(
            batch=batch,
            advantages=request["advantages"],
            old_policy_log_probs=request["old_log_probs"],
            old_policy_version=1,
            reference_log_probs=None,
            global_step=1,
            global_loss_denominator=None,
        )
    )
    result = _learner_state_arrays(learner)
    result.update(restored_checkpoint)
    result.update(
        {
            "restored_log_probs": restored_log_probs,
            "training_key": np.asarray(jax.random.key_data(learner._trainer_state.training_key)),
            "policy_version": np.asarray(learner.state.policy_version),
            "optimizer_step": np.asarray(update.metrics["optimizer_step"]),
        }
    )
    np.savez(result_path, **result)
    learner.close()


def _learner_state_arrays(learner: LevanterSnowballLearner) -> dict[str, np.ndarray]:
    arrays = {f"parameter::{name}": np.asarray(value).copy() for name, value in learner.model.to_state_dict().items()}
    arrays["training_key"] = np.asarray(jax.random.key_data(learner._trainer_state.training_key)).copy()
    arrays["optimizer_step"] = np.asarray(learner._trainer_state.step).copy()
    for key_path, value in jax.tree_util.tree_flatten_with_path(learner._trainer_state.opt_state)[0]:
        arrays[f"optimizer::{jax.tree_util.keystr(key_path)}"] = np.asarray(jax.device_get(value)).copy()
    return arrays


def test_checkpoint_resume_in_a_fresh_process_matches_the_next_update(tmp_path):
    learner = _make_learner(tmp_path / "parent-logs")
    batch = _batch()
    advantages_one = np.asarray([[1.0, -0.5, 0.0], [0.25, -1.0, 0.5]], dtype=np.float32)
    old_one = learner.compute_log_probs(batch).policy_log_probs
    learner.update(UpdateRequest(batch, advantages_one, old_one, 0, None, 0, None))

    checkpoint_path = tmp_path / "checkpoint"
    learner.save_checkpoint(str(checkpoint_path))
    expected_checkpoint_state = _learner_state_arrays(learner)
    preupdate_two = learner.compute_log_probs(batch).policy_log_probs
    old_two = preupdate_two + np.asarray([[0.13, -0.17, 0.0], [-0.11, 0.19, -0.07]], dtype=np.float32)
    advantages_two = np.asarray([[-0.75, 0.5, 0.0], [1.25, 0.25, -0.5]], dtype=np.float32)
    request_path = tmp_path / "request.npz"
    np.savez(request_path, old_log_probs=old_two, advantages=advantages_two)

    learner.update(UpdateRequest(batch, advantages_two, old_two, 1, None, 1, None))
    expected_state = _learner_state_arrays(learner)
    learner.close()

    result_path = tmp_path / "fresh-process-result.npz"
    result = subprocess.run(
        [
            sys.executable,
            __file__,
            "--checkpoint-worker",
            str(checkpoint_path),
            str(request_path),
            str(result_path),
            str(tmp_path / "child-logs"),
        ],
        cwd=Path(__file__).parents[3],
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"

    restored = np.load(result_path)
    np.testing.assert_allclose(restored["restored_log_probs"], preupdate_two, rtol=0, atol=1e-7)
    for name, expected in expected_checkpoint_state.items():
        np.testing.assert_array_equal(restored[f"checkpoint::{name}"], expected)
    np.testing.assert_array_equal(restored["training_key"], expected_state["training_key"])
    assert int(restored["policy_version"]) == int(restored["optimizer_step"]) == 2
    category_max_deviation = {"parameter": 0.0, "optimizer": 0.0, "metadata": 0.0}
    category_max_deviation_name: dict[str, str | None] = dict.fromkeys(category_max_deviation)
    for name, expected in expected_state.items():
        actual = restored[name]
        if not np.issubdtype(expected.dtype, np.inexact):
            np.testing.assert_array_equal(actual, expected)
            continue
        deviation = float(np.max(np.abs(actual - expected)))
        category = name.partition("::")[0] if "::" in name else "metadata"
        if deviation > category_max_deviation[category]:
            category_max_deviation[category] = deviation
            category_max_deviation_name[category] = name
    print(
        json.dumps(
            {
                "fresh_process_next_update_category_max_abs_diff": category_max_deviation,
                "fresh_process_next_update_category_max_abs_diff_name": category_max_deviation_name,
            }
        )
    )
    # Fresh XLA CPU processes can select slightly different floating-point
    # reduction orders. The checkpoint itself is exact above. At E6's 1e-5
    # learning rate, the replayed model update stays within 2e-5; Adam moment
    # arrays record the underlying gradient variation directly.
    assert category_max_deviation["metadata"] == 0.0
    assert category_max_deviation["parameter"] <= 2e-5
    assert category_max_deviation["optimizer"] <= 5e-3


if __name__ == "__main__":
    if sys.argv[1] == "--checkpoint-worker":
        _checkpoint_worker(*sys.argv[2:])
    elif sys.argv[1] == "--sharded-probe-worker":
        _sharded_probe_worker()
