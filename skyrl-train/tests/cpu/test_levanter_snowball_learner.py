# SPDX-FileCopyrightText: 2026 NovaSkyAI
# SPDX-License-Identifier: Apache-2.0

"""Numerical and durable-state checks for the optional Levanter learner."""

from __future__ import annotations

import asyncio
from dataclasses import replace
import json
import os
import socket
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
    _SNOWBALL_CE_BLOCK_SIZES,
    _all_next_token_log_probs,
    _parameter_probe,
    _replicated_host_copy,
    _regular_grpo_loss,
    _resolve_local_model_snapshot,
    prepare_snowball_batch,
)
from skyrl_train.models.grug_moe import GrugMoeConfig, GrugMoeForCausalLM
from skyrl_train.weight_sync.install_receipt import weight_name_digest
from skyrl_train.weight_sync.vllm_weight_conversion import (
    expected_expert_slice_names,
    expected_vllm_parameter_names,
)
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


def test_logprob_loss_forces_bounded_xla_stream(monkeypatch):
    Batch = Axis("batch", 2)
    Pos = Axis("position", 5)
    Embed = Axis("embed", 3)
    Vocab = Axis("vocab", 7)
    tokens = hax.named(jnp.arange(Batch.size * Pos.size, dtype=jnp.int32).reshape(Batch.size, Pos.size), (Batch, Pos))

    class Model:
        def activations(self, _tokens):
            return hax.named(jnp.ones((Batch.size, Pos.size, Embed.size), dtype=jnp.bfloat16), (Batch, Pos, Embed))

        def get_lm_head(self):
            return hax.named(jnp.ones((Embed.size, Vocab.size), dtype=jnp.bfloat16), (Embed, Vocab))

    calls = []

    def fused(hidden, lm_head, labels, **kwargs):
        calls.append((hidden.shape, lm_head.shape, labels.shape, kwargs))
        return jnp.ones(labels.shape, dtype=jnp.float32)

    monkeypatch.setattr("skyrl_train.learners.levanter_snowball.fused_linear_softmax_cross_entropy_loss", fused)
    monkeypatch.setattr(jax.sharding, "reshard", lambda value, _spec: value)

    result = _all_next_token_log_probs(Model(), tokens, 1.0)

    np.testing.assert_array_equal(result, -np.ones((Batch.size, Pos.size - 1), dtype=np.float32))
    assert calls == [
        (
            (Batch.size, Pos.size - 1, Embed.size),
            (Embed.size, Vocab.size),
            (Batch.size, Pos.size - 1),
            {
                "reduction": "none",
                "dtype": jnp.float32,
                "implementation": "xla",
                "block_sizes": _SNOWBALL_CE_BLOCK_SIZES,
            },
        )
    ]


def test_hub_model_is_resolved_to_the_pinned_local_snapshot(monkeypatch, tmp_path):
    revision = "6808fe5c219471517bd51df35addefd38ebebf89"
    calls = []

    def resolve(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        return "/cache/exact-snapshot"

    monkeypatch.setattr("skyrl_train.learners.levanter_snowball.snapshot_download", resolve)

    assert _resolve_local_model_snapshot("marin-community/model", revision) == "/cache/exact-snapshot"
    assert calls == [
        (
            "marin-community/model",
            {"revision": revision, "local_files_only": True},
        )
    ]
    assert _resolve_local_model_snapshot(str(tmp_path), revision) == str(tmp_path)


def _snowball_config() -> SnowballConfig:
    return SnowballConfig(**_MODEL_VALUES, attention_implementation="reference")


def _runtime(log_dir: Path) -> LevanterSnowballRuntimeConfig:
    return LevanterSnowballRuntimeConfig(
        model_path="unused-test-model",
        seed=7,
        training_nodes=1,
        training_gpus_per_node=1,
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


def test_publication_rejects_one_missing_expert_slice_receipt(tmp_path):
    learner = _make_learner(tmp_path / "missing-expert-receipt")
    state_dict = learner.model.to_state_dict()
    names = list(state_dict)
    parameters = sorted(expected_vllm_parameter_names(names))
    expert_slices = [
        expert_slice
        for name, value in state_dict.items()
        for expert_slice in expected_expert_slice_names(name, value.shape[0])
    ]
    assert expert_slices

    class FakeInferenceClient:
        async def begin_weight_reload(self):
            return None

        async def finish_weight_reload(self):
            return [
                {
                    "kind": "weight_install_receipt",
                    "finalized": True,
                    "received_weight_count": len(names),
                    "received_name_digest": weight_name_digest(names),
                    "loaded_parameter_count": len(parameters),
                    "loaded_parameter_digest": weight_name_digest(parameters),
                    "loaded_expert_slices": expert_slices[:-1],
                    "host": "fake-worker",
                }
            ]

        async def reset_prefix_cache(self):
            return None

    async def discard_publication(_batch):
        return None

    learner._inference_client = FakeInferenceClient()
    learner._weight_group = object()
    learner._publish_weight_batch = discard_publication

    with pytest.raises(RuntimeError, match="rank-zero weight publication failed") as error:
        asyncio.run(learner.publish_policy())
    assert error.value.__cause__ is not None
    assert "incomplete inference expert-slice installation" in str(error.value.__cause__)
    assert learner.state.lifecycle.value == "failed"
    learner._weight_group = None
    learner.close()


def _sharded_probe_worker() -> None:
    devices = np.asarray(jax.devices())
    assert devices.size == 2
    mesh = jax.sharding.Mesh(devices, ("data",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))
    parameter = jax.device_put(jnp.arange(12, dtype=jnp.float32), sharding)
    np.testing.assert_array_equal(_parameter_probe({"parameter": parameter}, size=3), np.arange(3))


def test_microbatching_does_not_split_batch_sized_model_arrays(tmp_path):
    runtime = replace(_runtime(tmp_path / "microbatch-logs"), train_batch_size=4)
    model_config = _snowball_config()
    learner = LevanterSnowballLearner(
        runtime,
        model_factory=lambda: SnowballLMHeadModel.init(
            Axis("vocab", model_config.vocab_size),
            model_config,
            key=jax.random.key(7),
        ),
    )
    learner.initialize(_learner_config())
    assert any(
        isinstance(value, jax.Array) and value.ndim > 0 and value.shape[0] == runtime.train_batch_size
        for value in jax.tree_util.tree_leaves(learner.model)
    )

    original = _batch()
    batch = LearnerBatch(
        sequences=np.concatenate((original.sequences, original.sequences)),
        attention_mask=np.concatenate((original.attention_mask, original.attention_mask)),
        response_mask=np.concatenate((original.response_mask, original.response_mask)),
        loss_mask=np.concatenate((original.loss_mask, original.loss_mask)),
        rollout_log_probs=None,
        behavior_policy_versions=np.zeros(4, dtype=np.int64),
    )
    old_log_probs = learner.compute_log_probs(batch).policy_log_probs
    update = learner.update(
        UpdateRequest(
            batch=batch,
            advantages=np.asarray(
                [[1.0, -0.5, 0.0], [0.25, -1.0, 0.5], [1.0, -0.5, 0.0], [0.25, -1.0, 0.5]],
                dtype=np.float32,
            ),
            old_policy_log_probs=old_log_probs,
            old_policy_version=0,
            reference_log_probs=None,
            global_step=0,
            global_loss_denominator=None,
        )
    )
    assert update.status.value == "succeeded"
    assert np.isfinite(update.metrics["final_loss"])
    learner.close()


def test_standalone_scoring_uses_compute_dtype(tmp_path):
    runtime = replace(_runtime(tmp_path / "compute-dtype-logs"), compute_dtype="bfloat16")
    model_config = _snowball_config()
    learner = LevanterSnowballLearner(
        runtime,
        model_factory=lambda: SnowballLMHeadModel.init(
            Axis("vocab", model_config.vocab_size),
            model_config,
            key=jax.random.key(7),
        ),
    )
    learner.initialize(_learner_config())

    score_dtypes = []

    def score(model, tokens, _temperature):
        score_dtypes.append(model.transformer.blocks[0].attn.w_q.dtype)
        return jnp.zeros((tokens.array.shape[0], tokens.array.shape[1] - 1), dtype=jnp.float32)

    learner._score_fn = score
    result = learner.compute_log_probs(_batch())

    assert score_dtypes == [jnp.bfloat16, jnp.bfloat16]
    assert np.all(np.isfinite(result.policy_log_probs))
    learner.close()


def _four_device_learner_worker(log_dir: str) -> None:
    runtime = replace(
        _runtime(Path(log_dir)),
        training_gpus_per_node=4,
        training_gpus=4,
        train_batch_size=4,
    )
    model_config = _snowball_config()
    learner = LevanterSnowballLearner(
        runtime,
        model_factory=lambda: SnowballLMHeadModel.init(
            Axis("vocab", model_config.vocab_size),
            model_config,
            key=jax.random.key(7),
        ),
    )
    learner.initialize(_learner_config())

    w_q = learner.model.transformer.blocks[0].attn.w_q
    assert w_q.sharding.spec == jax.sharding.PartitionSpec("data", "model")
    assert w_q.addressable_shards[0].data.shape == (3, 16)

    original = _batch()
    batch = LearnerBatch(
        sequences=np.concatenate((original.sequences, original.sequences)),
        attention_mask=np.concatenate((original.attention_mask, original.attention_mask)),
        response_mask=np.concatenate((original.response_mask, original.response_mask)),
        loss_mask=np.concatenate((original.loss_mask, original.loss_mask)),
        rollout_log_probs=None,
        behavior_policy_versions=np.zeros(4, dtype=np.int64),
    )
    old_log_probs = learner.compute_log_probs(batch).policy_log_probs
    prepared = prepare_snowball_batch(batch, max_sequence_length=16)
    Batch = Axis("batch", 4)
    Pos = Axis("position", prepared.tokens.shape[1])
    dense_log_probs = learner._score_fn(
        learner.model,
        hax.named(jnp.asarray(prepared.tokens, dtype=jnp.int32), (Batch, Pos)),
        1.0,
    )
    batch_spec = dense_log_probs.sharding.spec[0]
    batch_axes = (batch_spec,) if isinstance(batch_spec, str) else batch_spec
    assert "data" in batch_axes
    assert len(dense_log_probs.addressable_shards) == 4
    assert all(shard.data.shape == (1, prepared.tokens.shape[1] - 1) for shard in dense_log_probs.addressable_shards)

    train_step = learner._trainer.train_step

    def train_step_with_sharding_check(state, *training_batch):
        for value in training_batch:
            batch_spec = value.array.sharding.spec[0]
            batch_axes = (batch_spec,) if isinstance(batch_spec, str) else batch_spec
            assert "data" in batch_axes
        return train_step(state, *training_batch)

    learner._trainer.train_step = train_step_with_sharding_check
    result = learner.update(
        UpdateRequest(
            batch=batch,
            advantages=np.asarray(
                [[1.0, -0.5, 0.0], [0.25, -1.0, 0.5], [1.0, -0.5, 0.0], [0.25, -1.0, 0.5]],
                dtype=np.float32,
            ),
            old_policy_log_probs=old_log_probs,
            old_policy_version=0,
            reference_log_probs=None,
            global_step=0,
            global_loss_denominator=None,
        )
    )
    assert result.status.value == "succeeded"
    assert np.isfinite(result.metrics["final_loss"])
    learner.close()


def _multihost_learner_worker(
    process_id: str,
    coordinator_address: str,
    log_dir: str,
    checkpoint_path: str,
    result_path: str,
    mode: str,
) -> None:
    runtime = replace(
        _runtime(Path(log_dir)),
        training_nodes=2,
        training_gpus_per_node=2,
        training_gpus=4,
        train_batch_size=16,
    )
    model_config = _snowball_config()
    learner = LevanterSnowballLearner(
        runtime,
        model_factory=lambda: SnowballLMHeadModel.init(
            Axis("vocab", model_config.vocab_size),
            model_config,
            key=jax.random.key(7),
        ),
        distributed_coordinator_address=coordinator_address,
        distributed_process_id=int(process_id),
        distributed_process_count=2,
    )
    learner.initialize(_learner_config())
    assert learner._trainer_config.device_mesh.shape == {
        "replica_dcn": 1,
        "data": 4,
        "expert": 1,
        "model": 1,
    }
    w_q = learner.model.transformer.blocks[0].attn.w_q
    assert w_q.sharding.spec == jax.sharding.PartitionSpec("data", "model")
    assert len(w_q.addressable_shards) == 2
    assert all(shard.data.shape == (3, 16) for shard in w_q.addressable_shards)
    global_values = jax.make_array_from_callback(
        (8,),
        jax.sharding.NamedSharding(learner._trainer_config.device_mesh, jax.sharding.PartitionSpec("data")),
        lambda index: np.arange(8, dtype=np.float32)[index],
    )
    assert not global_values.is_fully_addressable
    np.testing.assert_array_equal(_replicated_host_copy(global_values), np.arange(8, dtype=np.float32))
    original = _batch()
    repeats = 8
    batch = LearnerBatch(
        sequences=np.concatenate((original.sequences,) * repeats),
        attention_mask=np.concatenate((original.attention_mask,) * repeats),
        response_mask=np.concatenate((original.response_mask,) * repeats),
        loss_mask=np.concatenate((original.loss_mask,) * repeats),
        rollout_log_probs=None,
        behavior_policy_versions=np.zeros(16, dtype=np.int64),
    )
    advantages = np.tile(
        np.asarray([[1.0, -0.5, 0.0], [0.25, -1.0, 0.5]], dtype=np.float32),
        (repeats, 1),
    )
    if mode == "restore":
        learner.load_checkpoint(checkpoint_path)
        result = {f"restored::{name}": value for name, value in _multihost_state_arrays(learner).items()}
        old_log_probs = learner.compute_log_probs(batch).policy_log_probs
        update = learner.update(UpdateRequest(batch, advantages, old_log_probs, 1, None, 1, None))
        result.update(
            {
                "log_probs": old_log_probs,
                "final_loss": update.metrics["final_loss"],
                "policy_version": learner.state.policy_version,
                "update_count": learner.state.update_count,
                "restored_publication_status": "outdated",
            }
        )
        np.savez(result_path, **result)
        learner.close()
        return
    if mode != "save":
        raise ValueError(f"unknown multi-host checkpoint mode {mode!r}")

    old_log_probs = learner.compute_log_probs(batch).policy_log_probs
    update = learner.update(UpdateRequest(batch, advantages, old_log_probs, 0, None, 0, None))

    published_chunks = []

    async def record_publication(batch):
        published_chunks.append([name for name, _ in batch])

    learner._publish_weight_batch = record_publication

    if int(process_id) == 0:

        class FakeInferenceClient:
            async def begin_weight_reload(self):
                return None

            async def finish_weight_reload(self):
                state_dict = learner.model.to_state_dict()
                names = list(state_dict)
                parameters = sorted(expected_vllm_parameter_names(names))
                expert_slices = [
                    expert_slice
                    for name, value in state_dict.items()
                    for expert_slice in expected_expert_slice_names(name, value.shape[0])
                ]
                return [
                    {
                        "kind": "weight_install_receipt",
                        "finalized": True,
                        "received_weight_count": len(names),
                        "received_name_digest": weight_name_digest(names),
                        "loaded_parameter_count": len(parameters),
                        "loaded_parameter_digest": weight_name_digest(parameters),
                        "loaded_expert_slices": expert_slices,
                        "host": "fake-worker",
                    }
                ]

            async def reset_prefix_cache(self):
                return None

        learner._inference_client = FakeInferenceClient()
        learner._weight_group = object()

    asyncio.run(learner.publish_policy())
    installed_publication_status = learner.state.publication_status.value
    if int(process_id) == 0:
        assert published_chunks
        learner._weight_group = None
    learner.save_checkpoint(checkpoint_path)
    result = {f"checkpoint::{name}": value for name, value in _multihost_state_arrays(learner).items()}
    result.update(
        {
            "log_probs": old_log_probs,
            "final_loss": update.metrics["final_loss"],
            "policy_version": learner.state.policy_version,
            "update_count": learner.state.update_count,
            "publication_status": installed_publication_status,
        }
    )
    np.savez(result_path, **result)
    learner.close()


def _multihost_state_arrays(learner: LevanterSnowballLearner) -> dict[str, np.ndarray]:
    arrays = {
        f"parameter::{name}": _replicated_host_copy(value) for name, value in learner.model.to_state_dict().items()
    }
    arrays["training_key"] = _replicated_host_copy(jax.random.key_data(learner._trainer_state.training_key))
    arrays["optimizer_step"] = _replicated_host_copy(learner._trainer_state.step)
    for key_path, value in jax.tree_util.tree_flatten_with_path(learner._trainer_state.opt_state)[0]:
        arrays[f"optimizer::{jax.tree_util.keystr(key_path)}"] = _replicated_host_copy(value)
    return arrays


def test_parameter_probe_gathers_a_bounded_slice_from_two_devices(tmp_path):
    result = subprocess.run(
        [sys.executable, __file__, "--sharded-probe-worker"],
        cwd=Path(__file__).parents[3],
        env={
            **os.environ,
            "JAX_PLATFORMS": "cpu",
            "XLA_FLAGS": "--xla_force_host_platform_device_count=2",
            "PYTHONPATH": os.pathsep.join(filter(None, (str(Path(__file__).parents[2]), os.environ.get("PYTHONPATH")))),
        },
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"


def test_four_device_learner_shards_parameters_and_updates(tmp_path):
    result = subprocess.run(
        [sys.executable, __file__, "--four-device-learner-worker", str(tmp_path / "four-device-logs")],
        cwd=Path(__file__).parents[3],
        env={
            **os.environ,
            "JAX_PLATFORMS": "cpu",
            "XLA_FLAGS": "--xla_force_host_platform_device_count=4",
            "PYTHONPATH": os.pathsep.join(filter(None, (str(Path(__file__).parents[2]), os.environ.get("PYTHONPATH")))),
        },
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"


def test_two_process_learner_updates_and_checkpoints_collectively(tmp_path):
    source_root = Path(__file__).parents[2]
    environment = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "XLA_FLAGS": "--xla_force_host_platform_device_count=2",
        "PYTHONPATH": os.pathsep.join(filter(None, (str(source_root), os.environ.get("PYTHONPATH")))),
    }
    checkpoint_path = tmp_path / "multihost-checkpoint"

    def run_group(mode: str) -> list[np.lib.npyio.NpzFile]:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            coordinator_address = f"127.0.0.1:{listener.getsockname()[1]}"
        processes = [
            subprocess.Popen(
                [
                    sys.executable,
                    __file__,
                    "--multihost-learner-worker",
                    str(process_id),
                    coordinator_address,
                    str(tmp_path / f"{mode}-logs-{process_id}"),
                    str(checkpoint_path),
                    str(tmp_path / f"{mode}-result-{process_id}.npz"),
                    mode,
                ],
                cwd=Path(__file__).parents[3],
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for process_id in range(2)
        ]
        outputs = []
        try:
            for process in processes:
                outputs.append(process.communicate(timeout=240))
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
                    process.wait()
        for process_id, (process, (stdout, stderr)) in enumerate(zip(processes, outputs, strict=True)):
            assert process.returncode == 0, f"mode={mode} process={process_id}\nstdout={stdout}\nstderr={stderr}"
        return [np.load(tmp_path / f"{mode}-result-{process_id}.npz") for process_id in range(2)]

    saved = run_group("save")
    restored = run_group("restore")
    np.testing.assert_array_equal(saved[0]["log_probs"], saved[1]["log_probs"])
    np.testing.assert_array_equal(saved[0]["final_loss"], saved[1]["final_loss"])
    assert int(saved[0]["policy_version"]) == int(saved[1]["policy_version"]) == 1
    assert int(saved[0]["update_count"]) == int(saved[1]["update_count"]) == 1
    assert str(saved[0]["publication_status"]) == str(saved[1]["publication_status"]) == "installed"
    assert int(restored[0]["policy_version"]) == int(restored[1]["policy_version"]) == 2
    assert int(restored[0]["update_count"]) == int(restored[1]["update_count"]) == 2
    assert str(restored[0]["restored_publication_status"]) == "outdated"
    checkpoint_keys = [key for key in saved[0].files if key.startswith("checkpoint::")]
    assert checkpoint_keys
    for key in checkpoint_keys:
        restored_key = key.replace("checkpoint::", "restored::", 1)
        np.testing.assert_array_equal(saved[0][key], restored[0][restored_key], err_msg=key)
        np.testing.assert_array_equal(saved[1][key], restored[1][restored_key], err_msg=key)
    assert (checkpoint_path / "manifest.json").is_file()


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
    del old_log_probs
    old = log_probs.detach()
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

    objective_old = torch_dense_log_probs.detach()
    global_token_mean = (
        -torch.minimum(
            torch.exp(torch.clamp(torch_dense_log_probs - objective_old, -20.0, 20.0))
            * torch.as_tensor(dense_advantages),
            torch.clamp(
                torch.exp(torch.clamp(torch_dense_log_probs - objective_old, -20.0, 20.0)),
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
    assert update.metrics["ppo_ratio_mean"] == 1.0
    assert update.metrics["ppo_clip_ratio"] == 0.0
    assert update.metrics["preupdate_replay_clip_ratio"] > 0
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
        env={
            **os.environ,
            "JAX_PLATFORMS": "cpu",
            "PYTHONPATH": os.pathsep.join(filter(None, (str(Path(__file__).parents[2]), os.environ.get("PYTHONPATH")))),
        },
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
    for name, expected in expected_state.items():
        actual = restored[name]
        np.testing.assert_array_equal(actual, expected, err_msg=name)


if __name__ == "__main__":
    if sys.argv[1] == "--checkpoint-worker":
        _checkpoint_worker(*sys.argv[2:])
    elif sys.argv[1] == "--sharded-probe-worker":
        _sharded_probe_worker()
    elif sys.argv[1] == "--four-device-learner-worker":
        _four_device_learner_worker(sys.argv[2])
    elif sys.argv[1] == "--multihost-learner-worker":
        _multihost_learner_worker(*sys.argv[2:])
