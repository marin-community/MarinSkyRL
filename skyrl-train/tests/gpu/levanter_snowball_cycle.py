# SPDX-FileCopyrightText: 2026 NovaSkyAI
# SPDX-License-Identifier: Apache-2.0

"""Opt-in four-H100 Levanter/Snowball MSRL capstone.

This file deliberately lacks the ``test_`` prefix. Run it by exact path after
reading the repository GPU testing policy.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pytest
import ray
import torch
from skyrl_train.callbacks.builtin import CheckpointCallback
from skyrl_train.inference_engines.base import InferenceEngineInput
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_train.models.grug_moe import GrugMoeConfig, GrugMoeForCausalLM
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.trajectory_runners.base import TrajectoryRunner
from skyrl_train.utils import initialize_ray
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from marinskyrl.checkpoint_paths import CHECKPOINT_COMPLETE_FILENAME, POLICY_CHECKPOINT_SUBDIRECTORY
from tests.gpu.grug_gpu_gates import require_hoppers
from tests.gpu.grug_serving import (
    LM_HEAD_NAME,
    assert_engine_weights,
    grug_engine_client,
)
from tests.gpu.utils import get_test_actor_config

ACTIVE_GPUS = 4
LEARNER_GPUS = 2
GPU_REPLAY_ATOL = 1e-4
ROUTER_BIAS_NAME = "model.layers.0.mlp.router.bias"
QUERY_NAME = "model.layers.0.self_attn.q_proj.weight"
EXPERT_ZERO_NAME = "model.layers.0.mlp.experts.0.gate_proj.weight"
EXPERT_FOUR_NAME = "model.layers.0.mlp.experts.4.gate_proj.weight"
READBACK_NAMES = [QUERY_NAME, EXPERT_ZERO_NAME, EXPERT_FOUR_NAME, LM_HEAD_NAME, ROUTER_BIAS_NAME]


class _Dataset:
    _PROMPTS = (
        [3, 17, 29, 5, 11, 7],
        [4, 19, 31, 6, 13, 8],
        [5, 23, 37, 7, 17, 9],
        [6, 29, 41, 8, 19, 10],
    )

    def __len__(self):
        return len(self._PROMPTS)

    def __getitem__(self, index):
        return {
            "uid": f"row-{index}",
            "prompt": self._PROMPTS[index],
            "env_class": None,
            "env_extras": {},
        }

    def collate_fn(self, rows):
        return rows


class _TrajectoryRunner(TrajectoryRunner):
    def __init__(self, client) -> None:
        self.client = client
        self.requested_uids: list[list[str]] = []
        self.rollouts: list[dict[str, object]] = []

    async def _run(self, input_batch, disable_tqdm: bool = False):
        del disable_tqdm
        trajectory_ids = input_batch["trajectory_ids"]
        assert trajectory_ids is not None
        self.requested_uids.append([item.instance_id for item in trajectory_ids])
        responses = []
        response_logprobs = []
        stop_reasons = []
        for prompt, trajectory_id in zip(input_batch["prompts"], trajectory_ids, strict=True):
            sampling = dict(input_batch["sampling_params"] or {})
            row = int(trajectory_id.instance_id.rpartition("-")[2])
            sampling.update(
                {
                    "temperature": 1.0,
                    "ignore_eos": True,
                    "logprobs": 1,
                    "seed": 1000 + 10 * row + trajectory_id.repetition_id,
                }
            )
            result = await self.client.generate(
                InferenceEngineInput(prompt_token_ids=[prompt], sampling_params=sampling)
            )
            responses.append(result["response_ids"][0])
            response_logprobs.append(result["response_logprobs"][0])
            stop_reasons.append(result["stop_reasons"][0])

        for start in range(0, len(responses), 2):
            assert responses[start] != responses[start + 1], "stochastic GRPO responses unexpectedly matched"
        rewards = [float(item.repetition_id) for item in trajectory_ids]
        output = {
            "prompt_token_ids": list(input_batch["prompts"]),
            "response_ids": responses,
            "rewards": rewards,
            "unshaped_rewards": rewards.copy(),
            "loss_masks": [[1] * len(response) for response in responses],
            "stop_reasons": stop_reasons,
            "rollout_metrics": {},
            "rollout_logprobs": response_logprobs,
            "is_last_step": [True] * len(responses),
            "exclude_from_baseline": [False] * len(responses),
        }
        self.rollouts.append(output)
        return output

    async def shutdown(self) -> None:
        pass


class _Tracker:
    def log(self, metrics, step: int, commit: bool = False) -> None:
        del metrics, step, commit


class _CapstoneTrainer(RayPPOTrainer):
    """Run the public loop while postponing teardown for capstone assertions."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.defer_shutdown = True
        self.shutdown_requested = False
        self.evidence = {
            "learner_initialize_seconds": [],
            "forward_seconds": [],
            "update_seconds": [],
            "publication_seconds": [],
            "checkpoint_seconds": [],
            "iteration_seconds": [],
            "resume_load_seconds": [],
            "publication_probe_scores": [],
            "publication_expert_owners": [],
            "updates": [],
        }
        self.step_one_state = None
        self.restored_state = None
        self.restored_step = None
        self.restored_policy_version = None

    def _initialize_or_validate_learner(self, *, allow_failed: bool = False):
        should_measure = self.learner.state.lifecycle.value == "uninitialized"
        started = time.perf_counter()
        result = super()._initialize_or_validate_learner(allow_failed=allow_failed)
        if should_measure:
            self.evidence["learner_initialize_seconds"].append(time.perf_counter() - started)
        return result

    def fwd_logprobs_values_reward(self, training_input):
        started = time.perf_counter()
        result = super().fwd_logprobs_values_reward(training_input)
        self.evidence["forward_seconds"].append(time.perf_counter() - started)
        return result

    def train_critic_and_policy(self, training_input):
        before = np.asarray(self.learner.model.to_state_dict()[QUERY_NAME]).copy()
        started = time.perf_counter()
        result = super().train_critic_and_policy(training_input)
        self.evidence["update_seconds"].append(time.perf_counter() - started)
        after = np.asarray(self.learner.model.to_state_dict()[QUERY_NAME]).copy()
        learning_rate = float(self.cfg.trainer.policy.optimizer_config.lr)
        weight_decay = float(self.cfg.trainer.policy.optimizer_config.weight_decay)
        decay_only = before * (1.0 - learning_rate * weight_decay)
        result["query_delta_beyond_first_step_weight_decay_l2"] = float(np.linalg.norm(after - decay_only))
        self.evidence["updates"].append(result)
        return result

    async def _sync_policy_for_rollouts(self, *, reason: str) -> None:
        started = time.perf_counter()
        await super()._sync_policy_for_rollouts(reason=reason)
        self.evidence["publication_seconds"].append(time.perf_counter() - started)
        self.evidence["publication_expert_owners"].append(_readback(self.learner, self.inference_engine_client))
        score = await _score_token(
            self.inference_engine_client,
            self.cfg,
            list(_Dataset._PROMPTS[0]),
            3,
        )
        self.evidence["publication_probe_scores"].append(score)

    def _stage_checkpoint(self):
        started = time.perf_counter()
        result = super()._stage_checkpoint()
        self.evidence["checkpoint_seconds"].append(time.perf_counter() - started)
        if self.global_step == 1:
            self.step_one_state = _learner_state_arrays(self.learner)
        return result

    def _log_training_step_completed(self, *, epoch: int, duration_seconds: float) -> None:
        self.evidence["iteration_seconds"].append(duration_seconds)
        super()._log_training_step_completed(epoch=epoch, duration_seconds=duration_seconds)

    def load_checkpoints(self):
        started = time.perf_counter()
        result = super().load_checkpoints()
        self.evidence["resume_load_seconds"].append(time.perf_counter() - started)
        self.restored_step = result[0]
        self.restored_policy_version = self.learner.state.policy_version
        self.restored_state = _learner_state_arrays(self.learner)
        return result

    async def shutdown(self) -> None:
        if self.defer_shutdown:
            self.shutdown_requested = True
            return
        await super().shutdown()

    async def release(self) -> None:
        self.defer_shutdown = False
        await super().shutdown()

    @staticmethod
    def _start_exit_watchdog(timeout: int = 120) -> None:
        del timeout


def _write_tokenizer(path: Path, vocab_size: int) -> PreTrainedTokenizerFast:
    vocab = {"<pad>": 0, "<eos>": 1, "<unk>": 2}
    vocab.update({f"token-{index}": index for index in range(3, vocab_size)})
    raw_tokenizer = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    raw_tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw_tokenizer,
        pad_token="<pad>",
        eos_token="<eos>",
        unk_token="<unk>",
    )
    tokenizer.save_pretrained(path)
    return tokenizer


def _write_tiny_checkpoint(path: Path) -> None:
    vocab_size = 64
    path.mkdir()
    _write_tokenizer(path, vocab_size)
    config = GrugMoeConfig(
        vocab_size=vocab_size,
        hidden_size=64,
        intermediate_size=64,
        shared_expert_intermediate_size=64,
        num_local_experts=8,
        num_experts_per_tok=2,
        num_hidden_layers=5,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        max_position_embeddings=128,
        sliding_window=16,
        initializer_range=0.02,
        qk_mult=1.37,
    )
    torch.manual_seed(17)
    GrugMoeForCausalLM(config).save_pretrained(path, safe_serialization=True)


def _config(model_path: str, run_path: Path):
    cfg = get_test_actor_config()
    cfg.trainer.policy.model.path = model_path
    cfg.trainer.critic.model.path = ""
    cfg.trainer.seed = 42
    cfg.trainer.max_steps = 2
    cfg.trainer.epochs = 1
    cfg.trainer.ckpt_path = str(run_path / "checkpoints")
    cfg.trainer.export_path = str(run_path / "exports")
    cfg.trainer.ckpt_interval = 1
    cfg.trainer.restore_dataloader_state = True
    cfg.trainer.strategy = "fsdp2"
    cfg.trainer.train_batch_size = 2
    cfg.trainer.policy_mini_batch_size = 2
    cfg.trainer.micro_train_batch_size_per_gpu = 1
    cfg.trainer.micro_forward_batch_size_per_gpu = 1
    cfg.trainer.update_epochs_per_batch = 1
    cfg.trainer.use_sample_packing = False
    cfg.trainer.step_wise_training = False
    cfg.trainer.algorithm.max_seq_len = 128
    cfg.trainer.algorithm.advantage_estimator = "grpo"
    cfg.trainer.algorithm.grpo_norm_by_std = True
    cfg.trainer.algorithm.use_kl_loss = False
    cfg.trainer.algorithm.use_kl_in_reward = False
    cfg.trainer.algorithm.use_entropy_loss = False
    cfg.trainer.algorithm.use_tis = False
    cfg.trainer.algorithm.policy_loss_type = "regular"
    cfg.trainer.algorithm.loss_reduction = "token_mean"
    cfg.trainer.algorithm.eps_clip_low = 0.2
    cfg.trainer.algorithm.eps_clip_high = 0.2
    cfg.trainer.algorithm.advantage_batch_normalize = False
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.placement.policy_num_nodes = 1
    cfg.trainer.placement.policy_num_gpus_per_node = LEARNER_GPUS
    cfg.trainer.policy.fsdp_config.expert_model_parallel_size = 1
    cfg.trainer.policy.grug_query_bias_update_mode = "frozen"
    cfg.trainer.policy.optimizer_config.optimizer = "AdamW"
    cfg.trainer.policy.optimizer_config.lr = 1e-5
    cfg.trainer.policy.optimizer_config.adam_betas = [0.9, 0.999]
    cfg.trainer.policy.optimizer_config.weight_decay = 0.01
    cfg.trainer.policy.optimizer_config.max_grad_norm = 0.5
    cfg.trainer.policy.optimizer_config.num_warmup_steps = 0
    cfg.trainer.policy.optimizer_config.scheduler = "constant_with_warmup"
    cfg.trainer.policy.optimizer_config.optimizer_kwargs = {"eps": 1e-8}
    cfg.trainer.policy.levanter.parameter_dtype = "float32"
    cfg.trainer.policy.levanter.compute_dtype = "bfloat16"
    cfg.trainer.policy.levanter.output_dtype = "float32"
    cfg.trainer.policy.levanter.attention_implementation = "reference"
    cfg.trainer.policy.levanter.moe_implementation = "ring"
    cfg.trainer.policy.levanter.publication_backend = "gloo"
    cfg.trainer.policy.levanter.publication_max_chunk_bytes = 1 << 20
    cfg.trainer.policy.levanter.publication_timeout_seconds = 120
    cfg.trainer.policy.levanter.require_accelerator = True
    cfg.trainer.policy.levanter.log_dir = str(run_path / "levanter-logs")
    cfg.generator.backend = "vllm"
    cfg.generator.async_engine = True
    cfg.generator.run_engines_locally = True
    cfg.generator.fuse_weights = False
    cfg.generator.weight_sync_backend = "gloo"
    cfg.generator.model_dtype = "bfloat16"
    cfg.generator.num_inference_engines = 1
    cfg.generator.inference_engine_tensor_parallel_size = 1
    cfg.generator.inference_engine_pipeline_parallel_size = 1
    cfg.generator.inference_engine_data_parallel_size = 2
    cfg.generator.inference_engine_expert_parallel_size = 2
    cfg.generator.n_samples_per_prompt = 2
    cfg.trainer.algorithm.resolved_group_advantage.physical_group_size = 2
    cfg.generator.gpu_memory_utilization = 0.35
    cfg.generator.sampling_params.temperature = 1.0
    cfg.generator.sampling_params.top_p = 1.0
    cfg.generator.sampling_params.top_k = -1
    cfg.generator.sampling_params.max_generate_length = 4
    return cfg


def _make_trainer(cfg, client, tokenizer, learner) -> RayPPOTrainer:
    trainer = _CapstoneTrainer(
        cfg=cfg,
        tracker=_Tracker(),
        tokenizer=tokenizer,
        train_dataset=_Dataset(),
        eval_dataset=None,
        inference_engine_client=client,
        trajectory_runner=_TrajectoryRunner(client),
        callbacks=[CheckpointCallback(save_steps=1)],
        learner=learner,
    )
    trainer.build_models(None, None, None)
    return trainer


async def _generate(client, cfg, prompts: list[list[int]]):
    sampling = get_sampling_params_for_backend(cfg.generator.backend, cfg.generator.sampling_params)
    sampling.update({"temperature": 0.0, "ignore_eos": True, "logprobs": 1})
    return await client.generate(InferenceEngineInput(prompt_token_ids=prompts, sampling_params=sampling))


async def _score_token(client, cfg, prompt: list[int], token: int) -> float:
    sampling = get_sampling_params_for_backend(cfg.generator.backend, cfg.generator.sampling_params)
    sampling.update(
        {
            "temperature": 0.0,
            "ignore_eos": True,
            "max_tokens": 1,
            "prompt_logprobs": 1,
        }
    )
    result = await client.generate(InferenceEngineInput(prompt_token_ids=[prompt + [token]], sampling_params=sampling))
    return float(result["prompt_logprobs"][0][-1][token])


def _readback(learner, client) -> dict[str, int]:
    state = {name: torch.from_numpy(np.asarray(value).copy()) for name, value in learner.model.to_state_dict().items()}
    expert_owners = assert_engine_weights(
        client,
        READBACK_NAMES,
        state,
        [ROUTER_BIAS_NAME],
        {EXPERT_ZERO_NAME: 0, EXPERT_FOUR_NAME: 4},
    )
    assert expert_owners[EXPERT_ZERO_NAME] != expert_owners[EXPERT_FOUR_NAME], expert_owners
    return expert_owners


def _learner_state_arrays(learner) -> dict[str, np.ndarray]:
    import jax

    arrays = {f"parameter::{name}": np.asarray(value) for name, value in learner.model.to_state_dict().items()}
    arrays["training_key"] = np.asarray(jax.random.key_data(learner._trainer_state.training_key))
    arrays["optimizer_step"] = np.asarray(learner._trainer_state.step)
    for key_path, value in jax.tree_util.tree_flatten_with_path(learner._trainer_state.opt_state)[0]:
        arrays[f"optimizer::{jax.tree_util.keystr(key_path)}"] = np.asarray(jax.device_get(value))
    return arrays


def _save_expected_learner_state(learner, path: str) -> None:
    arrays = _learner_state_arrays(learner)
    np.savez(path, **arrays)


def _compare_expected_learner_state(learner, path: str) -> dict[str, object]:
    return _compare_state_arrays(_learner_state_arrays(learner), path)


def _compare_state_arrays(arrays: dict[str, np.ndarray], path: str) -> dict[str, object]:
    expected = np.load(path)
    max_deviation = 0.0
    max_deviation_name = None
    category_max_deviation = {"parameter": 0.0, "optimizer": 0.0, "metadata": 0.0}
    category_max_deviation_name: dict[str, str | None] = dict.fromkeys(category_max_deviation)
    assert set(arrays) == set(expected.files)
    exact = True
    for name, value in arrays.items():
        reference = expected[name]
        if not np.array_equal(value, reference):
            exact = False
            deviation = float(np.max(np.abs(value.astype(np.float64) - reference.astype(np.float64))))
            if deviation > max_deviation:
                max_deviation = deviation
                max_deviation_name = name
            category = name.partition("::")[0] if "::" in name else "metadata"
            if deviation > category_max_deviation[category]:
                category_max_deviation[category] = deviation
                category_max_deviation_name[category] = name
    return {
        "exact": exact,
        "max_abs_diff": max_deviation,
        "max_abs_diff_name": max_deviation_name,
        "category_max_abs_diff": category_max_deviation,
        "category_max_abs_diff_name": category_max_deviation_name,
    }


def _data_sharding_evidence(learner) -> dict[str, object]:
    import jax

    mesh = learner._trainer_config.device_mesh
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data", None))
    probe = jax.device_put(np.zeros((learner.runtime.train_batch_size, 8), dtype=np.int32), sharding)
    local_shapes = [list(shard.data.shape) for shard in probe.addressable_shards]
    parameter = learner.model.to_state_dict()[QUERY_NAME]
    parameter_local_shapes = [list(shard.data.shape) for shard in parameter.addressable_shards]
    expected_parameter_spec = jax.sharding.PartitionSpec("model", "data")
    assert int(mesh.shape["data"]) == LEARNER_GPUS
    assert len(local_shapes) == LEARNER_GPUS
    assert {tuple(shape) for shape in local_shapes} == {(learner.runtime.train_batch_size // LEARNER_GPUS, 8)}
    assert parameter.sharding.spec == expected_parameter_spec
    assert len(parameter_local_shapes) == LEARNER_GPUS
    assert {tuple(shape) for shape in parameter_local_shapes} == {(128, 32)}
    return {
        "mesh_data_axis_size": int(mesh.shape["data"]),
        "batch_addressable_shards": len(local_shapes),
        "batch_local_shapes": local_shapes,
        "query_parameter_global_shape": list(parameter.shape),
        "query_parameter_partition_spec": str(parameter.sharding.spec),
        "query_parameter_addressable_shards": len(parameter_local_shapes),
        "query_parameter_local_shapes": parameter_local_shapes,
        "per_device_train_microbatch": learner.runtime.micro_train_batch_size_per_gpu,
    }


@ray.remote(num_gpus=LEARNER_GPUS, num_cpus=4, max_calls=1, max_retries=0)
def _phase_one(
    model_path: str,
    run_path: str,
    checkpoint_state_path: str,
    expected_path: str,
) -> dict:
    import jax
    from skyrl_train.learners.levanter_config import LevanterSnowballRuntimeConfig
    from skyrl_train.learners.levanter_snowball import LevanterSnowballLearner

    run_dir = Path(run_path)
    cfg = _config(model_path, run_dir)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
    client = grug_engine_client(cfg, model_path)
    learner = LevanterSnowballLearner(LevanterSnowballRuntimeConfig.from_msrl(cfg))
    learner.connect_inference_engine(client)
    trainer = _make_trainer(cfg, client, tokenizer, learner)
    try:
        asyncio.run(trainer.train())
        assert trainer.shutdown_requested
        assert jax.device_count() == LEARNER_GPUS
        assert learner.state.policy_version == learner.state.update_count == 2
        assert learner.state.installed_policy_version == 2
        assert learner.state.publication_status.value == "installed"
        runner = trainer.trajectory_runner
        assert len(runner.rollouts) == 2
        first_rollout, second_rollout = runner.rollouts
        for rollout in runner.rollouts:
            assert rollout["stop_reasons"] == ["length"] * 4
            assert all(math.isfinite(value) for row in rollout["rollout_logprobs"] for value in row)
        update_one, update_two = trainer.evidence["updates"]
        expert_owners = _readback(learner, client)
        _save_expected_learner_state(learner, expected_path)

        checkpoint_path = Path(cfg.trainer.ckpt_path) / "global_step_1"
        assert (checkpoint_path / POLICY_CHECKPOINT_SUBDIRECTORY / "metadata.json").is_file()
        assert (checkpoint_path / CHECKPOINT_COMPLETE_FILENAME).read_text() == "1"
        assert trainer.step_one_state is not None
        np.savez(checkpoint_state_path, **trainer.step_one_state)
        publication_probe_scores = trainer.evidence["publication_probe_scores"]
        assert len(publication_probe_scores) == 3
        assert len(trainer.evidence["publication_expert_owners"]) == 3

        assert update_one["parameter_probe_delta_l2"] > 0
        assert update_two["parameter_probe_delta_l2"] > 0
        assert update_one["query_delta_beyond_first_step_weight_decay_l2"] > 1e-8
        assert update_one["router_bias_max_delta"] == update_two["router_bias_max_delta"] == 0
        for update in (update_one, update_two):
            assert update["preupdate_logprob_max_abs_diff"] == 0.0
            assert update["preupdate_logprob_mean_abs_diff"] == 0.0
            assert update["ppo_ratio_min"] == 1.0
            assert update["ppo_ratio_mean"] == 1.0
            assert update["ppo_ratio_max"] == 1.0
            assert update["ppo_clip_ratio"] == 0.0
        assert all(math.isfinite(update[key]) for update in (update_one, update_two) for key in ("final_loss",))
        assert abs(publication_probe_scores[2] - publication_probe_scores[0]) > 1e-7
        return {
            "checkpoint_path": str(checkpoint_path),
            "publication_probe_scores": publication_probe_scores,
            "stop_reasons": [first_rollout["stop_reasons"], second_rollout["stop_reasons"]],
            "timings": trainer.evidence,
            "updates": [update_one, update_two],
            "expert_owners": expert_owners,
            "first_step_uids": runner.requested_uids[0],
            "expected_resumed_uids": runner.requested_uids[1],
            "expected_resumed_response_ids": second_rollout["response_ids"],
            "sharding": _data_sharding_evidence(learner),
            "geometry": {
                "learner_gpus": LEARNER_GPUS,
                "inference_gpus": 2,
                "prompt_batch": 2,
                "generated_trajectory_batch": 4,
                "microbatch_per_gpu": 1,
                "prompt_tokens_per_iteration": 24,
                "response_tokens_per_iteration": 16,
            },
        }
    finally:
        asyncio.run(trainer.release())


@ray.remote(num_gpus=LEARNER_GPUS, num_cpus=4, max_calls=1, max_retries=0)
def _phase_two(
    model_path: str,
    run_path: str,
    checkpoint_path: str,
    checkpoint_state_path: str,
    expected_path: str,
    expected_uids: list[str],
    expected_response_ids: list[list[int]],
    expected_publication_probe_scores: list[float],
) -> dict:
    from skyrl_train.learners.levanter_config import LevanterSnowballRuntimeConfig
    from skyrl_train.learners.levanter_snowball import LevanterSnowballLearner

    cfg = _config(model_path, Path(run_path))
    cfg.trainer.resume_mode = "from_path"
    cfg.trainer.resume_path = checkpoint_path
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
    client = grug_engine_client(cfg, model_path)
    learner = LevanterSnowballLearner(LevanterSnowballRuntimeConfig.from_msrl(cfg))
    learner.connect_inference_engine(client)
    trainer = _make_trainer(cfg, client, tokenizer, learner)
    try:
        asyncio.run(trainer.train())
        assert trainer.shutdown_requested
        assert trainer.restored_step == trainer.restored_policy_version == 1
        assert trainer.restored_state is not None
        restore_comparison = _compare_state_arrays(trainer.restored_state, checkpoint_state_path)
        assert restore_comparison["exact"], restore_comparison
        runner = trainer.trajectory_runner
        assert runner.requested_uids == [expected_uids]
        assert runner.rollouts[0]["response_ids"] == expected_response_ids
        publication_probe_scores = trainer.evidence["publication_probe_scores"]
        assert len(publication_probe_scores) == 2
        assert len(trainer.evidence["publication_expert_owners"]) == 2
        assert learner.state.installed_policy_version == 2
        assert learner.state.publication_status.value == "installed"
        assert publication_probe_scores[0] == pytest.approx(expected_publication_probe_scores[1], abs=1e-5)
        assert publication_probe_scores[1] == pytest.approx(expected_publication_probe_scores[2], abs=1e-4)
        update = trainer.evidence["updates"][0]
        next_update_comparison = _compare_expected_learner_state(learner, expected_path)
        expert_owners = _readback(learner, client)

        prompt = runner.rollouts[0]["prompt_token_ids"][0]
        rollout = asyncio.run(_generate(client, cfg, [prompt, prompt]))
        token = rollout["response_ids"][0][0]
        score = asyncio.run(_score_token(client, cfg, prompt, token))
        assert rollout["stop_reasons"] == ["length", "length"]
        assert update["preupdate_logprob_max_abs_diff"] == 0.0
        assert update["preupdate_logprob_mean_abs_diff"] == 0.0
        assert update["ppo_ratio_min"] == 1.0
        assert update["ppo_ratio_mean"] == 1.0
        assert update["ppo_ratio_max"] == 1.0
        assert update["ppo_clip_ratio"] == 0.0
        replay_differences = next_update_comparison["category_max_abs_diff"]
        # A fresh XLA process may choose a different GPU reduction order. The
        # checkpoint itself remains byte-exact; only the replayed update uses
        # an absolute tolerance for floating-point model and optimizer arrays.
        assert replay_differences["metadata"] == 0.0, next_update_comparison
        assert replay_differences["parameter"] <= GPU_REPLAY_ATOL, next_update_comparison
        assert replay_differences["optimizer"] <= GPU_REPLAY_ATOL, next_update_comparison
        return {
            "restored_global_step": trainer.restored_step,
            "completed_global_step": trainer.global_step,
            "restored_policy_version": trainer.restored_policy_version,
            "completed_policy_version": learner.state.policy_version,
            "restore_comparison": restore_comparison,
            "next_update_comparison": next_update_comparison,
            "timings": trainer.evidence,
            "resumed_uids": runner.requested_uids[0],
            "resumed_response_ids": runner.rollouts[0]["response_ids"],
            "publication_probe_scores": publication_probe_scores,
            "score": score,
            "update": update,
            "expert_owners": expert_owners,
        }
    finally:
        asyncio.run(trainer.release())


def test_four_h100_msrl_update_publication_generation_and_fresh_resume(tmp_path):
    require_hoppers(ACTIVE_GPUS)
    # Pytest adds skyrl-train to the driver path. Ray workers import this test
    # module in fresh processes, so give them the same source root explicitly.
    source_root = str(Path(__file__).parents[2])
    os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, (source_root, os.environ.get("PYTHONPATH"))))
    model_path = tmp_path / "tiny-grug"
    _write_tiny_checkpoint(model_path)
    cfg = _config(str(model_path), tmp_path / "phase-one")
    initialize_ray(cfg)
    assert int(ray.cluster_resources().get("GPU", 0)) >= ACTIVE_GPUS
    checkpoint_state_path = tmp_path / "expected-checkpoint-state.npz"
    expected_path = tmp_path / "expected-next-state.npz"
    evidence_path = tmp_path / "levanter-snowball-gpu-evidence.json"
    try:
        phase_one = ray.get(
            _phase_one.remote(
                str(model_path),
                str(tmp_path / "phase-one"),
                str(checkpoint_state_path),
                str(expected_path),
            ),
            timeout=900,
        )
        # A non-colocated inference engine owns a placement group for its GPU.
        # Restarting the local Ray runtime proves a full process-level resume and
        # drops that placement group before the fresh engine is constructed.
        ray.shutdown()
        initialize_ray(cfg)
        assert int(ray.cluster_resources().get("GPU", 0)) >= ACTIVE_GPUS
        phase_two = ray.get(
            _phase_two.remote(
                str(model_path),
                str(tmp_path / "phase-two"),
                phase_one["checkpoint_path"],
                str(checkpoint_state_path),
                str(expected_path),
                phase_one["expected_resumed_uids"],
                phase_one["expected_resumed_response_ids"],
                phase_one["publication_probe_scores"],
            ),
            timeout=900,
        )
        evidence = {"phase_one": phase_one, "phase_two": phase_two}
        evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True))
        print(evidence_path.read_text())
    finally:
        ray.shutdown()
