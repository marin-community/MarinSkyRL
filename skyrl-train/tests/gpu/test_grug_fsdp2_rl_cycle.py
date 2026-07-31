"""Grug FSDP2 capstones on Hopper.

The two-H100 test covers colocated CUDA-IPC weight sync. The four-H100 test
puts the same vLLM TP1/DP2/EP2 and FSDP2 EP1 workers on disjoint GPUs and
covers the production NCCL-broadcast path. Both run rollout -> RL update +
query bias -> exact weight readback -> a second rollout.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from pathlib import Path

import pytest
import ray
import torch
from ray.util.placement_group import placement_group
from transformers import AutoConfig, AutoTokenizer

from skyrl_train.inference_engines.base import InferenceEngineInput
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.ray_wrapped_inference_engine import create_ray_wrapped_inference_engines
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_train.models.grug_moe import (
    GRUG_MOE_MODEL_TYPE,
    GRUG_ROUTER_BIAS_SUFFIX,
    GrugMoeConfig,
    GrugMoeForCausalLM,
)
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.utils import get_ray_pg_ready_with_timeout, initialize_ray
from tests.gpu.grug_gpu_gates import require_hoppers
from tests.gpu.utils import get_test_actor_config, init_worker_with_type


POLICY_WORLD_SIZE = 2
COLOCATED_NUM_GPUS = POLICY_WORLD_SIZE
DISAGGREGATED_NUM_GPUS = POLICY_WORLD_SIZE * 2
TOKENIZER = "Qwen/Qwen2.5-0.5B-Instruct"


@dataclass(frozen=True)
class _TrainingSnapshot:
    bias_names: list[str]
    representative_name: str
    bf16_sentinel_name: str
    weights: dict[str, torch.Tensor]


@dataclass(frozen=True)
class _RepresentativeNames:
    bias_names: list[str]
    trainable_name: str
    bf16_sentinel_name: str


def _write_tiny_checkpoint(path) -> None:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    config = GrugMoeConfig(
        vocab_size=len(tokenizer),
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
    tokenizer.save_pretrained(path)


def _config(
    model_path: str,
    *,
    colocate_all: bool = True,
):
    cfg = get_test_actor_config()
    cfg.trainer.policy.model.path = model_path
    cfg.trainer.critic.model.path = ""
    cfg.trainer.strategy = "fsdp2"
    cfg.trainer.flash_attn = True
    cfg.trainer.attn_backend = "auto"
    cfg.trainer.gradient_checkpointing = True
    cfg.trainer.gradient_checkpointing_use_reentrant = False
    cfg.trainer.use_sample_packing = False
    cfg.trainer.train_batch_size = POLICY_WORLD_SIZE
    cfg.trainer.policy_mini_batch_size = POLICY_WORLD_SIZE
    cfg.trainer.micro_train_batch_size_per_gpu = 1
    cfg.trainer.update_epochs_per_batch = 1
    cfg.trainer.algorithm.use_kl_loss = False
    cfg.trainer.algorithm.use_entropy_loss = False
    cfg.trainer.placement.colocate_all = colocate_all
    cfg.trainer.placement.policy_num_nodes = 1
    cfg.trainer.placement.policy_num_gpus_per_node = POLICY_WORLD_SIZE
    cfg.trainer.policy.fsdp_config.cpu_offload = True
    cfg.trainer.policy.fsdp_config.fsdp_size = POLICY_WORLD_SIZE
    cfg.trainer.policy.fsdp_config.expert_model_parallel_size = 1
    cfg.trainer.policy.fsdp_config.context_parallel_size = 1
    cfg.trainer.policy.fsdp_config.moe_router_replay = False
    cfg.trainer.policy.fsdp_config.moe_grouped_gemm = False
    cfg.trainer.policy.fsdp_config.use_grouped_mm = False
    cfg.trainer.policy.optimizer_config.lr = 1.0e-3
    cfg.generator.backend = "vllm"
    cfg.generator.async_engine = True
    cfg.generator.weight_sync_backend = "nccl"
    cfg.generator.inference_engine_tensor_parallel_size = 1
    cfg.generator.inference_engine_data_parallel_size = POLICY_WORLD_SIZE
    cfg.generator.inference_engine_expert_parallel_size = POLICY_WORLD_SIZE
    cfg.generator.num_inference_engines = 1
    cfg.generator.n_samples_per_prompt = 1
    cfg.generator.gpu_memory_utilization = 0.35
    cfg.generator.sampling_params.temperature = 1.0
    cfg.generator.sampling_params.top_p = 1.0
    cfg.generator.sampling_params.top_k = -1
    cfg.generator.sampling_params.max_generate_length = 4
    return cfg


def _make_training_batch(sequences: torch.Tensor, action_log_probs: torch.Tensor) -> TrainingInputBatch:
    advantage_pattern = torch.tensor([[-1.0, -0.25, 0.5, 1.0], [1.0, 0.5, -0.25, -1.0]])
    advantages = advantage_pattern.repeat(math.ceil(sequences.shape[0] / 2), 1)[: sequences.shape[0]]
    ones = torch.ones_like(action_log_probs)
    batch = TrainingInputBatch(
        {
            "sequences": sequences,
            "attention_mask": torch.ones_like(sequences),
            "action_log_probs": action_log_probs,
            "base_action_log_probs": action_log_probs.clone(),
            "rollout_logprobs": action_log_probs.clone(),
            "values": torch.zeros_like(ones),
            "returns": torch.zeros_like(ones),
            "advantages": advantages,
            "loss_mask": ones.long(),
            "response_mask": ones.long(),
        }
    )
    batch.metadata = {"response_length": action_log_probs.shape[1], "global_step": 0}
    return batch


def _training_batch(prompts: list[list[int]], rollout) -> TrainingInputBatch:
    assert rollout["response_logprobs"] is not None
    assert all(len(ids) == 4 for ids in rollout["response_ids"])
    sequences = torch.tensor(
        [prompt + response for prompt, response in zip(prompts, rollout["response_ids"])],
        dtype=torch.long,
    )
    return _make_training_batch(sequences, torch.tensor(rollout["response_logprobs"], dtype=torch.float32))


def _engine_client(cfg, model_path: str, shared_pg) -> InferenceEngineClient:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    engines = create_ray_wrapped_inference_engines(
        num_inference_engines=cfg.generator.num_inference_engines,
        tensor_parallel_size=1,
        data_parallel_size=cfg.generator.inference_engine_data_parallel_size,
        expert_parallel_size=cfg.generator.inference_engine_expert_parallel_size,
        model_dtype="bfloat16",
        pretrain=model_path,
        seed=23,
        vllm_v1_disable_multiproc=True,
        enable_prefix_caching=False,
        enforce_eager=True,
        shared_pg=shared_pg,
        gpu_memory_utilization=cfg.generator.gpu_memory_utilization,
        inference_engine_enable_sleep=cfg.trainer.placement.colocate_all,
        async_engine=True,
        max_num_batched_tokens=128 * cfg.generator.inference_engine_data_parallel_size,
        max_num_seqs=cfg.trainer.train_batch_size,
        tokenizer=tokenizer,
        backend="vllm",
        engine_init_kwargs={"max_model_len": 128},
    )
    return InferenceEngineClient(engines, tokenizer, cfg)


def _snapshot(policy, names=()):
    snapshots = ray.get(policy.async_run_ray_method("pass_through", "grug_validation_snapshot", names))
    rank0 = next(snapshot for snapshot in snapshots if snapshot.rank == 0)
    return rank0.weights


def _assert_policy_attention_backend(policy, expected: str) -> None:
    snapshots = ray.get(policy.async_run_ray_method("pass_through", "grug_validation_snapshot"))
    assert {snapshot.attention_backend for snapshot in snapshots} == {expected}


def _representative_names(model_path: str) -> _RepresentativeNames:
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=False, local_files_only=True)
    assert config.model_type == GRUG_MOE_MODEL_TYPE
    biases = [f"model.layers.{idx}{GRUG_ROUTER_BIAS_SUFFIX}" for idx in range(config.num_hidden_layers)]
    return _RepresentativeNames(
        bias_names=biases,
        trainable_name="model.layers.0.mlp.router.weight",
        bf16_sentinel_name="model.layers.0.input_layernorm.weight",
    )


def _train_and_snapshot(policy, batch: TrainingInputBatch, model_path: str) -> _TrainingSnapshot:
    names = _representative_names(model_path)
    before = _snapshot(policy, [names.trainable_name])
    train_output = ray.get(policy.async_run_ray_method("pass_through", "ppo_train", batch))[0]
    status = train_output.metadata["train_status"]
    assert math.isfinite(status["policy_loss"])
    assert math.isfinite(status["raw_grad_norm"])
    assert status["raw_grad_norm"] > 0
    assert status["optimizer_step_succeeded"] == 1.0
    weights = _snapshot(policy, [names.trainable_name, names.bf16_sentinel_name, *names.bias_names])
    assert not torch.equal(weights[names.trainable_name], before[names.trainable_name])
    assert all(torch.count_nonzero(weights[name]).item() > 0 for name in names.bias_names)
    return _TrainingSnapshot(names.bias_names, names.trainable_name, names.bf16_sentinel_name, weights)


def _assert_checkpoint_restores_biases(
    policy,
    mutation_batch: TrainingInputBatch,
    checkpoint_path: str,
    model_path: str,
    bias_names: list[str],
    expected_weights: dict[str, torch.Tensor],
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    ray.get(
        policy.async_run_ray_method("pass_through", "save_checkpoint", ckpt_dir=checkpoint_path, tokenizer=tokenizer)
    )
    mutation_batch.metadata["global_step"] = 1
    ray.get(policy.async_run_ray_method("pass_through", "ppo_train", mutation_batch))
    ray.get(policy.async_run_ray_method("pass_through", "load_checkpoint", ckpt_dir=checkpoint_path))
    resumed = _snapshot(policy, bias_names)
    for name in bias_names:
        torch.testing.assert_close(resumed[name], expected_weights[name], rtol=0, atol=0)


def _run_full_cycle(
    model_path: str,
    tmp_path: Path,
    *,
    colocate_all: bool = True,
) -> None:
    cfg = _config(model_path, colocate_all=colocate_all)
    initialize_ray(cfg)
    required_gpus = COLOCATED_NUM_GPUS if colocate_all else DISAGGREGATED_NUM_GPUS
    available_gpus = int(ray.cluster_resources().get("GPU", 0))
    if available_gpus < required_gpus:
        pytest.skip(f"topology requires {required_gpus} Ray GPUs, found {available_gpus}")
    pg = None
    if colocate_all:
        pg = placement_group(
            [{"GPU": 1, "CPU": 1}] * COLOCATED_NUM_GPUS,
            strategy="PACK",
        )
        get_ray_pg_ready_with_timeout(pg, timeout=60)
    client = _engine_client(cfg, model_path, pg)
    try:
        if colocate_all:
            asyncio.run(client.wake_up())
        prompt_pattern = [[1, 17, 29, 5, 11, 3], [1, 19, 31, 7, 13, 3]]
        prompts = [prompt_pattern[idx % len(prompt_pattern)] for idx in range(cfg.trainer.train_batch_size)]
        sampling_params = get_sampling_params_for_backend(cfg.generator.backend, cfg.generator.sampling_params)
        sampling_params.update({"temperature": 0.0, "ignore_eos": True, "logprobs": 1})
        first_rollout = asyncio.run(
            client.generate(InferenceEngineInput(prompt_token_ids=prompts, sampling_params=sampling_params))
        )
        first_token = first_rollout["response_ids"][0][0]
        score_params = dict(sampling_params)
        score_params.update({"max_tokens": 1, "prompt_logprobs": 1})
        first_score = asyncio.run(
            client.generate(
                InferenceEngineInput(
                    prompt_token_ids=[prompts[0] + [first_token]],
                    sampling_params=score_params,
                )
            )
        )
        first_logprob = first_score["prompt_logprobs"][0][-1][first_token]
        if colocate_all:
            asyncio.run(client.sleep())

        policy = init_worker_with_type(
            "policy",
            shared_pg=pg,
            colocate_all=colocate_all,
            num_gpus_per_node=POLICY_WORLD_SIZE,
            num_nodes=1,
            cfg=cfg,
        )
        _assert_policy_attention_backend(policy, "flash_attention_2")
        training = _train_and_snapshot(policy, _training_batch(prompts, first_rollout), model_path)
        sync_names = [*training.bias_names, training.representative_name, training.bf16_sentinel_name]

        checkpoint_path = str(tmp_path / "resume-checkpoint")
        _assert_checkpoint_restores_biases(
            policy,
            _training_batch(prompts, first_rollout),
            checkpoint_path,
            model_path,
            training.bias_names,
            training.weights,
        )

        if colocate_all:
            policy.offload_to_cpu(offload_optimizer=False, offload_model=True)
        ray.get(policy.async_run_ray_method("pass_through", "init_weight_sync_state", client))
        if colocate_all:
            asyncio.run(client.wake_up(tags=["weights"]))
        ray.get(policy.async_run_ray_method("pass_through", "broadcast_to_inference_engines", client))

        for engine in client.engines:
            per_rank = ray.get(engine.inference_engine_actor.read_engine_weights.remote(sync_names, False))
            if isinstance(per_rank, dict):
                per_rank = [per_rank]
            for rank_values in per_rank:
                for name in sync_names:
                    entry = rank_values[name]
                    assert entry["found"], (name, entry)
                    expected_dtype = "bfloat16" if name == training.bf16_sentinel_name else "float32"
                    assert entry["dtype"] == expected_dtype, (name, entry["dtype"])
                    torch.testing.assert_close(entry["tensor"], training.weights[name], rtol=0, atol=0)

        if colocate_all:
            policy.offload_to_cpu(offload_optimizer=False, offload_model=True)
            asyncio.run(client.wake_up(tags=["kv_cache"]))
        asyncio.run(client.reset_prefix_cache())
        second_score = asyncio.run(
            client.generate(
                InferenceEngineInput(
                    prompt_token_ids=[prompts[0] + [first_token]],
                    sampling_params=score_params,
                )
            )
        )
        second_logprob = second_score["prompt_logprobs"][0][-1][first_token]
        logprob_delta = second_logprob - first_logprob
        assert abs(logprob_delta) > 1e-7
    finally:
        ray.shutdown()


@pytest.mark.vllm
def test_grug_two_h100_rollout_train_sync_rollout(tmp_path):
    require_hoppers(COLOCATED_NUM_GPUS)

    _write_tiny_checkpoint(tmp_path)
    _run_full_cycle(str(tmp_path), tmp_path)


@pytest.mark.vllm
def test_grug_four_h100_disaggregated_rollout_train_broadcast_rollout(tmp_path):
    """Exercise mixed-dtype Grug sync with trainer and rollout on disjoint GPUs."""

    require_hoppers(DISAGGREGATED_NUM_GPUS)

    _write_tiny_checkpoint(tmp_path)
    _run_full_cycle(str(tmp_path), tmp_path, colocate_all=False)
