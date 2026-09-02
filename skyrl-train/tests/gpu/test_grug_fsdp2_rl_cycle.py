"""Grug FSDP2 capstones on Hopper.

The four-H100 test covers disaggregated EP1 training and serving. The six-H100
test adds FSDP2 ``(fsdp=2, ep=2)`` grouped-expert workers. Both run rollout,
train, checkpoint restore, exact serving readback, and a second rollout.
"""

from __future__ import annotations

import asyncio
import importlib.util
import math
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import pytest
import ray
import torch
from transformers import AutoConfig, AutoTokenizer

from skyrl_train.inference_engines.base import InferenceEngineInput
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_train.models.grug_moe import (
    GRUG_MOE_MODEL_TYPE,
    GRUG_ROUTER_BIAS_SUFFIX,
    GrugMoeConfig,
    GrugMoeForCausalLM,
)
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.utils import initialize_ray
from tests.gpu.grug_gpu_gates import require_hoppers
from tests.gpu.grug_serving import (
    LM_HEAD_NAME,
    ROUTER_NAME,
    STACKED_EXPERT_NAME,
    assert_engine_weights,
    grug_engine_client,
    rank0_validation_snapshot,
    rollout_training_batch,
)
from tests.gpu.utils import get_test_actor_config, init_worker_with_type


EP1_POLICY_WORLD_SIZE = 2
MIXED_EP_POLICY_WORLD_SIZE = 4
ROLLOUT_WORLD_SIZE = 2
EP1_DISAGGREGATED_NUM_GPUS = EP1_POLICY_WORLD_SIZE + ROLLOUT_WORLD_SIZE
MIXED_EP_DISAGGREGATED_NUM_GPUS = MIXED_EP_POLICY_WORLD_SIZE + ROLLOUT_WORLD_SIZE
TOKENIZER = "Qwen/Qwen2.5-0.5B-Instruct"
SERVING_EXPERT_INDEX_BY_NAME = {
    "model.layers.0.mlp.experts.0.gate_proj.weight": 0,
    "model.layers.0.mlp.experts.4.gate_proj.weight": 4,
}


# The fused policy path needs the optional compiled FlashAttention package, which the
# Megatron runtime closure does not ship.
requires_flash_attention = pytest.mark.skipif(
    importlib.util.find_spec("flash_attn") is None, reason="flash_attn is not installed"
)


class _PolicyAttentionBackend(StrEnum):
    EAGER = "eager"
    FLASH_ATTENTION = "flash_attention_2"


@dataclass(frozen=True)
class _RepresentativeNames:
    bias_names: list[str]
    parameter_names: list[str]
    sync_parameter_names: list[str]
    wire_bf16_sentinel_name: str


@dataclass(frozen=True)
class _TrainingSnapshot(_RepresentativeNames):
    weights: dict[str, torch.Tensor]


def _write_tiny_checkpoint(path) -> torch.Tensor:
    """Write the tiny fixture and return its pre-sharding stacked experts."""
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
    model = GrugMoeForCausalLM(config)
    stacked_experts = model.state_dict()[STACKED_EXPERT_NAME].detach().cpu().clone()
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    return stacked_experts


def _config(
    model_path: str,
    *,
    policy_attention_backend: _PolicyAttentionBackend,
    policy_world_size: int = EP1_POLICY_WORLD_SIZE,
    expert_model_parallel_size: int = 1,
):
    assert policy_world_size % expert_model_parallel_size == 0
    cfg = get_test_actor_config()
    cfg.trainer.policy.model.path = model_path
    cfg.trainer.critic.model.path = ""
    cfg.trainer.strategy = "fsdp2"
    cfg.trainer.flash_attn = policy_attention_backend is _PolicyAttentionBackend.FLASH_ATTENTION
    cfg.trainer.attn_backend = "auto"
    cfg.trainer.gradient_checkpointing = True
    cfg.trainer.gradient_checkpointing_use_reentrant = False
    cfg.trainer.use_sample_packing = False
    cfg.trainer.train_batch_size = policy_world_size
    cfg.trainer.policy_mini_batch_size = policy_world_size
    cfg.trainer.micro_train_batch_size_per_gpu = 1
    cfg.trainer.update_epochs_per_batch = 1
    cfg.trainer.algorithm.use_kl_loss = False
    cfg.trainer.algorithm.use_entropy_loss = False
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.placement.policy_num_nodes = 1
    cfg.trainer.placement.policy_num_gpus_per_node = policy_world_size
    cfg.trainer.policy.fsdp_config.cpu_offload = True
    cfg.trainer.policy.fsdp_config.fsdp_size = policy_world_size // expert_model_parallel_size
    cfg.trainer.policy.fsdp_config.expert_model_parallel_size = expert_model_parallel_size
    cfg.trainer.policy.fsdp_config.context_parallel_size = 1
    cfg.trainer.policy.fsdp_config.moe_router_replay = False
    cfg.trainer.policy.fsdp_config.moe_grouped_gemm = False
    cfg.trainer.policy.grug_query_bias_update_mode = "loss_free"
    cfg.trainer.policy.grug_query_bias_update_rate = 0.001
    cfg.trainer.policy.fsdp_config.use_grouped_mm = expert_model_parallel_size > 1
    cfg.trainer.policy.optimizer_config.max_grad_norm = 0.0
    if expert_model_parallel_size > 1:
        cfg.trainer.policy.optimizer_config.optimizer = "AdamW"
        cfg.trainer.policy.optimizer_config.lr = 4.0e-3
        cfg.trainer.policy.optimizer_config.adam_betas = [0.9, 0.95]
        cfg.trainer.policy.optimizer_config.weight_decay = 1.0e-2
        cfg.trainer.policy.optimizer_config.optimizer_kwargs = {}
    else:
        cfg.trainer.policy.optimizer_config.optimizer = "MuonH"
        cfg.trainer.policy.optimizer_config.lr = 3.0e-2
        cfg.trainer.policy.optimizer_config.weight_decay = 0.0
        cfg.trainer.policy.optimizer_config.adam_betas = [0.9, 0.95]
        cfg.trainer.policy.optimizer_config.optimizer_kwargs = {
            "adam_lr": 4.0e-3,
            "momentum": 0.95,
            "nesterov": True,
            "backend_steps": 5,
            "epsilon": 1.0e-8,
            "muon_epsilon": 1.0e-8,
        }
    cfg.generator.backend = "vllm"
    cfg.generator.async_engine = True
    cfg.generator.weight_sync_backend = "nccl"
    cfg.generator.inference_engine_tensor_parallel_size = 1
    cfg.generator.inference_engine_data_parallel_size = ROLLOUT_WORLD_SIZE
    cfg.generator.inference_engine_expert_parallel_size = ROLLOUT_WORLD_SIZE
    cfg.generator.num_inference_engines = 1
    cfg.generator.n_samples_per_prompt = 1
    cfg.generator.gpu_memory_utilization = 0.35
    cfg.generator.sampling_params.temperature = 1.0
    cfg.generator.sampling_params.top_p = 1.0
    cfg.generator.sampling_params.top_k = -1
    cfg.generator.sampling_params.max_generate_length = 4
    return cfg


def _training_batch(prompts: list[list[int]], rollout) -> TrainingInputBatch:
    assert all(len(ids) == 4 for ids in rollout["response_ids"])
    return rollout_training_batch(prompts, rollout)


def _validation_snapshots(policy, names=()):
    return ray.get(policy.async_run_ray_method("pass_through", "grug_validation_snapshot", names))


def _assert_policy_attention_backend(policy, expected: str) -> None:
    snapshots = _validation_snapshots(policy)
    assert {snapshot.attention_backend for snapshot in snapshots} == {expected}


def _representative_names(model_path: str) -> _RepresentativeNames:
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=False, local_files_only=True)
    assert config.model_type == GRUG_MOE_MODEL_TYPE
    biases = [f"model.layers.{idx}{GRUG_ROUTER_BIAS_SUFFIX}" for idx in range(config.num_hidden_layers)]
    parameter_names = [
        "model.layers.0.self_attn.q_proj.weight",
        STACKED_EXPERT_NAME,
        LM_HEAD_NAME,
        ROUTER_NAME,
    ]
    return _RepresentativeNames(
        bias_names=biases,
        parameter_names=parameter_names,
        sync_parameter_names=[
            parameter_names[0],
            *SERVING_EXPERT_INDEX_BY_NAME,
            parameter_names[2],
            parameter_names[3],
        ],
        wire_bf16_sentinel_name="model.layers.0.input_layernorm.weight",
    )


def _train_andrank0_validation_snapshot(policy, batch: TrainingInputBatch, model_path: str) -> _TrainingSnapshot:
    """Run one policy step and return its post-update FP32 training snapshot."""
    names = _representative_names(model_path)
    before = rank0_validation_snapshot(policy, names.parameter_names)
    train_output = ray.get(policy.async_run_ray_method("pass_through", "ppo_train", batch))[0]
    status = train_output.metadata["train_status"]
    assert math.isfinite(status["policy_loss"])
    # Marin's recipe disables clipping, so the worker deliberately omits this
    # clipping-only metric.
    assert "raw_grad_norm" not in status
    assert status["optimizer_step_succeeded"] == 1.0
    snapshot_names = [*names.parameter_names, names.wire_bf16_sentinel_name, *names.bias_names]
    weights = rank0_validation_snapshot(policy, snapshot_names)
    for name in names.parameter_names:
        assert not torch.equal(weights[name], before[name]), f"{name} did not update"
        assert (weights[name] - before[name]).dtype == torch.float32
    assert all(torch.count_nonzero(weights[name]).item() > 0 for name in names.bias_names)
    return _TrainingSnapshot(
        bias_names=names.bias_names,
        parameter_names=names.parameter_names,
        sync_parameter_names=names.sync_parameter_names,
        wire_bf16_sentinel_name=names.wire_bf16_sentinel_name,
        weights=weights,
    )


def _assert_checkpoint_resume_next_step(
    policy,
    mutation_batch: TrainingInputBatch,
    checkpoint_path: str,
    model_path: str,
    names: list[str],
    expected_weights: dict[str, torch.Tensor],
) -> _TrainingSnapshot:
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    ray.get(
        policy.async_run_ray_method("pass_through", "save_checkpoint", ckpt_dir=checkpoint_path, tokenizer=tokenizer)
    )
    mutation_batch.metadata["global_step"] = 1
    expected_next_step = _train_andrank0_validation_snapshot(policy, mutation_batch, model_path)
    ray.get(policy.async_run_ray_method("pass_through", "load_checkpoint", ckpt_dir=checkpoint_path))
    resumed = rank0_validation_snapshot(policy, names)
    for name in names:
        torch.testing.assert_close(resumed[name], expected_weights[name], rtol=0, atol=0)
    resumed_next_step = _train_andrank0_validation_snapshot(policy, mutation_batch, model_path)
    for name in names:
        torch.testing.assert_close(
            resumed_next_step.weights[name],
            expected_next_step.weights[name],
            rtol=0,
            atol=0,
        )
    return resumed_next_step


def _run_full_cycle(
    model_path: str,
    tmp_path: Path,
    *,
    policy_attention_backend: _PolicyAttentionBackend,
    policy_world_size: int = EP1_POLICY_WORLD_SIZE,
    expert_model_parallel_size: int = 1,
    unsharded_stacked_experts: torch.Tensor | None = None,
) -> None:
    cfg = _config(
        model_path,
        policy_attention_backend=policy_attention_backend,
        policy_world_size=policy_world_size,
        expert_model_parallel_size=expert_model_parallel_size,
    )
    initialize_ray(cfg)
    required_gpus = policy_world_size + ROLLOUT_WORLD_SIZE
    available_gpus = int(ray.cluster_resources().get("GPU", 0))
    if available_gpus < required_gpus:
        pytest.skip(f"topology requires {required_gpus} Ray GPUs, found {available_gpus}")
    client = grug_engine_client(cfg, model_path)
    try:
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
        policy = init_worker_with_type(
            "policy",
            shared_pg=None,
            colocate_all=False,
            num_gpus_per_node=policy_world_size,
            num_nodes=1,
            cfg=cfg,
        )
        _assert_policy_attention_backend(policy, policy_attention_backend.value)
        if expert_model_parallel_size > 1:
            geometry = ray.get(policy.async_run_ray_method("pass_through", "diag_ep_geometry"))
            assert all(item["mesh_dim_names"] == ["ddp", "fsdp", "ep"] for item in geometry)
            assert all(tuple(item["mesh_shape"]) == (1, 2, 2) for item in geometry)
            assert {item["ep_coord"] for item in geometry} == {0, 1}
            assert unsharded_stacked_experts is not None
            gathered_stacked_experts = rank0_validation_snapshot(policy, [STACKED_EXPERT_NAME])[STACKED_EXPERT_NAME]
            stored_stacked_experts = unsharded_stacked_experts.to(torch.bfloat16).to(torch.float32)
            torch.testing.assert_close(gathered_stacked_experts, stored_stacked_experts, rtol=0, atol=0)
        training = _train_andrank0_validation_snapshot(policy, _training_batch(prompts, first_rollout), model_path)
        checkpoint_names = [
            *training.bias_names,
            *training.parameter_names,
            training.wire_bf16_sentinel_name,
        ]

        checkpoint_path = str(tmp_path / "resume-checkpoint")
        training = _assert_checkpoint_resume_next_step(
            policy,
            _training_batch(prompts, first_rollout),
            checkpoint_path,
            model_path,
            checkpoint_names,
            training.weights,
        )
        sync_names = [
            *training.bias_names,
            *training.sync_parameter_names,
            training.wire_bf16_sentinel_name,
        ]

        ray.get(policy.async_run_ray_method("pass_through", "init_weight_sync_state", client))
        ray.get(policy.async_run_ray_method("pass_through", "broadcast_to_inference_engines", client))

        serving_owners = assert_engine_weights(
            client, sync_names, training.weights, training.bias_names, SERVING_EXPERT_INDEX_BY_NAME
        )
        if expert_model_parallel_size > 1:
            experts_per_trainer_owner = (
                AutoConfig.from_pretrained(model_path, trust_remote_code=False, local_files_only=True).num_local_experts
                // expert_model_parallel_size
            )
            trainer_owners = {
                name: expert_index // experts_per_trainer_owner
                for name, expert_index in SERVING_EXPERT_INDEX_BY_NAME.items()
            }
            assert len(set(trainer_owners.values())) > 1, trainer_owners
            assert len(set(serving_owners.values())) > 1, serving_owners
            assert serving_owners == trainer_owners

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
def test_grug_one_gpu_vllm_generation(tmp_path):
    """Load the training checkpoint with Marin vLLM and generate real tokens."""

    require_hoppers(1)

    from vllm import LLM, SamplingParams  # noqa: PLC0415
    from vllm.inputs import TokensPrompt  # noqa: PLC0415

    model_path = tmp_path / "model"
    model_path.mkdir()
    _write_tiny_checkpoint(model_path)
    engine = LLM(
        model=str(model_path),
        dtype="bfloat16",
        enforce_eager=True,
        gpu_memory_utilization=0.35,
        max_model_len=128,
        max_num_batched_tokens=128,
        max_num_seqs=1,
        trust_remote_code=False,
        disable_log_stats=True,
    )
    outputs = engine.generate(
        prompts=[TokensPrompt(prompt_token_ids=[1, 17, 29, 5, 11, 3])],
        sampling_params=SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True),
    )

    assert len(outputs) == 1
    assert len(outputs[0].outputs) == 1
    assert len(outputs[0].outputs[0].token_ids) == 4


@pytest.mark.vllm
@pytest.mark.parametrize(
    "policy_attention_backend",
    [
        pytest.param(_PolicyAttentionBackend.FLASH_ATTENTION, id="fused", marks=requires_flash_attention),
        pytest.param(_PolicyAttentionBackend.EAGER, id="eager"),
    ],
)
def test_grug_four_gpu_disaggregated_rollout_train_broadcast_rollout(
    tmp_path,
    policy_attention_backend: _PolicyAttentionBackend,
):
    """Exercise mixed-dtype Grug sync with trainer and rollout on disjoint GPUs."""
    require_hoppers(EP1_DISAGGREGATED_NUM_GPUS)

    _write_tiny_checkpoint(tmp_path)
    _run_full_cycle(
        str(tmp_path),
        tmp_path,
        policy_attention_backend=policy_attention_backend,
    )


@requires_flash_attention
@pytest.mark.vllm
def test_grug_six_h100_mixed_ep_disaggregated_rollout_train_broadcast_rollout(tmp_path):
    """Exercise EP-owned experts and exact sync on disjoint trainer/rollout GPUs."""
    require_hoppers(MIXED_EP_DISAGGREGATED_NUM_GPUS)

    unsharded_stacked_experts = _write_tiny_checkpoint(tmp_path)
    _run_full_cycle(
        str(tmp_path),
        tmp_path,
        policy_attention_backend=_PolicyAttentionBackend.FLASH_ATTENTION,
        policy_world_size=MIXED_EP_POLICY_WORLD_SIZE,
        expert_model_parallel_size=2,
        unsharded_stacked_experts=unsharded_stacked_experts,
    )
