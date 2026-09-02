"""Grug on the Megatron trainer, pipeline-parallel first.

The parity tests compare Megatron policy log-probabilities against the eager
HF reference at PP1, PP2, and PP2+EP2. The training test takes one PPO step
at PP2, checks that exported weights moved while the frozen router bias did
not, and checks that the exported HF checkpoint reproduces the post-update
Megatron log-probabilities. The four-H100 test adds disaggregated vLLM
serving: rollout, PP2 update, weight broadcast, serving readback, rollout.
"""

from __future__ import annotations

import asyncio
import math
from pathlib import Path

import pytest
import ray
import torch
from transformers import AutoTokenizer

from skyrl_train.distributed.dispatch import concatenate_outputs_after_mesh_dispatch
from skyrl_train.inference_engines.base import InferenceEngineInput
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.ray_wrapped_inference_engine import create_ray_wrapped_inference_engines
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_train.models.grug_moe import GRUG_ROUTER_BIAS_SUFFIX, GrugMoeConfig, GrugMoeForCausalLM
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.utils import initialize_ray
from skyrl_train.utils.torch_utils import logprobs_from_logits
from tests.gpu.grug_gpu_gates import require_hoppers
from tests.gpu.utils import get_test_actor_config, init_worker_with_type

TOKENIZER = "Qwen/Qwen2.5-0.5B-Instruct"
NUM_LAYERS = 8
NUM_EXPERTS = 8
ROLLOUT_WORLD_SIZE = 2
RESPONSE_LENGTH = 8
PROMPT_LENGTH = 12
STACKED_EXPERT_NAME = "model.layers.0.mlp.experts.gate_proj.weight"
SERVING_EXPERT_INDEX_BY_NAME = {
    "model.layers.0.mlp.experts.0.gate_proj.weight": 0,
    "model.layers.0.mlp.experts.5.gate_proj.weight": 5,
}
LM_HEAD_NAME = "lm_head.weight"
ROUTER_NAME = "model.layers.0.mlp.router.weight"
LONG_LAYER_Q_NAME = "model.layers.3.self_attn.q_proj.weight"
ATTN_GATE_NAME = "model.layers.1.self_attn.attn_gate.weight"
GATED_NORM_NAME = "model.layers.1.attn_gated_norm.up_proj.weight"
FINAL_NORM_NAME = "model.norm.weight"
EMBED_GATE_NAME = "model.embed_gated_norm.down_proj.weight"
PARAMETER_NAMES = [
    "model.layers.0.self_attn.q_proj.weight",
    LONG_LAYER_Q_NAME,
    ATTN_GATE_NAME,
    GATED_NORM_NAME,
    EMBED_GATE_NAME,
    FINAL_NORM_NAME,
    STACKED_EXPERT_NAME,
    "model.layers.7.mlp.experts.down_proj.weight",
    "model.layers.2.shared_expert.up_proj.weight",
    LM_HEAD_NAME,
    ROUTER_NAME,
]
BIAS_NAMES = [f"model.layers.{idx}{GRUG_ROUTER_BIAS_SUFFIX}" for idx in range(NUM_LAYERS)]
LOGPROB_MAX_ABS_TOLERANCE = 2e-1
LOGPROB_MEAN_ABS_TOLERANCE = 3e-2


def _write_tiny_checkpoint(path: Path) -> None:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    config = GrugMoeConfig(
        vocab_size=len(tokenizer),
        hidden_size=64,
        intermediate_size=64,
        shared_expert_intermediate_size=64,
        num_local_experts=NUM_EXPERTS,
        num_experts_per_tok=2,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        max_position_embeddings=128,
        sliding_window=16,
        initializer_range=0.02,
        qk_mult=1.37,
        qk_mult_long_scale=1.1,
    )
    torch.manual_seed(17)
    model = GrugMoeForCausalLM(config)
    with torch.no_grad():
        # Non-trivial gates and router biases so the Megatron port has to reproduce them.
        for module in model.modules():
            if module.__class__.__name__ == "GrugMoeGatedNorm":
                module.down_proj.weight.normal_(std=0.2)
                module.up_proj.weight.normal_(std=0.2)
        for layer in model.model.layers:
            layer.self_attn.attn_gate.weight.normal_(std=0.2)
            layer.mlp.router.bias.copy_(torch.linspace(-0.3, 0.3, NUM_EXPERTS))
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)


def _config(model_path: str, *, world_size: int, pp: int, ep: int):
    cfg = get_test_actor_config()
    cfg.trainer.policy.model.path = model_path
    cfg.trainer.critic.model.path = None
    cfg.trainer.strategy = "megatron"
    cfg.trainer.flash_attn = False
    cfg.trainer.bf16 = True
    cfg.trainer.gradient_checkpointing = True
    cfg.trainer.use_sample_packing = False
    cfg.trainer.train_batch_size = 4
    cfg.trainer.policy_mini_batch_size = 4
    cfg.trainer.micro_train_batch_size_per_gpu = 2
    cfg.trainer.micro_forward_batch_size_per_gpu = 2
    cfg.trainer.update_epochs_per_batch = 1
    cfg.trainer.algorithm.use_kl_loss = False
    cfg.trainer.algorithm.use_entropy_loss = False
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.placement.policy_num_nodes = 1
    cfg.trainer.placement.policy_num_gpus_per_node = world_size
    cfg.trainer.policy.megatron_config.tensor_model_parallel_size = 1
    cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = pp
    cfg.trainer.policy.megatron_config.context_parallel_size = 1
    cfg.trainer.policy.megatron_config.expert_model_parallel_size = ep
    cfg.trainer.policy.megatron_config.expert_tensor_parallel_size = 1
    # Large enough that one Adam step moves bf16 norm weights sitting at 1.0 (ulp 2^-7).
    cfg.trainer.policy.optimizer_config.lr = 2.0e-2
    cfg.trainer.policy.optimizer_config.max_grad_norm = 0.0
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


def _padded_batch(pad_token_id: int, batch_size: int = 4) -> TrainingInputBatch:
    """Return a bf16-friendly batch with left and right padding and fixed content."""
    generator = torch.Generator().manual_seed(5)
    body_length = PROMPT_LENGTH + RESPONSE_LENGTH
    total_length = body_length + 6
    pad_before = [3, 0, 5, 1]
    sequences = []
    masks = []
    for row in range(batch_size):
        body = torch.randint(10, 500, (body_length,), generator=generator).tolist()
        before = pad_before[row % len(pad_before)]
        after = total_length - body_length - before
        sequences.append([pad_token_id] * before + body + [pad_token_id] * after)
        masks.append([0] * before + [1] * body_length + [0] * after)
    sequences = torch.tensor(sequences, dtype=torch.long)
    attention_mask = torch.tensor(masks, dtype=torch.long)
    response_mask = attention_mask[:, -RESPONSE_LENGTH:]
    zeros = torch.zeros(batch_size, RESPONSE_LENGTH, dtype=torch.float32)
    advantages = torch.tensor([[-1.0, -0.25, 0.5, 1.0, 1.0, 0.5, -0.25, -1.0]]).repeat(batch_size, 1)
    batch = TrainingInputBatch(
        {
            "sequences": sequences,
            "attention_mask": attention_mask,
            "action_log_probs": zeros.clone(),
            "base_action_log_probs": zeros.clone(),
            "rollout_logprobs": zeros.clone(),
            "values": zeros.clone(),
            "returns": zeros.clone(),
            "advantages": advantages,
            "loss_mask": response_mask.clone(),
            "response_mask": response_mask.clone(),
        }
    )
    batch.metadata = {"response_length": RESPONSE_LENGTH, "global_step": 0}
    return batch


@ray.remote(num_gpus=1)
def _hf_response_logprobs(model_path: str, batch: TrainingInputBatch) -> torch.Tensor:
    model = GrugMoeForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, attn_implementation="eager")
    model.eval().to("cuda")
    sequences = batch["sequences"].to("cuda")
    attention_mask = batch["attention_mask"].to("cuda")
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    num_actions = batch.metadata["response_length"]
    with torch.no_grad():
        logits = model(sequences, attention_mask=attention_mask, position_ids=position_ids).logits
        log_probs = logprobs_from_logits(logits, torch.roll(sequences, shifts=-1, dims=1))
    return log_probs[:, -num_actions - 1 : -1].float().cpu()


def _megatron_response_logprobs(policy, batch: TrainingInputBatch) -> torch.Tensor:
    outputs = ray.get(policy.async_run_ray_method("mesh", "forward", data=batch))
    return concatenate_outputs_after_mesh_dispatch(policy.actor_infos, outputs)["output"].float()


def _assert_logprobs_close(actual: torch.Tensor, expected: torch.Tensor, response_mask: torch.Tensor) -> None:
    valid = response_mask.bool()
    diff = (actual[valid] - expected[valid]).abs()
    assert diff.max().item() < LOGPROB_MAX_ABS_TOLERANCE, diff.max()
    assert diff.mean().item() < LOGPROB_MEAN_ABS_TOLERANCE, diff.mean()


def _snapshot(policy, names: list[str]) -> dict[str, torch.Tensor]:
    snapshots = ray.get(policy.async_run_ray_method("pass_through", "grug_validation_snapshot", names))
    return next(snapshot for snapshot in snapshots if snapshot.rank == 0).weights


def _init_policy(cfg, world_size: int):
    return init_worker_with_type(
        "policy", shared_pg=None, colocate_all=False, num_gpus_per_node=world_size, num_nodes=1, cfg=cfg
    )


def _train_step(policy, batch: TrainingInputBatch) -> dict[str, float]:
    train_output = ray.get(policy.async_run_ray_method("pass_through", "ppo_train", batch))[0]
    status = train_output.metadata["train_status"]
    assert math.isfinite(status["policy_loss"])
    return status


@pytest.mark.parametrize(
    ("world_size", "pp", "ep"),
    [(1, 1, 1), (2, 2, 1), (4, 2, 2)],
    ids=["pp1", "pp2", "pp2_ep2"],
)
def test_grug_megatron_forward_matches_hf(tmp_path, world_size: int, pp: int, ep: int):
    require_hoppers(world_size)
    model_path = tmp_path / "model"
    model_path.mkdir()
    _write_tiny_checkpoint(model_path)
    cfg = _config(str(model_path), world_size=world_size, pp=pp, ep=ep)
    pad_token_id = AutoTokenizer.from_pretrained(model_path).pad_token_id
    batch = _padded_batch(pad_token_id)
    initialize_ray(cfg)
    try:
        expected = ray.get(_hf_response_logprobs.remote(str(model_path), batch))
        policy = _init_policy(cfg, world_size)
        actual = _megatron_response_logprobs(policy, batch)
        _assert_logprobs_close(actual, expected, batch["response_mask"])
    finally:
        ray.shutdown()


def test_grug_megatron_pp2_train_step_updates_weights_and_exports(tmp_path):
    world_size = 2
    require_hoppers(world_size)
    model_path = tmp_path / "model"
    model_path.mkdir()
    _write_tiny_checkpoint(model_path)
    cfg = _config(str(model_path), world_size=world_size, pp=2, ep=1)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    batch = _padded_batch(tokenizer.pad_token_id)
    export_dir = tmp_path / "export"
    initialize_ray(cfg)
    try:
        policy = _init_policy(cfg, world_size)
        names = [*PARAMETER_NAMES, *BIAS_NAMES]
        original = GrugMoeForCausalLM.from_pretrained(model_path, dtype=torch.float32).state_dict()
        before = _snapshot(policy, names)
        for name in PARAMETER_NAMES:
            # Megatron holds bf16 parameters; the fp32 checkpoint rounds once on load.
            torch.testing.assert_close(before[name], original[name].to(torch.bfloat16).float(), rtol=0, atol=0)
        for name in BIAS_NAMES:
            torch.testing.assert_close(before[name], original[name].float(), rtol=0, atol=0)

        _train_step(policy, batch)
        after = _snapshot(policy, names)
        for name in PARAMETER_NAMES:
            assert not torch.equal(after[name], before[name]), f"{name} did not update"
        for name in BIAS_NAMES:
            torch.testing.assert_close(after[name], before[name], rtol=0, atol=0)

        post_update = _megatron_response_logprobs(policy, batch)
        ray.get(policy.async_run_ray_method("pass_through", "save_hf_model", str(export_dir), tokenizer))
        exported = GrugMoeForCausalLM.from_pretrained(export_dir, dtype=torch.float32).state_dict()
        for name in names:
            torch.testing.assert_close(exported[name].float(), after[name], rtol=0, atol=0)
        assert all(exported[name].dtype == torch.float32 for name in BIAS_NAMES)
        reloaded = ray.get(_hf_response_logprobs.remote(str(export_dir), batch))
        _assert_logprobs_close(post_update, reloaded, batch["response_mask"])
    finally:
        ray.shutdown()


def _engine_client(cfg, model_path: str) -> InferenceEngineClient:
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
        engine_init_timeout_seconds=cfg.generator.engine_init_timeout_seconds,
        shared_pg=None,
        gpu_memory_utilization=cfg.generator.gpu_memory_utilization,
        inference_engine_enable_sleep=False,
        async_engine=True,
        max_num_batched_tokens=128 * cfg.generator.inference_engine_data_parallel_size,
        max_num_seqs=cfg.trainer.train_batch_size,
        tokenizer=tokenizer,
        backend="vllm",
        engine_init_kwargs={"max_model_len": 128},
    )
    return InferenceEngineClient(engines, tokenizer, cfg)


def _rollout_batch(prompts: list[list[int]], rollout) -> TrainingInputBatch:
    sequences = torch.tensor(
        [prompt + response for prompt, response in zip(prompts, rollout["response_ids"])], dtype=torch.long
    )
    logprobs = torch.tensor(rollout["response_logprobs"], dtype=torch.float32)
    ones = torch.ones_like(logprobs)
    advantages = torch.tensor([[-1.0, -0.25, 0.5, 1.0], [1.0, 0.5, -0.25, -1.0]]).repeat(2, 1)[: sequences.shape[0]]
    batch = TrainingInputBatch(
        {
            "sequences": sequences,
            "attention_mask": torch.ones_like(sequences),
            "action_log_probs": logprobs,
            "base_action_log_probs": logprobs.clone(),
            "rollout_logprobs": logprobs.clone(),
            "values": torch.zeros_like(ones),
            "returns": torch.zeros_like(ones),
            "advantages": advantages,
            "loss_mask": ones.long(),
            "response_mask": ones.long(),
        }
    )
    batch.metadata = {"response_length": logprobs.shape[1], "global_step": 0}
    return batch


def _assert_engine_weights(client, training: dict[str, torch.Tensor]) -> None:
    names = [*BIAS_NAMES, *SERVING_EXPERT_INDEX_BY_NAME, LM_HEAD_NAME, ROUTER_NAME, ATTN_GATE_NAME, GATED_NORM_NAME]
    found = {name: False for name in names}
    for engine in client.engines:
        per_rank = ray.get(engine.inference_engine_actor.read_engine_weights.remote(names, False))
        if isinstance(per_rank, dict):
            per_rank = [per_rank]
        for rank_values in per_rank:
            for name in names:
                entry = rank_values[name]
                if entry.get("skip"):
                    continue
                assert entry["found"], (name, entry)
                found[name] = True
                expected_dtype = "float32" if name in BIAS_NAMES or name == ROUTER_NAME else "bfloat16"
                assert entry["dtype"] == expected_dtype, (name, entry["dtype"])
                expert_index = SERVING_EXPERT_INDEX_BY_NAME.get(name)
                expected = training[STACKED_EXPERT_NAME][expert_index] if expert_index is not None else training[name]
                actual = entry["tensor"]
                if name not in BIAS_NAMES:
                    expected = expected.to(torch.bfloat16).to(actual.dtype)
                if name == LM_HEAD_NAME:
                    assert actual.shape[0] >= expected.shape[0], (actual.shape, expected.shape)
                    actual = actual[: expected.shape[0]]
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert all(found.values()), found


@pytest.mark.vllm
def test_grug_megatron_four_gpu_pp2_disaggregated_rollout_train_broadcast_rollout(tmp_path):
    """Rollout, PP2 Megatron update, mixed-dtype broadcast, serving readback, rollout."""
    policy_world_size = 2
    require_hoppers(policy_world_size + ROLLOUT_WORLD_SIZE)
    model_path = tmp_path / "model"
    model_path.mkdir()
    _write_tiny_checkpoint(model_path)
    cfg = _config(str(model_path), world_size=policy_world_size, pp=2, ep=1)
    initialize_ray(cfg)
    client = _engine_client(cfg, str(model_path))
    try:
        prompt_pattern = [[1, 17, 29, 5, 11, 3], [1, 19, 31, 7, 13, 3]]
        prompts = [prompt_pattern[idx % 2] for idx in range(cfg.trainer.train_batch_size)]
        sampling_params = get_sampling_params_for_backend(cfg.generator.backend, cfg.generator.sampling_params)
        sampling_params.update({"temperature": 0.0, "ignore_eos": True, "logprobs": 1})
        first_rollout = asyncio.run(
            client.generate(InferenceEngineInput(prompt_token_ids=prompts, sampling_params=sampling_params))
        )
        first_token = first_rollout["response_ids"][0][0]
        score_params = dict(sampling_params)
        score_params.update({"max_tokens": 1, "prompt_logprobs": 1})
        score_input = InferenceEngineInput(prompt_token_ids=[prompts[0] + [first_token]], sampling_params=score_params)
        first_logprob = asyncio.run(client.generate(score_input))["prompt_logprobs"][0][-1][first_token]

        policy = _init_policy(cfg, policy_world_size)
        names = [*PARAMETER_NAMES, *BIAS_NAMES, GATED_NORM_NAME]
        before = _snapshot(policy, names)
        _train_step(policy, _rollout_batch(prompts, first_rollout))
        training = _snapshot(policy, names)
        for name in PARAMETER_NAMES:
            assert not torch.equal(training[name], before[name]), f"{name} did not update"

        ray.get(policy.async_run_ray_method("pass_through", "init_weight_sync_state", client))
        ray.get(policy.async_run_ray_method("pass_through", "broadcast_to_inference_engines", client))
        _assert_engine_weights(client, training)

        asyncio.run(client.reset_prefix_cache())
        second_logprob = asyncio.run(client.generate(score_input))["prompt_logprobs"][0][-1][first_token]
        assert abs(second_logprob - first_logprob) > 1e-7
    finally:
        ray.shutdown()
