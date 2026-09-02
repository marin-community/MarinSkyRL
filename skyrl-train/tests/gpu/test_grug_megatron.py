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
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_train.models.grug_moe import GRUG_ROUTER_BIAS_SUFFIX, GrugMoeConfig, GrugMoeForCausalLM
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.utils import initialize_ray
from skyrl_train.utils.torch_utils import logprobs_from_logits
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

TOKENIZER = "Qwen/Qwen2.5-0.5B-Instruct"
NUM_LAYERS = 8
NUM_EXPERTS = 8
ROLLOUT_WORLD_SIZE = 2
RESPONSE_LENGTH = 8
PROMPT_LENGTH = 12
SERVING_EXPERT_INDEX_BY_NAME = {
    "model.layers.0.mlp.experts.0.gate_proj.weight": 0,
    "model.layers.0.mlp.experts.5.gate_proj.weight": 5,
}
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
TRAIN_EVAL_LOGPROB_MAX_ABS_TOLERANCE = 1e-3


TOY_SHAPE = dict(
    hidden_size=64,
    intermediate_size=64,
    shared_expert_intermediate_size=64,
    num_local_experts=NUM_EXPERTS,
    num_hidden_layers=NUM_LAYERS,
    num_attention_heads=2,
    num_key_value_heads=1,
    head_dim=64,
    sliding_window=16,
)
# Snowball's attention geometry, expert count, and window at a fraction of its width and depth.
SNOWBALL_LIKE_SHAPE = dict(
    hidden_size=2560,
    intermediate_size=128,
    shared_expert_intermediate_size=256,
    num_local_experts=256,
    num_hidden_layers=4,
    num_attention_heads=20,
    num_key_value_heads=5,
    head_dim=128,
    sliding_window=2048,
)


def _write_tiny_checkpoint(
    path: Path,
    max_position_embeddings: int = 128,
    num_experts_per_tok: int = 2,
    shape: dict | None = None,
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    shape = TOY_SHAPE if shape is None else shape
    config = GrugMoeConfig(
        vocab_size=len(tokenizer),
        num_experts_per_tok=num_experts_per_tok,
        max_position_embeddings=max_position_embeddings,
        initializer_range=0.02,
        qk_mult=1.37,
        qk_mult_long_scale=1.1,
        **shape,
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
            layer.mlp.router.bias.copy_(torch.linspace(-0.3, 0.3, config.num_local_experts))
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


def _padded_batch(
    pad_token_id: int,
    batch_size: int = 4,
    prompt_length: int = PROMPT_LENGTH,
    response_length: int = RESPONSE_LENGTH,
    variable_lengths: bool = False,
) -> TrainingInputBatch:
    """Return a bf16-friendly batch with left and right padding and fixed content.

    With ``variable_lengths`` each row's body is shorter than the previous one, as rollouts
    are, so consecutive micro-batches have different widths.
    """
    generator = torch.Generator().manual_seed(5)
    full_body_length = prompt_length + response_length
    total_length = full_body_length + 6
    pad_before = [3, 0, 5, 1]
    sequences = []
    masks = []
    # Each row loses this many tokens relative to the previous one; the last row still
    # keeps some response tokens once total padding is accounted for.
    shrink = response_length // (2 * batch_size) if variable_lengths else 0
    for row in range(batch_size):
        body_length = full_body_length - shrink * row
        body = torch.randint(10, 500, (body_length,), generator=generator).tolist()
        before = pad_before[row % len(pad_before)]
        after = total_length - body_length - before
        assert after < response_length, (row, after, response_length)
        sequences.append([pad_token_id] * before + body + [pad_token_id] * after)
        masks.append([0] * before + [1] * body_length + [0] * after)
    sequences = torch.tensor(sequences, dtype=torch.long)
    attention_mask = torch.tensor(masks, dtype=torch.long)
    response_mask = attention_mask[:, -response_length:]
    zeros = torch.zeros(batch_size, response_length, dtype=torch.float32)
    advantages = torch.linspace(-1.0, 1.0, response_length).unsqueeze(0).repeat(batch_size, 1)
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
    batch.metadata = {"response_length": response_length, "global_step": 0}
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
    print(
        f"logprob parity over {valid.sum().item()} tokens: max abs {diff.max().item():.4f}, mean abs {diff.mean().item():.4f}"
    )
    assert diff.max().item() < LOGPROB_MAX_ABS_TOLERANCE, diff.max()
    assert diff.mean().item() < LOGPROB_MEAN_ABS_TOLERANCE, diff.mean()


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


@pytest.mark.parametrize(
    ("world_size", "pp", "ep"),
    [(1, 1, 1), (2, 2, 1), (4, 2, 2)],
    ids=["pp1", "pp2", "pp2_ep2"],
)
@pytest.mark.parametrize(
    ("shape", "num_experts_per_tok", "prompt_length", "response_length"),
    [
        (TOY_SHAPE, 2, 24, 16),
        (TOY_SHAPE, 4, 1000, 200),
        (SNOWBALL_LIKE_SHAPE, 4, 2400, 300),
    ],
    ids=["toy_top2_short", "toy_top4_long", "snowball_like_top4_long"],
)
def test_grug_megatron_train_forward_matches_eval_forward(
    tmp_path,
    world_size: int,
    pp: int,
    ep: int,
    shape: dict,
    num_experts_per_tok: int,
    prompt_length: int,
    response_length: int,
):
    """The training forward must reproduce the eval-mode log-probs it is scored against.

    With one update per batch the PPO ratio is exp(train_logprob - eval_logprob), so any
    train/eval drift shows up as spurious clipping. FSDP2 reports exactly zero here. Top-4
    routing exposed Megatron's unfused, atomic unpermute (the bridge now forces the fused
    kernels), and Snowball's width exposed cuBLAS kernel selection changing with the
    micro-batch shape (the two passes must use equal micro-batch sizes).
    """
    require_hoppers(world_size)
    model_path = tmp_path / "model"
    model_path.mkdir()
    _write_tiny_checkpoint(
        model_path, max_position_embeddings=4096, num_experts_per_tok=num_experts_per_tok, shape=shape
    )
    cfg = _config(str(model_path), world_size=world_size, pp=pp, ep=ep)
    # The forward and training passes must see the same micro-batch shapes; see the trainer docs.
    cfg.trainer.micro_forward_batch_size_per_gpu = 1
    cfg.trainer.micro_train_batch_size_per_gpu = 1
    pad_token_id = AutoTokenizer.from_pretrained(model_path).pad_token_id
    batch = _padded_batch(
        pad_token_id, prompt_length=prompt_length, response_length=response_length, variable_lengths=True
    )
    initialize_ray(cfg)
    try:
        policy = _init_policy(cfg, world_size)
        eval_logprobs = _megatron_response_logprobs(policy, batch)
        batch["action_log_probs"] = (eval_logprobs * batch["response_mask"]).float()
        train_output = ray.get(policy.async_run_ray_method("mesh", "ppo_train", batch))[0]
        status = train_output.metadata["train_status"]
        print(
            f"train/eval log-ratio: mean abs {status['log_ratio_abs_mean']:.6f}, max abs {status['log_ratio_abs_max']:.6f}, "
            f"exact-unit fraction {status['ppo_ratio_exact_unit_fraction']:.4f}"
        )
        assert status["log_ratio_abs_max"] < TRAIN_EVAL_LOGPROB_MAX_ABS_TOLERANCE, status["log_ratio_abs_max"]
    finally:
        ray.shutdown()


@pytest.mark.parametrize(("world_size", "pp", "ep"), [(2, 1, 2), (4, 2, 2)], ids=["pp1_ep2", "pp2_ep2"])
@pytest.mark.parametrize(
    ("shape", "num_experts_per_tok", "prompt_length", "response_length"),
    [(TOY_SHAPE, 4, PROMPT_LENGTH, RESPONSE_LENGTH), (SNOWBALL_LIKE_SHAPE, 4, 2400, 300)],
    ids=["toy_top4", "snowball_like_top4_long"],
)
def test_grug_megatron_eval_forward_is_independent_of_peer_rank_batch(
    tmp_path,
    world_size: int,
    pp: int,
    ep: int,
    shape: dict,
    num_experts_per_tok: int,
    prompt_length: int,
    response_length: int,
):
    """A rank's log-probs must not change when only the peer expert-parallel rank's rows change.

    Expert parallelism co-batches tokens from every rank in the expert group, so a
    composition-dependent kernel would make old log-probs depend on which rows the
    other ranks happened to hold.
    """
    require_hoppers(world_size)
    model_path = tmp_path / "model"
    model_path.mkdir()
    _write_tiny_checkpoint(
        model_path, max_position_embeddings=4096, num_experts_per_tok=num_experts_per_tok, shape=shape
    )
    cfg = _config(str(model_path), world_size=world_size, pp=pp, ep=ep)
    pad_token_id = AutoTokenizer.from_pretrained(model_path).pad_token_id
    rows = _padded_batch(pad_token_id, batch_size=6, prompt_length=prompt_length, response_length=response_length)
    first = rows.slice(0, 4)
    peer_swapped = TrainingInputBatch({key: torch.cat([rows[key][0:2], rows[key][4:6]]) for key in rows.keys()})
    peer_swapped.metadata = rows.metadata
    initialize_ray(cfg)
    try:
        policy = _init_policy(cfg, world_size)
        baseline = _megatron_response_logprobs(policy, first)
        repeated = _megatron_response_logprobs(policy, first)
        swapped = _megatron_response_logprobs(policy, peer_swapped)
        valid = first["response_mask"][:2].bool()
        repeat_diff = (repeated[:2][valid] - baseline[:2][valid]).abs().max().item()
        swap_diff = (swapped[:2][valid] - baseline[:2][valid]).abs().max().item()
        print(f"rank-0 rows: repeat max abs {repeat_diff:.6f}, peer-swapped max abs {swap_diff:.6f}")
        assert repeat_diff == 0.0, repeat_diff
        assert swap_diff == 0.0, swap_diff
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
        before = rank0_validation_snapshot(policy, names)
        for name in PARAMETER_NAMES:
            # Megatron holds bf16 parameters; the fp32 checkpoint rounds once on load.
            torch.testing.assert_close(before[name], original[name].to(torch.bfloat16).float(), rtol=0, atol=0)
        for name in BIAS_NAMES:
            torch.testing.assert_close(before[name], original[name].float(), rtol=0, atol=0)

        _train_step(policy, batch)
        after = rank0_validation_snapshot(policy, names)
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
    client = grug_engine_client(cfg, str(model_path))
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
        before = rank0_validation_snapshot(policy, names)
        _train_step(policy, rollout_training_batch(prompts, first_rollout))
        training = rank0_validation_snapshot(policy, names)
        for name in PARAMETER_NAMES:
            assert not torch.equal(training[name], before[name]), f"{name} did not update"

        ray.get(policy.async_run_ray_method("pass_through", "init_weight_sync_state", client))
        ray.get(policy.async_run_ray_method("pass_through", "broadcast_to_inference_engines", client))
        sync_names = [
            *BIAS_NAMES,
            *SERVING_EXPERT_INDEX_BY_NAME,
            LM_HEAD_NAME,
            ROUTER_NAME,
            ATTN_GATE_NAME,
            GATED_NORM_NAME,
        ]
        assert_engine_weights(client, sync_names, training, BIAS_NAMES, SERVING_EXPERT_INDEX_BY_NAME)

        asyncio.run(client.reset_prefix_cache())
        second_logprob = asyncio.run(client.generate(score_input))["prompt_logprobs"][0][-1][first_token]
        assert abs(second_logprob - first_logprob) > 1e-7
    finally:
        ray.shutdown()
