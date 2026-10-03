import asyncio
import time
from pathlib import Path

import numpy as np
import ray
import torch
from transformers import AutoTokenizer

from skyrl_train.config.grug_vllm_shapes import VOCAB
from skyrl_train.config.numerics import Numerics
from skyrl_train.distributed.dispatch import concatenate_outputs_after_mesh_dispatch
from skyrl_train.entrypoints.main_base import create_ray_wrapped_inference_engines_from_config
from skyrl_train.inference_engines.base import InferenceEngineInput, InferenceEngineOutput
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_train.models.grug_moe import GrugMoeConfig, GrugMoeForCausalLM
from skyrl_train.models.grug_vllm_kernels import NO_SERVING_RANK
from skyrl_train.training_batch import ENGINE_DP_RANKS_KEY, TrainingInputBatch
from skyrl_train.utils import initialize_ray
from skyrl_train.utils.utils import validate_cfg
from tests.gpu.grug_gpu_gates import require_hoppers
from tests.gpu.utils import get_test_actor_config, init_worker_with_type

TOKENIZER = "Qwen/Qwen2.5-0.5B-Instruct"
POLICY_WORLD_SIZE = 2
ENGINE_DATA_PARALLEL_SIZE = 2
PAD_TOKEN_ID = 0
# Prompts on both sides of the 2,048-token sliding window, served half by each engine rank; one more row is a
# trajectory that failed before its model call (one placeholder token, no loss tokens, no serving rank).
PROMPT_LENGTHS = (1500, 1700, 1900, 2000, 2047, 2100, 2200, 2400, 2600, 2800, 3000, 3100, 1600, 2300, 2700)
RESPONSE_LENGTH = 384
# Each engine step prefills at most this many tokens, so prompts prefill in chunks over several steps while earlier
# requests decode.
MAX_BATCHED_TOKENS = 2048
SAMPLING_SEED = 11


def _write_checkpoint(path: Path) -> None:
    """A Snowball-shaped Grug model cut to four layers with narrow experts and random weights."""
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    config = GrugMoeConfig(
        vocab_size=VOCAB,
        num_hidden_layers=4,
        intermediate_size=128,
        max_position_embeddings=8192,
        initializer_range=0.02,
    )
    torch.manual_seed(17)
    model = GrugMoeForCausalLM(config)
    with torch.no_grad():
        for module in model.modules():
            if module.__class__.__name__ == "GrugMoeGatedNorm":
                module.down_proj.weight.normal_(std=0.2)
                module.up_proj.weight.normal_(std=0.2)
        for layer in model.model.layers:
            layer.self_attn.attn_gate.weight.normal_(std=0.2)
            layer.mlp.router.bias.copy_(torch.linspace(-0.3, 0.3, config.num_local_experts))
            # A decode-invariant engine multiplies the router weight in bf16 and refuses values bf16 cannot hold.
            layer.mlp.router.weight.copy_(layer.mlp.router.weight.to(torch.bfloat16))
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)


def _config(model_path: str):
    rows = len(PROMPT_LENGTHS) + 1
    cfg = get_test_actor_config()
    cfg.trainer.policy.model.path = model_path
    cfg.trainer.critic.model.path = None
    cfg.trainer.strategy = "megatron"
    cfg.trainer.flash_attn = False
    cfg.trainer.bf16 = True
    cfg.trainer.gradient_checkpointing = False
    cfg.trainer.use_sample_packing = False
    cfg.trainer.max_prompt_length = max(PROMPT_LENGTHS)
    cfg.trainer.train_batch_size = rows
    cfg.trainer.policy_mini_batch_size = rows
    cfg.trainer.micro_train_batch_size_per_gpu = 4
    cfg.trainer.micro_forward_batch_size_per_gpu = 4
    cfg.trainer.update_epochs_per_batch = 1
    cfg.trainer.algorithm.use_kl_loss = False
    cfg.trainer.algorithm.use_entropy_loss = False
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.placement.policy_num_nodes = 1
    cfg.trainer.placement.policy_num_gpus_per_node = POLICY_WORLD_SIZE
    cfg.trainer.policy.megatron_config.tensor_model_parallel_size = 1
    cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = POLICY_WORLD_SIZE
    cfg.trainer.policy.megatron_config.context_parallel_size = 1
    cfg.trainer.policy.megatron_config.expert_model_parallel_size = 1
    cfg.trainer.policy.megatron_config.expert_tensor_parallel_size = 1
    # Large enough that one Adam step moves bf16 weights.
    cfg.trainer.policy.optimizer_config.lr = 2.0e-2
    cfg.trainer.policy.optimizer_config.max_grad_norm = 0.0
    cfg.generator.backend = "vllm"
    cfg.generator.weight_sync_backend = "nccl"
    cfg.generator.inference_engine_tensor_parallel_size = 1
    cfg.generator.inference_engine_data_parallel_size = ENGINE_DATA_PARALLEL_SIZE
    cfg.generator.inference_engine_expert_parallel_size = ENGINE_DATA_PARALLEL_SIZE
    cfg.generator.num_inference_engines = 1
    cfg.generator.n_samples_per_prompt = 1
    cfg.generator.max_num_batched_tokens = MAX_BATCHED_TOKENS
    cfg.generator.sampling_params.temperature = 1.0
    cfg.generator.sampling_params.top_p = 1.0
    cfg.generator.sampling_params.top_k = -1
    cfg.generator.sampling_params.max_generate_length = RESPONSE_LENGTH
    validate_cfg(cfg)
    assert cfg.trainer.algorithm.resolved_numerics == Numerics.EXACT, cfg.trainer.algorithm.numerics_fallback_reasons
    return cfg


def _prompts() -> list[list[int]]:
    generator = torch.Generator().manual_seed(5)
    return [torch.randint(10, VOCAB, (length,), generator=generator).tolist() for length in PROMPT_LENGTHS]


def _generate(client: InferenceEngineClient, prompts: list[list[int]], sampling_params: dict) -> InferenceEngineOutput:
    started = time.monotonic()
    rollout = asyncio.run(
        client.generate(InferenceEngineInput(prompt_token_ids=prompts, sampling_params=sampling_params))
    )
    print(f"generated {len(prompts)} responses in {time.monotonic() - started:.1f}s: {rollout['stop_reasons']}")
    assert all(len(response) == RESPONSE_LENGTH for response in rollout["response_ids"]), rollout["stop_reasons"]
    assert set(rollout["engine_dp_ranks"]) == set(range(ENGINE_DATA_PARALLEL_SIZE)), rollout["engine_dp_ranks"]
    return rollout


def _engine_logprobs(rollout: InferenceEngineOutput) -> np.ndarray:
    return np.asarray(rollout["response_logprobs"], dtype=np.float32)


def _batch(prompts: list[list[int]], rollout: InferenceEngineOutput) -> TrainingInputBatch:
    """The rollout's rows and the failed trajectory's placeholder row, left-padded, as the trainer receives them."""
    rows = [prompt + response for prompt, response in zip(prompts, rollout["response_ids"], strict=True)]
    rows.append([PAD_TOKEN_ID, PAD_TOKEN_ID])
    width = max(len(row) for row in rows)
    sequences = torch.full((len(rows), width), PAD_TOKEN_ID, dtype=torch.long)
    attention_mask = torch.zeros(len(rows), width, dtype=torch.long)
    for index, row in enumerate(rows):
        sequences[index, width - len(row) :] = torch.tensor(row)
        attention_mask[index, width - len(row) :] = 1
    loss_mask = attention_mask[:, -RESPONSE_LENGTH:].clone()
    loss_mask[-1] = 0
    engine_logprobs = torch.zeros(len(rows), RESPONSE_LENGTH)
    engine_logprobs[:-1] = torch.from_numpy(_engine_logprobs(rollout))
    advantages = torch.linspace(-1.0, 1.0, RESPONSE_LENGTH).unsqueeze(0).repeat(len(rows), 1) * loss_mask
    batch = TrainingInputBatch(
        {
            "sequences": sequences,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "response_mask": attention_mask[:, -RESPONSE_LENGTH:].clone(),
            "rollout_logprobs": engine_logprobs,
            "base_action_log_probs": engine_logprobs.clone(),
            "values": torch.zeros_like(engine_logprobs),
            "returns": torch.zeros_like(engine_logprobs),
            "advantages": advantages,
            ENGINE_DP_RANKS_KEY: torch.tensor([*rollout["engine_dp_ranks"], NO_SERVING_RANK], dtype=torch.long),
        }
    )
    batch.metadata = {"response_length": RESPONSE_LENGTH, "global_step": 0}
    return batch


def _trainer_logprobs(policy, batch: TrainingInputBatch) -> torch.Tensor:
    started = time.monotonic()
    outputs = ray.get(policy.async_run_ray_method("mesh", "forward", data=batch))
    logprobs = concatenate_outputs_after_mesh_dispatch(policy.actor_infos, outputs)["output"].float().cpu()
    print(f"scored {batch.batch_size} rows in {time.monotonic() - started:.1f}s")
    return logprobs


def _mismatches(name: str, trainer: torch.Tensor, rollout: InferenceEngineOutput) -> int:
    """How many of the rollout's log-probabilities the trainer's rows (the rollout's rows first) do not reproduce bit
    for bit."""
    engine = _engine_logprobs(rollout)
    scores = trainer[: engine.shape[0]].numpy()
    differing = scores.view(np.int32) != engine.view(np.int32)
    largest = np.abs(scores.astype(np.float64) - engine.astype(np.float64)).max()
    print(
        f"{name}: {int(differing.sum())} of {engine.size} log-probabilities differ "
        f"({int(differing.any(axis=1).sum())} of {engine.shape[0]} rows; largest difference {largest:.3g})"
    )
    return int(differing.sum())


def test_exact_trainer_reproduces_decode_invariant_engine_logprobs_across_a_weight_update(tmp_path):
    """Two decode-invariant engines (DP 2, EP 2) generate 384-token responses to prompts on both sides of the sliding
    window, in steps that mix chunked prefill with decode; the exact Megatron trainer (PP 2) must give every response
    token the engine's log-probability bit for bit, from each sequence's serving engine rank, with a failed
    trajectory's placeholder row in the batch. Re-reading the same prompts through the prefix cache must give the same
    responses, and after one optimizer step and the production weight sync the new responses must match again.
    """
    require_hoppers(POLICY_WORLD_SIZE + ENGINE_DATA_PARALLEL_SIZE)
    model_path = tmp_path / "model"
    model_path.mkdir()
    _write_checkpoint(model_path)
    cfg = _config(str(model_path))
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    prompts = _prompts()
    sampling_params = {
        **get_sampling_params_for_backend(cfg.generator.backend, cfg.generator.sampling_params),
        "max_tokens": RESPONSE_LENGTH,
        "logprobs": 0,
        "seed": SAMPLING_SEED,
        "stop": None,
    }
    initialize_ray(cfg)
    try:
        engines = create_ray_wrapped_inference_engines_from_config(cfg, colocate_pg=None, tokenizer=tokenizer)
        client = InferenceEngineClient(engines, tokenizer, cfg)
        first = _generate(client, prompts, sampling_params)
        # The second read of the same prompts hits the prefix cache: the same bytes, so the same seeded samples.
        cached = _generate(client, prompts, sampling_params)
        assert cached["response_ids"] == first["response_ids"]
        assert _mismatches("prefix-cache re-read", torch.from_numpy(_engine_logprobs(cached)), first) == 0

        policy = init_worker_with_type(
            "policy", shared_pg=None, colocate_all=False, num_gpus_per_node=POLICY_WORLD_SIZE, num_nodes=1, cfg=cfg
        )
        batch = _batch(prompts, first)
        logprobs = _trainer_logprobs(policy, batch)
        assert _mismatches("initial weights", logprobs, first) == 0

        batch["action_log_probs"] = logprobs * batch["loss_mask"]
        status = ray.get(policy.async_run_ray_method("mesh", "ppo_train", batch))[0].metadata["train_status"]
        print(f"policy step: loss {status['policy_loss']}")
        ray.get(policy.async_run_ray_method("pass_through", "init_weight_sync_state", client))
        ray.get(policy.async_run_ray_method("pass_through", "broadcast_to_inference_engines", client))

        updated = _generate(client, prompts, sampling_params)
        assert updated["response_ids"] != first["response_ids"]
        assert _mismatches("updated weights", _trainer_logprobs(policy, _batch(prompts, updated)), updated) == 0
    finally:
        ray.shutdown()
