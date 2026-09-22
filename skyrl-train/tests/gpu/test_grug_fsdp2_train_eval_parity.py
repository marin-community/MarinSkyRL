# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train/eval log-prob parity for the FSDP2 backend.

This is the FSDP2 version of `test_grug_megatron_train_forward_matches_eval_forward` in
`tests/gpu/test_grug_megatron.py`.

With `train_batch_size == policy_mini_batch_size` and `update_epochs_per_batch: 1`, each step makes
one optimizer update, so the training forward and the old-log-prob forward run on the same weights.
Every token's PPO ratio, `exp(train_logprob - eval_logprob)`, must then be exactly 1 for any weights,
data or reward. A ratio other than 1 produces clipping with no policy change behind it.

Summing each token's expert outputs with `scatter_add` over repeated indices breaks this identity,
because CUDA runs those atomic adds in a different order on each launch. The grouped path sums them
in a fixed order (`combine_routed_rows`), as the Megatron backend does.

A flipped last bit changes a log-prob only when a downstream router picks a different expert, so
the chance of catching a regression grows with tokens times layers. Two arms target that: a
32-expert arm at Snowball's width, where each expert receives production-sized row counts, and a
26-layer arm at Snowball's depth and the production sequence length (1024 + 8192).
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import ray
import torch
from transformers import AutoTokenizer

from skyrl_train.distributed.dispatch import concatenate_outputs_after_mesh_dispatch
from skyrl_train.models.grug_moe import GrugMoeConfig, GrugMoeForCausalLM
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.utils import initialize_ray
from tests.gpu.grug_gpu_gates import require_hoppers
from tests.gpu.utils import get_test_actor_config, init_worker_with_type


TOKENIZER = "Qwen/Qwen2.5-0.5B-Instruct"
POLICY_WORLD_SIZE = 2
# The batch has exactly this many rows, and train_batch_size is set to match. The ratio diagnostics
# finalize on the last micro-step of the accumulation window (the `(local_step + 1) %
# accumulation_steps == 0` branch of PolicyWorkerBase.ppo_train), so a wider batch leaves the window
# open and the metric never appears.
BATCH_ROWS = 4

# The tolerance the Megatron version of this gate uses. It is above zero because cuBLAS picks
# kernels per GEMM shape, so some rounding remains even with equal micro-batch sizes.
TRAIN_EVAL_LOGPROB_MAX_ABS_TOLERANCE = 1e-3

# Snowball's attention geometry and window at a fraction of its width and depth.
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
# Snowball's width with few experts, so each expert receives as many rows per micro-batch as it does
# at full scale. With 256 experts the same tokens are spread too thinly to reproduce the contention.
SNOWBALL_LIKE_DENSE_EXPERTS_SHAPE = {**SNOWBALL_LIKE_SHAPE, "num_local_experts": 32}
# Snowball's depth. Order matters only for elements whose top-4 addends span more than 14 binades,
# and a flip matters only when it changes a downstream route. 4 layers of 2,700 tokens is about a
# fortieth of what a production rank processes per step; 26 layers at the production sequence
# length is the smallest shape here with a reasonable chance of catching a regression.
SNOWBALL_DEPTH_SHAPE = {**SNOWBALL_LIKE_SHAPE, "num_hidden_layers": 26}


def _write_checkpoint(path: Path, *, shape: dict, num_experts_per_tok: int = 4) -> None:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    config = GrugMoeConfig(
        vocab_size=len(tokenizer),
        num_experts_per_tok=num_experts_per_tok,
        max_position_embeddings=16384,
        initializer_range=0.02,
        qk_mult=1.37,
        **shape,
    )
    torch.manual_seed(17)
    # Saved in bf16, which is what the trainer loads anyway (model_wrapper.py requests bf16); the
    # 26-layer arm is 15 GB rather than 29 GB on disk and in each rank's host RAM at load.
    model = GrugMoeForCausalLM(config).to(torch.bfloat16)
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)


def _config(model_path: str, *, use_grouped_mm: bool):
    cfg = get_test_actor_config()
    cfg.trainer.policy.model.path = model_path
    cfg.trainer.critic.model.path = ""
    cfg.trainer.strategy = "fsdp2"
    # Flash attention does not affect this identity; it stays on so the arm matches production.
    cfg.trainer.flash_attn = True
    cfg.trainer.attn_backend = "auto"
    cfg.trainer.gradient_checkpointing = True
    cfg.trainer.gradient_checkpointing_use_reentrant = False
    cfg.trainer.use_sample_packing = False
    # One optimizer update per step, so both forwards see the same weights.
    cfg.trainer.train_batch_size = BATCH_ROWS
    cfg.trainer.policy_mini_batch_size = BATCH_ROWS
    # n_samples_per_prompt multiplies the per-rank mini-batch (policy_mini_batch_size *
    # n_samples_per_prompt // dp_size, in PolicyWorkerBase._normalize_mini_batch_size). The base
    # config's 5 makes the accumulation window 10 micro-steps wide while each rank holds 2 rows, so
    # the window never closes and no metric is published. This batch is a fixed set of rows.
    cfg.generator.n_samples_per_prompt = 1
    cfg.trainer.update_epochs_per_batch = 1
    # Equal forward and train micro-batch sizes. cuBLAS selects kernels per GEMM shape, so unequal
    # sizes put the two passes on different kernels. Megatron's validate_cfg rejects unequal sizes;
    # FSDP2 does not, so the test sets them.
    cfg.trainer.micro_train_batch_size_per_gpu = 1
    cfg.trainer.micro_forward_batch_size_per_gpu = 1
    cfg.trainer.algorithm.use_kl_loss = False
    cfg.trainer.algorithm.use_entropy_loss = False
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.placement.policy_num_nodes = 1
    cfg.trainer.placement.policy_num_gpus_per_node = POLICY_WORLD_SIZE
    cfg.trainer.policy.fsdp_config.cpu_offload = True
    cfg.trainer.policy.fsdp_config.fsdp_size = POLICY_WORLD_SIZE
    cfg.trainer.policy.fsdp_config.expert_model_parallel_size = 1
    cfg.trainer.policy.fsdp_config.context_parallel_size = 1
    cfg.trainer.policy.fsdp_config.moe_router_replay = False
    cfg.trainer.policy.fsdp_config.moe_grouped_gemm = False
    cfg.trainer.policy.fsdp_config.use_grouped_mm = use_grouped_mm
    cfg.trainer.policy.optimizer_config.max_grad_norm = 0.0
    return cfg


def _variable_length_batch(pad_token_id: int, *, prompt_length: int, response_length: int) -> TrainingInputBatch:
    """Rows of differing real length, padded to a common width.

    Variable lengths are deliberate. Equal-length rows give every micro-batch the same GEMM shape and
    the same routing histogram, which is the easiest case for both passes to agree on and would let a
    composition-dependent kernel pass unnoticed.
    """
    torch.manual_seed(17)
    width = prompt_length + response_length
    rows = []
    for real in (width, width - 7, width - 23, width - 41)[:BATCH_ROWS]:
        row = torch.randint(low=1, high=1000, size=(width,), dtype=torch.long)
        row[real:] = pad_token_id
        rows.append(row)
    sequences = torch.stack(rows)
    attention_mask = (sequences != pad_token_id).long()
    ones = torch.ones((sequences.shape[0], response_length))
    advantage_pattern = torch.tensor([[-1.0, -0.25, 0.5, 1.0], [1.0, 0.5, -0.25, -1.0]])
    advantages = advantage_pattern.repeat(math.ceil(sequences.shape[0] / 2), 1)[: sequences.shape[0]]
    advantages = advantages[:, :1].expand(-1, response_length).contiguous()
    batch = TrainingInputBatch(
        {
            "sequences": sequences,
            "attention_mask": attention_mask,
            "action_log_probs": torch.zeros_like(ones),
            "base_action_log_probs": torch.zeros_like(ones),
            "rollout_logprobs": torch.zeros_like(ones),
            "values": torch.zeros_like(ones),
            "returns": torch.zeros_like(ones),
            "advantages": advantages,
            "loss_mask": attention_mask[:, -response_length:].clone(),
            "response_mask": attention_mask[:, -response_length:].clone(),
        }
    )
    batch.metadata = {"response_length": response_length, "global_step": 0}
    return batch


@pytest.mark.parametrize("use_grouped_mm", [False, True], ids=["eager_experts", "grouped_mm"])
@pytest.mark.parametrize(
    ("shape", "prompt_length", "response_length"),
    [
        (SNOWBALL_LIKE_SHAPE, 2400, 300),
        (SNOWBALL_LIKE_DENSE_EXPERTS_SHAPE, 2400, 300),
        (SNOWBALL_DEPTH_SHAPE, 1024, 8192),
    ],
    ids=["experts256", "experts32_full_scale_tokens_per_expert", "depth26_production_tokens"],
)
def test_grug_fsdp2_train_forward_matches_eval_forward(
    tmp_path,
    use_grouped_mm: bool,
    shape: dict,
    prompt_length: int,
    response_length: int,
):
    """The training forward must reproduce the eval-mode log-probs it is scored against.

    Feed the eval forward's output back as ``action_log_probs`` and take one step. The ratio the
    trainer computes is then the train/eval drift, and `log_ratio_abs_max` reports it. This is the
    metric production publishes. It is averaged across the two ranks, so a drift on one rank reads
    at half its size.

    Do not xfail a `grouped_mm` arm that starts failing. A failure here means the identity broke.
    """
    require_hoppers(POLICY_WORLD_SIZE)
    model_path = tmp_path / "model"
    model_path.mkdir()
    _write_checkpoint(model_path, shape=shape)
    cfg = _config(str(model_path), use_grouped_mm=use_grouped_mm)
    pad_token_id = AutoTokenizer.from_pretrained(model_path).pad_token_id
    batch = _variable_length_batch(pad_token_id, prompt_length=prompt_length, response_length=response_length)

    # Check the derived geometry. Either mistake disables the measurement without an error: a window
    # wider than a rank's rows never closes, and unequal micro sizes change the cuBLAS kernels.
    dp_size = POLICY_WORLD_SIZE
    per_rank_mini_batch = BATCH_ROWS * cfg.generator.n_samples_per_prompt // dp_size
    accumulation_steps = per_rank_mini_batch // cfg.trainer.micro_train_batch_size_per_gpu
    assert accumulation_steps == BATCH_ROWS // dp_size, (
        f"accumulation window is {accumulation_steps} micro-steps but each rank holds "
        f"{BATCH_ROWS // dp_size} rows; the ratio diagnostics would never finalize"
    )
    assert cfg.trainer.micro_forward_batch_size_per_gpu == cfg.trainer.micro_train_batch_size_per_gpu

    initialize_ray(cfg)
    try:
        policy = init_worker_with_type(
            "policy", cfg=cfg, num_gpus_per_node=POLICY_WORLD_SIZE, num_nodes=1, colocate_all=False
        )
        ray.get(policy.async_init_model(str(model_path)))

        outputs = ray.get(policy.async_run_ray_method("mesh", "forward", data=batch))
        eval_logprobs = concatenate_outputs_after_mesh_dispatch(policy.actor_infos, outputs)["output"].float()
        batch["action_log_probs"] = eval_logprobs * batch["response_mask"]

        train_output = ray.get(policy.async_run_ray_method("mesh", "ppo_train", batch))[0]
        status = train_output.metadata["train_status"]
        # A geometry mistake removes the key entirely, so report that case separately from a
        # failed comparison.
        assert "log_ratio_abs_max" in status, (
            "the ratio diagnostics never finalized, so this run measured NOTHING about the "
            f"invariant. Check the accumulation geometry. status keys: {sorted(status)}"
        )
        print(
            f"use_grouped_mm={use_grouped_mm} experts={shape['num_local_experts']}: "
            f"log_ratio_abs_max {status['log_ratio_abs_max']:.6e}, "
            f"exact-unit fraction {status['ppo_ratio_exact_unit_fraction']:.6f}"
        )
        assert status["log_ratio_abs_max"] < TRAIN_EVAL_LOGPROB_MAX_ABS_TOLERANCE, (
            f"train/eval log-ratio {status['log_ratio_abs_max']} exceeds {TRAIN_EVAL_LOGPROB_MAX_ABS_TOLERANCE}; "
            f"PPO is clipping on numerical noise"
        )
    finally:
        ray.shutdown()
