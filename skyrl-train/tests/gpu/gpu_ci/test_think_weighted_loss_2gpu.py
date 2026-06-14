"""Stage D (F7) — 2-GPU integration gate for the think-weighted policy loss.

Validates on real GPUs (2 GPUs, fsdp2 policy worker):

  1. The weighted loss mask (think_token_weight < 1 on SPAN_THINK tokens) feeds
     the REAL 2-GPU `ppo_train` policy-loss path and produces a FINITE loss with
     >0 update steps — i.e. F7's per-token weighting does NOT break the
     advantage/loss denominator (the recurring seqnorm-style failure mode). The
     weighting rides the existing reduce_loss / masked_mean weighted-mean seam.
  2. Loss-path PARITY at weight 1.0: with think_token_weight == 1.0 the loss is
     bit-identical whether or not response_span_tags are present on the batch
     (the byte-identical contract, exercised through the real worker).

Run with:
    uv run --isolated --extra dev --extra deepspeed pytest \
        tests/gpu/gpu_ci/test_think_weighted_loss_2gpu.py
(or via tests/gpu/run_think_weighted_loss_2gpu.sbatch on Jupiter reformo)
"""

import math

import pytest
import ray
import torch
from omegaconf import DictConfig

from tests.gpu.utils import init_worker_with_type, get_test_actor_config, validate_cfg
from skyrl_train.training_batch import TrainingInputBatch

SPAN_THINK = 1
SPAN_ACTION = 2


@pytest.fixture
def cfg() -> DictConfig:
    cfg = get_test_actor_config()
    cfg.trainer.update_epochs_per_batch = 1
    cfg.trainer.micro_train_batch_size_per_gpu = 1
    cfg.trainer.policy_mini_batch_size = 2
    cfg.generator.n_samples_per_prompt = 1
    cfg.trainer.placement.policy_num_gpus_per_node = 2
    cfg.trainer.logger = "console"
    cfg.generator.inference_engine_tensor_parallel_size = 2
    cfg.trainer.algorithm.advantage_estimator = "rloo_n"
    cfg.trainer.algorithm.enable_token_reward_channel = True
    cfg.trainer.strategy = "fsdp2"  # production path; deepspeed absent in the rl venv
    validate_cfg(cfg)
    return cfg


def _run_ppo_train(cfg, think_token_weight, with_span_tags: bool):
    """Run the real 2-GPU ppo_train with a dummy batch; optionally attach
    response_span_tags and set the F7 think_token_weight. Returns worker status."""
    cfg.trainer.algorithm.think_token_weight = think_token_weight
    bsz, num_actions, seq_len = 2, 4, 10
    torch.manual_seed(42)
    advantages = torch.tensor([[0.6, 0.6, 0.6, 0.6], [-0.4, -0.4, -0.4, -0.4]])
    batch_data = {
        "sequences": torch.randint(0, 100, (bsz, seq_len)),
        "attention_mask": torch.ones((bsz, seq_len), dtype=int),
        "action_log_probs": 0.4 * torch.ones((bsz, num_actions)),
        "base_action_log_probs": 0.3 * torch.ones((bsz, num_actions)),
        "values": 0.5 * torch.ones((bsz, num_actions)),
        "returns": advantages.clone(),
        "advantages": advantages,
        "loss_mask": torch.ones((bsz, num_actions), dtype=int),
        "response_mask": torch.ones((bsz, num_actions), dtype=int),
    }
    if with_span_tags:
        # Tag the first two response tokens THINK, the rest ACTION.
        tags = torch.tensor(
            [[SPAN_THINK, SPAN_THINK, SPAN_ACTION, SPAN_ACTION],
             [SPAN_THINK, SPAN_ACTION, SPAN_ACTION, SPAN_ACTION]],
            dtype=torch.long,
        )
        batch_data["response_span_tags"] = tags
    data = TrainingInputBatch(batch_data)
    data.metadata = {"response_length": num_actions, "global_step": 0}
    actor_group = init_worker_with_type(
        "policy",
        shared_pg=None,
        colocate_all=False,
        num_gpus_per_node=cfg.trainer.placement.policy_num_gpus_per_node,
        cfg=cfg,
    )
    results = ray.get(actor_group.async_run_ray_method("pass_through", "ppo_train", data))
    assert len(results) == cfg.trainer.placement.policy_num_gpus_per_node
    return results[0].metadata["train_status"]


def test_think_weighted_loss_finite(ray_init_fixture, cfg):
    """Gate (1): think_token_weight=0.3 with THINK span tags feeds the real
    2-GPU policy-loss path -> finite loss + >0 update steps (no denom break)."""
    try:
        status = _run_ppo_train(cfg, think_token_weight=0.3, with_span_tags=True)
        assert status["policy_update_steps"] > 0
        assert math.isfinite(status["policy_loss"]), "weighted loss must be finite"
        assert math.isfinite(status["final_loss"])
    finally:
        ray.shutdown()


def test_think_weighted_loss_byte_identical_at_weight_one(ray_init_fixture, cfg):
    """Gate (2): at think_token_weight=1.0 the policy loss is identical whether or
    not span tags are present (the byte-identical contract through the worker)."""
    try:
        status_no_tags = _run_ppo_train(cfg, think_token_weight=1.0, with_span_tags=False)
        status_with_tags = _run_ppo_train(cfg, think_token_weight=1.0, with_span_tags=True)
        assert math.isclose(
            status_no_tags["policy_loss"], status_with_tags["policy_loss"], rel_tol=0.0, abs_tol=0.0
        ), "weight=1.0 must be byte-identical with vs without span tags"
        assert math.isclose(
            status_no_tags["final_loss"], status_with_tags["final_loss"], rel_tol=0.0, abs_tol=0.0
        )
    finally:
        ray.shutdown()
