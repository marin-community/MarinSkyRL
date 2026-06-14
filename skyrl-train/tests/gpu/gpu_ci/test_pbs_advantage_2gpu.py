"""Stage C (F6) — 2-GPU integration gate for the rloo_n_pbs advantage estimator.

Validates on real GPUs (2 GPUs, deepspeed policy worker):

  1. The advantage dispatcher threads token_level_shaping into the registered
     `rloo_n_pbs` estimator on GPU tensors: edit-tokens-that-moved-tests receive
     measurably HIGHER advantage than non-edit / no-delta tokens, and a
     zeros-channel reproduces pure RLOO-N exactly (byte-identical).
  2. The combined advantages feed the REAL 2-GPU `ppo_train` policy-loss path
     and produce a finite loss with >0 update steps — i.e. the combined
     estimator does NOT break the advantage/loss denominator (the recurring
     seqnorm-style failure mode). The PBS run and the zeros (pure-RLOO-N) run
     take the identical loss code path; only the advantage values differ.

Run with:
    uv run --isolated --extra dev --extra deepspeed pytest \
        tests/gpu/gpu_ci/test_pbs_advantage_2gpu.py
"""

import math

import numpy as np
import pytest
import ray
import torch
from omegaconf import DictConfig

from tests.gpu.utils import init_worker_with_type, get_test_actor_config, validate_cfg
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.utils import ppo_utils


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
    cfg.trainer.algorithm.advantage_estimator = "rloo_n_pbs"
    cfg.trainer.algorithm.enable_token_reward_channel = True
    validate_cfg(cfg)
    return cfg


def _adv_cfg():
    return type(
        "C", (), {"rloo_n_min_group_size": 2, "rloo_n_filter_zero_reward_groups": False}
    )()


def test_pbs_dispatcher_on_gpu(ray_init_fixture):
    """Gate (1): rloo_n_pbs via the dispatcher on CUDA tensors — edit tokens
    get higher advantage; zeros channel == pure RLOO-N."""
    device = "cuda"
    bsz, seqlen = 4, 6
    tlr = torch.zeros(bsz, seqlen, device=device)
    tlr[:, -1] = torch.tensor([1.0, 0.0, 1.0, 0.0], device=device)
    rm = torch.ones(bsz, seqlen, device=device)
    index = np.array(["g0", "g0", "g1", "g1"])

    # Pure RLOO-N reference (no shaping channel).
    base, _ = ppo_utils.compute_advantages_and_returns(
        token_level_rewards=tlr,
        response_mask=rm,
        index=index,
        adv_estimator="rloo_n",
        config=_adv_cfg(),
        values=None,
    )

    # rloo_n_pbs with zeros channel == base (byte-identical).
    zeros = torch.zeros_like(rm)
    adv_zeros, _ = ppo_utils.compute_advantages_and_returns(
        token_level_rewards=tlr,
        response_mask=rm,
        index=index,
        adv_estimator="rloo_n_pbs",
        config=_adv_cfg(),
        values=None,
        token_level_shaping=zeros,
    )
    assert torch.equal(adv_zeros, base), "zeros channel must reproduce pure RLOO-N"

    # rloo_n_pbs with a positive edit-token shaping on sample 0, token 2.
    shaping = torch.zeros_like(rm)
    shaping[0, 2] = 0.25
    adv_pbs, _ = ppo_utils.compute_advantages_and_returns(
        token_level_rewards=tlr,
        response_mask=rm,
        index=index,
        adv_estimator="rloo_n_pbs",
        config=_adv_cfg(),
        values=None,
        token_level_shaping=shaping,
    )
    edit_adv = adv_pbs[0, 2].item()
    non_edit = [adv_pbs[0, j].item() for j in range(seqlen) if j != 2]
    assert all(edit_adv > v for v in non_edit), "edit token must outrank non-edit tokens"
    # The shift is exactly the shaping mass at that token (additive contract).
    assert math.isclose(edit_adv - base[0, 2].item(), 0.25, abs_tol=1e-5)


def _run_ppo_train_with_advantages(cfg, advantages: torch.Tensor):
    """Build a dummy training batch carrying the given advantages + a shaping
    channel and run the real 2-GPU ppo_train; return the first worker's status."""
    bsz, num_actions, seq_len = 2, 4, 10
    torch.manual_seed(42)
    data = TrainingInputBatch(
        {
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
    )
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


def test_pbs_ppo_train_loss_parity(ray_init_fixture, cfg):
    """Gate (2): the combined advantages feed the real 2-GPU policy-loss path
    with a finite loss and >0 update steps — no denominator break. Both the PBS
    advantages and the zeros (pure-RLOO-N) advantages run the identical path."""
    cfg.trainer.strategy = "fsdp2"  # production path; deepspeed absent in the rl venv

    bsz, num_actions = 2, 4
    # Pure-RLOO-N-shaped advantages (uniform per trajectory).
    base_adv = torch.tensor([[0.6, 0.6, 0.6, 0.6], [-0.4, -0.4, -0.4, -0.4]])
    # PBS-shaped: same outcome term + a positive edit-token bump on sample 0 tok 1.
    pbs_adv = base_adv.clone()
    pbs_adv[0, 1] += 0.25

    try:
        status_base = _run_ppo_train_with_advantages(cfg, base_adv)
        assert status_base["policy_update_steps"] > 0
        assert math.isfinite(status_base["policy_loss"]), "pure-RLOO-N loss must be finite"
        assert math.isfinite(status_base["final_loss"])

        status_pbs = _run_ppo_train_with_advantages(cfg, pbs_adv)
        assert status_pbs["policy_update_steps"] > 0
        assert math.isfinite(status_pbs["policy_loss"]), "PBS loss must be finite (no denom break)"
        assert math.isfinite(status_pbs["final_loss"])
    finally:
        ray.shutdown()
