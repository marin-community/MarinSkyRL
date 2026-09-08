"""Eight-H100 native numerical qualification; requires an explicitly staged Qwen snapshot."""

import os
from functools import partial

import pytest
import ray
import torch

from skyrl_train.utils.algorithm_registry import PolicyLossRegistry
from skyrl_train.utils.utils import validate_cfg
from tests.gpu.test_megatron_worker import get_test_actor_config, get_test_training_batch, _megatron_forward
from tests.gpu.utils import init_worker_with_type
from tests.offpolicy_mask_reference import regular_correction_reference_policy_loss


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["offpolicy", "m2"])
async def test_megatron_correction_mask_matches_independent_actor(ray_init_fixture, request, mode):
    """Two TP2/DP2 actor groups compare real losses, gradients and updated logprobs."""
    model_path = os.environ.get("MARINSKYRL_TEST_QWEN_MODEL_PATH")
    assert model_path, "stage the pinned Qwen3-0.6B snapshot before this regional test"
    reference_name = "test_correction_" + mode
    PolicyLossRegistry.register(reference_name, partial(regular_correction_reference_policy_loss, mode=mode))
    request.addfinalizer(lambda: PolicyLossRegistry.unregister(reference_name))

    configs = []
    for reference in (False, True):
        cfg = get_test_actor_config(model_name=model_path)
        cfg.trainer.strategy = "megatron"
        cfg.trainer.placement.colocate_all = False
        cfg.trainer.placement.policy_num_gpus_per_node = 4
        cfg.trainer.policy.megatron_config.tensor_model_parallel_size = 2
        cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = 1
        cfg.trainer.use_sample_packing = False
        cfg.trainer.train_batch_size = cfg.trainer.policy_mini_batch_size = 4
        cfg.trainer.micro_train_batch_size_per_gpu = cfg.trainer.micro_forward_batch_size_per_gpu = 1
        cfg.trainer.algorithm.policy_loss_type = reference_name if reference else "regular"
        cfg.trainer.algorithm.use_tis = False
        cfg.trainer.algorithm.require_rollout_logprobs = True
        cfg.trainer.algorithm.use_kl_loss = cfg.trainer.algorithm.use_entropy_loss = False
        cfg.generator.n_samples_per_prompt = 1
        cfg.generator.sampling_params.logprobs = 0
        if not reference:
            getattr(cfg.trainer.algorithm, "m2_mask" if mode == "m2" else "offpolicy_mask").enabled = True
        validate_cfg(cfg)
        configs.append(cfg)

    groups = [
        init_worker_with_type("policy", shared_pg=None, colocate_all=False, num_gpus_per_node=4, cfg=cfg)
        for cfg in configs
    ]
    batches = [get_test_training_batch(batch_size=4, model_name=model_path) for _ in groups]
    before = [_megatron_forward(group, batch) for group, batch in zip(groups, batches, strict=True)]
    torch.testing.assert_close(before[0], before[1], atol=1e-4, rtol=1e-4)
    # M2 harmful deltas lie inside PPO's lower clipping bound: masking must change
    # actual gradients, not merely remove already-clipped zero-gradient terms.
    delta = torch.tensor(
        [-0.221, -0.220, -0.219, -0.218, -0.217, -0.216, -0.215, -0.214, -0.213, 0.05], dtype=before[0].dtype
    )
    old = before[0] - delta if mode == "m2" else before[0].clone()
    mismatch = torch.tensor([0.0, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=old.dtype).expand_as(old).clone()
    if mode == "offpolicy":
        mismatch[0, 0] = -18.0  # One real row veto, while other rows keep nonzero gradients.
    rollout = old - mismatch
    advantages = -torch.ones_like(old)
    mask = torch.ones_like(old)
    for batch in batches:
        batch["action_log_probs"] = old.clone()
        batch["rollout_logprobs"] = rollout.clone()
        batch["advantages"] = advantages.clone()
        batch["loss_mask"] = mask.clone()
        batch.metadata["global_step"] = 3
    results = [
        ray.get(group.async_run_ray_method("mesh", "ppo_train", batch))
        for group, batch in zip(groups, batches, strict=True)
    ]
    statuses = [[item.metadata["train_status"] for item in result] for result in results]
    assert all(len(status) == 4 for status in statuses)
    for actual, reference in zip(*statuses, strict=True):
        assert actual["policy_loss"] == pytest.approx(reference["policy_loss"], abs=1e-6)
        assert actual["raw_grad_norm"] == pytest.approx(reference["raw_grad_norm"], abs=1e-6)
        assert torch.isfinite(torch.tensor(actual["raw_grad_norm"])) and actual["raw_grad_norm"] > 0
        activity_key = "m2_mask/masked_fraction" if mode == "m2" else "offpolicy_mask/masked_fraction"
        assert actual[activity_key] > 0
        if mode == "m2":
            assert actual["m2_mask/m2_before"] > 0.04
        else:
            assert actual["offpolicy_mask/vetoed_sequence_fraction"] > 0
    after = [_megatron_forward(group, batch) for group, batch in zip(groups, batches, strict=True)]
    assert not torch.equal(before[0], after[0])
    torch.testing.assert_close(after[0], after[1], atol=1e-4, rtol=1e-4)
