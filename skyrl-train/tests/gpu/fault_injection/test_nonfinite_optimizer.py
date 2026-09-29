from copy import deepcopy
from datetime import timedelta
from pathlib import Path

import pytest
import ray
import torch
import torch.distributed as dist
from megatron.core import parallel_state as mpu
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.transformer.transformer_config import TransformerConfig
from torch.utils._pytree import tree_flatten

import skyrl_train
from skyrl_train.config.utils import get_default_config
from skyrl_train.distributed.megatron.megatron_strategy import MegatronStrategy
from tests.gpu.gpu_ci.test_megatron_objective_scaling import TokenLogits, run_distributed
from tests.gpu.test_megatron_worker import get_test_actor_config, get_test_training_batch
from tests.gpu.utils import init_worker_with_type
from skyrl_train.utils.utils import validate_cfg


def assert_state_equal(actual, expected):
    actual_leaves, actual_structure = tree_flatten(actual)
    expected_leaves, expected_structure = tree_flatten(expected)
    assert actual_structure == expected_structure
    for index, (actual_leaf, expected_leaf) in enumerate(zip(actual_leaves, expected_leaves, strict=True)):
        if isinstance(actual_leaf, torch.Tensor):
            torch.testing.assert_close(actual_leaf, expected_leaf, rtol=0, atol=0)
        else:
            assert actual_leaf == expected_leaf, (index, actual_leaf, expected_leaf)


def run_nonfinite_rank(rank, dtype, rendezvous):
    assert Path(skyrl_train.__file__).resolve().parent == Path(__file__).resolve().parents[3] / "skyrl_train"
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=4, timeout=timedelta(seconds=120))
    try:
        mpu.initialize_model_parallel(context_parallel_size=2)
        model_config = TransformerConfig(
            num_layers=1,
            hidden_size=16,
            num_attention_heads=2,
            context_parallel_size=2,
            params_dtype=dtype,
            pipeline_dtype=dtype,
            bf16=dtype is torch.bfloat16,
            fp16=dtype is torch.float16,
        )
        model = TokenLogits(model_config).to(dtype=dtype)
        ddp = DistributedDataParallel(
            model_config,
            DistributedDataParallelConfig(grad_reduce_in_fp32=True, overlap_grad_reduce=False),
            model,
        )
        optimizer_config = OptimizerConfig(
            optimizer="adam",
            lr=0.01,
            min_lr=0.001,
            weight_decay=0,
            clip_grad=1,
            bf16=dtype is torch.bfloat16,
            fp16=dtype is torch.float16,
            loss_scale=128 if dtype is torch.float16 else None,
        )
        optimizer = get_megatron_optimizer(optimizer_config, [ddp])
        scheduler_args = dict(
            init_lr=0.01,
            max_lr=0.01,
            min_lr=0.001,
            lr_warmup_steps=0,
            lr_decay_steps=10,
            lr_decay_style="linear",
            start_wd=0,
            end_wd=0,
            wd_incr_steps=10,
            wd_incr_style="constant",
        )
        scheduler = OptimizerParamScheduler(optimizer, **scheduler_args)
        strategy = MegatronStrategy(get_default_config().trainer.policy.megatron_config)
        tokens = torch.arange(16, device="cuda").reshape(2, 8)

        def backward(inject_nan=False):
            ddp.zero_grad_buffer()
            logits = ddp(tokens, None, None, packed_seq_params=None, fp32_output=False)
            optimizer.scale_loss(logits.float().square().mean()).backward()
            ddp.finish_grad_sync()
            if inject_nan and rank == 0:
                model.logits.weight.main_grad.flatten()[0] = torch.nan

        def snapshot():
            return deepcopy(
                {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()}
            )

        backward()
        assert strategy.optimizer_step(optimizer, ddp, scheduler, max_consecutive_nonfinite_steps=3).applied
        before = snapshot()
        backward()
        assert strategy.optimizer_step(optimizer, ddp, scheduler, max_consecutive_nonfinite_steps=3).applied
        expected_recovery = snapshot()
        assert not torch.equal(before["model"]["logits.weight"], expected_recovery["model"]["logits.weight"])
        model.load_state_dict(before["model"])
        optimizer.load_state_dict(before["optimizer"])
        scheduler = OptimizerParamScheduler(optimizer, **scheduler_args)
        scheduler.load_state_dict(before["scheduler"])
        assert_state_equal(snapshot(), before)

        for streak in range(3):
            backward(inject_nan=True)
            result = strategy.optimizer_step(
                optimizer,
                ddp,
                scheduler,
                consecutive_nonfinite_steps=streak,
                max_consecutive_nonfinite_steps=3,
            )
            assert not result.applied and result.grad_norm is None
            assert_state_equal(snapshot(), before)
        for limit, streak in ((3, 3), (None, 0)):
            backward(inject_nan=True)
            with pytest.raises(RuntimeError, match="nonfinite policy gradients"):
                strategy.optimizer_step(
                    optimizer,
                    ddp,
                    scheduler,
                    consecutive_nonfinite_steps=streak,
                    max_consecutive_nonfinite_steps=limit,
                )
            assert_state_equal(snapshot(), before)

        optimizer_config.grad_norm_skip_threshold = 0
        backward()
        result = strategy.optimizer_step(optimizer, ddp, scheduler, max_consecutive_nonfinite_steps=3)
        assert not result.applied and result.grad_norm > 0
        assert_state_equal(snapshot(), before)
        optimizer_config.grad_norm_skip_threshold = float("inf")

        backward()
        result = strategy.optimizer_step(
            optimizer,
            ddp,
            scheduler,
            consecutive_nonfinite_steps=3,
            max_consecutive_nonfinite_steps=3,
        )
        assert result.applied
        assert_state_equal(snapshot(), expected_recovery)
        print(
            f"nonfinite DP=2 CP=2 rank={rank} dtype={dtype}: model/master/Adam/LR/step unchanged; recovery exact",
            flush=True,
        )
    finally:
        mpu.destroy_model_parallel()
        dist.destroy_process_group()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_nonfinite_gradient_preserves_optimizer_state_on_every_rank(tmp_path, dtype):
    assert torch.cuda.device_count() >= 4, "Run on an allocation with four GPUs"
    run_distributed(run_nonfinite_rank, 4, (dtype, (tmp_path / "nonfinite").as_uri()))


def test_policy_worker_reports_skips_and_resets_streak_after_clean_update(ray_init_fixture):
    assert torch.cuda.device_count() >= 4, "Run on an allocation with four GPUs"
    cfg = get_test_actor_config(logger="console")
    cfg.trainer.policy.model.revision = "c1899de289a04d12100db370d81485cdf75e47ca"
    cfg.trainer.flash_attn = True
    cfg.trainer.use_sample_packing = True
    cfg.trainer.placement.policy_num_gpus_per_node = 4
    cfg.trainer.policy.megatron_config.context_parallel_size = 2
    cfg.trainer.policy.max_consecutive_nonfinite_steps = 1
    cfg.trainer.algorithm.advantage_estimator = "uniform"
    cfg.trainer.algorithm.policy_loss_type = "importance_sampling"
    cfg.trainer.algorithm.use_kl_loss = False
    cfg.trainer.train_batch_size = 4
    cfg.trainer.policy_mini_batch_size = 4
    cfg.trainer.micro_train_batch_size_per_gpu = 1
    cfg.trainer.micro_forward_batch_size_per_gpu = 1
    cfg.generator.n_samples_per_prompt = 1
    validate_cfg(cfg)
    policy = init_worker_with_type("policy", num_gpus_per_node=4, cfg=cfg)

    for step, nonfinite in enumerate((True, False, True)):
        batch = get_test_training_batch(4)
        batch.metadata["global_step"] = step
        if nonfinite:
            batch["advantages"][0, 0] = torch.nan
        outputs = ray.get(policy.async_run_ray_method("mesh", "ppo_train", data=batch), timeout=180)
        assert len(outputs) == 4
        for output in outputs:
            status = output.metadata["train_status"]
            assert status["skipped_steps"] == int(nonfinite)
            assert status["policy_update_steps"] == int(not nonfinite)
            if nonfinite:
                assert "raw_grad_norm" not in status
            else:
                assert 0 < status["raw_grad_norm"] < float("inf")

    batch.metadata["global_step"] = 3
    with pytest.raises(ray.exceptions.RayTaskError, match="nonfinite policy gradients after 1 consecutive"):
        ray.get(policy.async_run_ray_method("mesh", "ppo_train", data=batch), timeout=180)
