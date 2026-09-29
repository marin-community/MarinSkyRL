from copy import deepcopy
from datetime import timedelta
from pathlib import Path

import pytest
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


def assert_state_equal(actual, expected):
    actual_leaves, actual_structure = tree_flatten(actual)
    expected_leaves, expected_structure = tree_flatten(expected)
    assert actual_structure == expected_structure
    for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves, strict=True):
        if isinstance(actual_leaf, torch.Tensor):
            torch.testing.assert_close(actual_leaf, expected_leaf, rtol=0, atol=0)
        else:
            assert actual_leaf == expected_leaf


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
        scheduler = OptimizerParamScheduler(
            optimizer,
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
