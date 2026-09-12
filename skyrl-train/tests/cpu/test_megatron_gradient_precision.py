"""Configuration gates for gradient-only FP16, independent of optional Megatron imports."""

from copy import deepcopy

from omegaconf import OmegaConf
import pytest
import torch

from skyrl_train.config.utils import get_default_config
from skyrl_train.distributed.megatron.gradient_precision import (
    fp16_optimizer_overrides,
    validate_fp16_gradient_config,
)
from skyrl_train.distributed.megatron.optimizer import megatron_optimizer_kwargs


def gradient_config():
    return OmegaConf.create(
        {
            "fp16_grad_reduce": True,
            "ddp_config": {"grad_reduce_in_fp32": False},
            "dynamic_loss_scale": {"initial_scale": 128.0, "min_scale": 1.0, "growth_interval": 2, "hysteresis": 1},
        }
    )


@pytest.mark.parametrize("reduce_fp32", [True, False])
def test_existing_reduction_modes_keep_native_optimizer_without_scaling(reduce_fp32):
    config = gradient_config()
    config.fp16_grad_reduce = False
    config.ddp_config.grad_reduce_in_fp32 = reduce_fp32
    before = deepcopy(config)
    validate_fp16_gradient_config(config, bf16=True)
    native = megatron_optimizer_kwargs({"lr": 2e-6}, {})
    assert native["bf16"] is True and "fp16" not in native and "loss_scale" not in native
    assert config == before


def test_fp16_gradient_configuration_preserves_bf16_parameters_and_fp32_state():
    config = gradient_config()
    validate_fp16_gradient_config(config, bf16=True)
    optimizer = megatron_optimizer_kwargs({"lr": 2e-6}, fp16_optimizer_overrides(config))
    assert optimizer["params_dtype"] == torch.bfloat16
    assert optimizer["fp16"] is True and optimizer["loss_scale"] is None
    assert optimizer["initial_loss_scale"] == 128 and optimizer["loss_scale_window"] == 2
    assert optimizer.get("main_params_dtype", torch.float32) == torch.float32
    assert optimizer.get("exp_avg_dtype", torch.float32) == torch.float32


@pytest.mark.parametrize(
    "key,value",
    [
        ("ddp_config.grad_reduce_in_fp32", True),
        ("ddp_config.use_distributed_optimizer", False),
        ("ddp_config.overlap_grad_reduce", True),
        ("ddp_config.nccl_ub", True),
        ("ddp_config.param_name_patterns_for_fp32_local_accumulation", ["all"]),
        ("ddp_config.check_for_nan_in_grad", True),
        ("ddp_config.num_distributed_optimizer_instances", 2),
        ("expert_model_parallel_size", 8),
        ("pipeline_model_parallel_size", 2),
        ("optimizer_config_kwargs.use_precision_aware_optimizer", True),
        ("optimizer_config_kwargs.optimizer", "sgd"),
        ("optimizer_config_kwargs.use_distributed_optimizer", False),
        ("optimizer_config_kwargs.overlap_param_gather", True),
        ("optimizer_config_kwargs.use_layer_wise_distributed_optimizer", True),
        ("optimizer_config_kwargs.loss_scale", 128.0),
        ("dynamic_loss_scale.initial_scale", float("inf")),
        ("dynamic_loss_scale.initial_scale", 1e40),
        ("dynamic_loss_scale.min_scale", 1e-40),
        ("dynamic_loss_scale.initial_scale", 0.5),
        ("dynamic_loss_scale.growth_interval", 0),
    ],
)
def test_unqualified_geometry_buffer_or_scaler_cannot_silently_enter_fp16_mode(key, value):
    config = gradient_config()
    OmegaConf.update(config, key, value)
    with pytest.raises(ValueError):
        validate_fp16_gradient_config(config, bf16=True)


def test_hydra_policy_exposes_gradient_only_fp16_with_independent_scale_controls():
    cfg = get_default_config()
    precision = cfg.trainer.policy.megatron_config
    OmegaConf.update(cfg, "trainer.policy.megatron_config.fp16_grad_reduce", True)
    OmegaConf.update(cfg, "trainer.policy.megatron_config.ddp_config.grad_reduce_in_fp32", False)
    OmegaConf.update(cfg, "trainer.policy.megatron_config.dynamic_loss_scale.initial_scale", 32.0)
    validate_fp16_gradient_config(precision, bf16=cfg.trainer.bf16)
    native = megatron_optimizer_kwargs(cfg.trainer.policy.optimizer_config, fp16_optimizer_overrides(precision))
    assert native["params_dtype"] == torch.bfloat16
    assert native["initial_loss_scale"] == 32.0
    assert native["min_loss_scale"] == 1.0 and native["loss_scale_window"] == 1000
