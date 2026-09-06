# Utils ported from Verl
# https://github.com/volcengine/verl/blob/e1603dc97f3c20c58feed1f5be34acd5c72a830c/verl/utils/megatron/optimizer.py#L4
# The original copyright is reproduced below:

# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from skyrl_train.utils.utils import str_to_torch_dtype


OPTIMIZER_DTYPES = {
    "params_dtype": {torch.float32, torch.float16, torch.bfloat16},
    "main_params_dtype": {torch.float32, torch.float16},
    # This worker uses ordinary MCore DDP, whose buffers follow
    # ddp_config.grad_reduce_in_fp32; main_grads_dtype does not change them.
    "main_grads_dtype": {torch.float32},
    "exp_avg_dtype": {torch.float32, torch.float16, torch.bfloat16, torch.uint8},
    "exp_avg_sq_dtype": {torch.float32, torch.float16, torch.bfloat16, torch.uint8},
}


def megatron_optimizer_kwargs(optim_config: dict, optimizer_config_kwargs: dict) -> dict:
    """Normalize declared wire dtypes before constructing the pinned native config.

    The native defaults remain implicit. TE 2.11 supports FP32/FP16 masters and
    FP32/FP16/BF16/uint8 moment storage; runtime qualification is still required.
    """
    # megatron-core only recognizes 'adam' / 'sgd' as standard optimizers (anything
    # else routes to `_get_megatron_emerging_optimizer`, which raises
    # `ValueError: Unsupported emerging optimizer: AdamW`). megatron's 'adam' IS
    # AdamW (decoupled weight decay via weight_decay), so normalize the common
    # HF-style names to 'adam'.
    _optim_name = str(optim_config.get("optimizer", "adam")).lower()
    if _optim_name == "adamw":
        _optim_name = "adam"
    optim_args = {
        "optimizer": _optim_name,
        "lr": optim_config.get("lr"),
        "min_lr": optim_config.get("min_lr", 0.0),
        "clip_grad": optim_config.get("max_grad_norm", 1.0),
        "weight_decay": optim_config.get("weight_decay", 0.01),
        "bf16": True,
        "params_dtype": torch.bfloat16,
        "use_distributed_optimizer": True,
    }

    optim_args.update(optimizer_config_kwargs)

    for name, supported in OPTIMIZER_DTYPES.items():
        if name not in optim_args:
            continue
        value = optim_args[name]
        dtype = str_to_torch_dtype(value) if isinstance(value, str) else value
        if not isinstance(dtype, torch.dtype) or dtype not in supported:
            if name == "main_grads_dtype":
                raise ValueError(
                    "main_grads_dtype must be float32 here; actual gradients follow DDP grad_reduce_in_fp32"
                )
            raise ValueError(f"Unsupported {name}={value!r} for pinned Megatron/TE optimizer")
        optim_args[name] = dtype

    state_names = ("main_params_dtype", "main_grads_dtype", "exp_avg_dtype", "exp_avg_sq_dtype")
    lower_precision_state = any(optim_args.get(name, torch.float32) != torch.float32 for name in state_names)
    precision_aware = optim_args.get("use_precision_aware_optimizer", False)
    if lower_precision_state and not precision_aware:
        raise ValueError("Non-FP32 optimizer state requires use_precision_aware_optimizer=true")
    if precision_aware:
        if optim_args["optimizer"] != "adam" or not optim_args["use_distributed_optimizer"]:
            raise ValueError("Precision-aware state requires distributed Adam")
        if optim_args.get("main_params_dtype", torch.float32) != torch.float32 and optim_args.get(
            "store_param_remainders", True
        ):
            raise ValueError("Parameter remainders require FP32 master weights")
        if optim_args.get("optimizer_cuda_graph", False) and (
            lower_precision_state or optim_args.get("store_param_remainders", True)
        ):
            raise ValueError("Optimizer CUDA graphs require FP32 state and store_param_remainders=false")
    return optim_args


def init_megatron_optim_config(optim_config: dict, optimizer_config_kwargs: dict):
    # Megatron is optional in the CPU profile; argument validation is CPU-safe.
    from megatron.core.optimizer import OptimizerConfig

    return OptimizerConfig(**megatron_optimizer_kwargs(optim_config, optimizer_config_kwargs))


def get_megatron_optimizer(
    model,
    config,
    no_weight_decay_cond=None,
    scale_lr_cond=None,
    lr_mult=1.0,
):
    from megatron.core.optimizer import get_megatron_optimizer as get_megatron_optimizer_native

    # megatron-core 0.18.x removed the per-param-group knobs
    # (no_weight_decay_cond / scale_lr_cond / lr_mult) from get_megatron_optimizer
    # and replaced them with a `config_overrides` mapping; forwarding the old kwargs
    # raises `TypeError: get_megatron_optimizer() got an unexpected keyword argument
    # 'no_weight_decay_cond'`. This trainer never requests non-default conditioning
    # (all callers use the defaults), so forward only the args 0.18.x accepts. If
    # custom per-group conditioning is ever needed, translate it into
    # `config_overrides` here.
    if no_weight_decay_cond is not None or scale_lr_cond is not None or lr_mult != 1.0:
        raise NotImplementedError(
            "no_weight_decay_cond / scale_lr_cond / lr_mult are not wired to "
            "megatron-core 0.18.x's config_overrides mapping; only the defaults "
            "are supported."
        )
    # Base optimizer.
    return get_megatron_optimizer_native(
        config=config,
        model_chunks=model,
    )


def get_megatron_optimizer_param_scheduler(
    optimizer,
    config,
    num_training_steps: int = 1e9,  # default to a large number for constant lr/wd
):
    """
    Get the optimizer parameter scheduler for Megatron.
    """
    from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler

    # TODO: support other schedulers for Megatron
    if config.get("scheduler", "constant_with_warmup") != "constant_with_warmup":
        raise ValueError("Only constant_with_warmup scheduler is supported for Megatron")

    lr_warmup_steps = config.num_warmup_steps
    if config.get("lr_decay_steps", None) is None:
        lr_decay_steps = num_training_steps
    if config.get("lr_warmup_steps_ratio", None) is not None and (
        config.get("lr_warmup_steps", None) is None or config.lr_warmup_steps <= 0
    ):
        lr_warmup_steps = int(config.lr_warmup_steps_ratio * lr_decay_steps)

    opt_param_scheduler = OptimizerParamScheduler(
        optimizer,
        init_lr=config.get("lr_warmup_init", 0.0),
        max_lr=config.lr,
        min_lr=config.get("min_lr", 0.0),
        lr_warmup_steps=lr_warmup_steps,
        lr_decay_steps=lr_decay_steps,
        lr_decay_style="constant",
        start_wd=config.weight_decay,
        end_wd=config.weight_decay,
        wd_incr_steps=num_training_steps,
        wd_incr_style="constant",
        use_checkpoint_opt_param_scheduler=False,
        override_opt_param_scheduler=True,
        wsd_decay_steps=None,
        lr_wsd_decay_style="exponential",
    )

    return opt_param_scheduler


def get_megatron_last_lr(optimizer):
    """
    Get the last learning rate from the optimizer parameter scheduler.
    """
    return optimizer.param_groups[0]["lr"]
