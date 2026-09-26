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

from collections.abc import Mapping

import torch
from megatron.core.distributed import DistributedDataParallel
from megatron.core.optimizer import OptimizerConfig
from megatron.core.optimizer import get_megatron_optimizer as get_megatron_optimizer_native
from megatron.core.optimizer.emerging_optimizers import _EMERGING_OPTIMIZERS, EmergingOptimizerEntry
from megatron.core.optimizer.optimizer_config import ParamKey, ParamWithNamePredicate
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from skyrl_train.distributed.megatron.grug_muonh import (
    DEFAULT_BETAS,
    DEFAULT_EPSILON,
    DEFAULT_MOMENTUM,
    DEFAULT_NESTEROV,
    DEFAULT_NS_STEPS,
    ADAMH_ROUTE,
    ADAM_ROUTE,
    GATE_UP_LAYOUT,
    GrugMegatronMuonH,
    LAYOUT_KEY,
    MUONH_ROUTE,
    QKV_LAYOUT,
    ROUTE_KEY,
    grug_muonh_route,
)

_GRUG_MUONH_KEY = "grug_muonh"


class _GrugMuonHParamScheduler(OptimizerParamScheduler):
    def get_lr(self, param_group: dict) -> float:
        if (
            param_group.get(ROUTE_KEY) != ADAM_ROUTE
            or self.lr_warmup_steps <= 0
            or self.num_steps > self.lr_warmup_steps
        ):
            return super().get_lr(param_group)
        init_lr = self.init_lr * param_group.get("lr_mult", 1.0)
        max_lr = param_group.get("max_lr", self.max_lr)
        return init_lr + (max_lr - init_lr) * self.num_steps / self.lr_warmup_steps


def _grug_muonh_kwargs(optim_config: dict, config: OptimizerConfig) -> dict:
    if float(config.weight_decay) != 0.0:
        raise ValueError("MuonH requires weight_decay=0")
    extra = optim_config.get("optimizer_kwargs", {})
    if not isinstance(extra, Mapping):
        raise TypeError("MuonH optimizer_kwargs must be a mapping")
    known = {"adam_lr", "momentum", "nesterov", "backend_steps", "epsilon", "muon_epsilon"}
    unknown = sorted(set(extra) - known)
    if unknown:
        raise ValueError(f"Unknown MuonH optimizer_kwargs: {unknown}")
    return {
        "lr": float(config.lr),
        "min_lr": float(config.min_lr),
        "adam_lr": float(extra["adam_lr"]) if "adam_lr" in extra else None,
        "momentum": float(extra.get("momentum", DEFAULT_MOMENTUM)),
        "nesterov": bool(extra.get("nesterov", DEFAULT_NESTEROV)),
        "ns_steps": int(extra.get("backend_steps", DEFAULT_NS_STEPS)),
        "betas": (float(config.adam_beta1), float(config.adam_beta2)),
        "eps": float(config.adam_eps),
        "muon_eps": float(extra.get("muon_epsilon", DEFAULT_EPSILON)),
    }


def _register_grug_muonh() -> None:
    if _GRUG_MUONH_KEY in _EMERGING_OPTIMIZERS:
        return

    _EMERGING_OPTIMIZERS[_GRUG_MUONH_KEY] = EmergingOptimizerEntry(
        optimizer_cls=GrugMegatronMuonH,
        init_state_fn=lambda optimizer, _config=None: optimizer.initialize_state(),
        config_to_kwargs=lambda config, _chunks, _groups: config._grug_muonh_kwargs,
        default_param_overrides={
            # MCore matches checkpoint parameter groups by wd_mult/lr_mult, not
            # by grug_route. Weight decay is zero for this recipe, so distinct
            # wd_mult values preserve all update math while making route state
            # unambiguous at checkpoint load.
            ParamKey(
                with_name_predicate=ParamWithNamePredicate(
                    name="grug_muonh_adamh", fn=lambda parameter, name: grug_muonh_route(name, parameter) == ADAMH_ROUTE
                )
            ): {ROUTE_KEY: ADAMH_ROUTE, "wd_mult": 2.0},
            ParamKey(
                with_name_predicate=ParamWithNamePredicate(
                    name="grug_muonh_adam_matrix",
                    fn=lambda parameter, name: grug_muonh_route(name, parameter) == ADAM_ROUTE and parameter.ndim >= 2,
                )
            ): {ROUTE_KEY: ADAM_ROUTE, "wd_mult": 3.0},
            ParamKey(
                with_name_predicate=ParamWithNamePredicate(
                    name="grug_muonh_adam_vector",
                    fn=lambda parameter, name: grug_muonh_route(name, parameter) == ADAM_ROUTE and parameter.ndim < 2,
                )
            ): {ROUTE_KEY: ADAM_ROUTE},
            ParamKey(
                with_name_predicate=ParamWithNamePredicate(
                    name="grug_muonh_qkv", fn=lambda parameter, name: ".linear_qkv.weight" in name
                )
            ): {ROUTE_KEY: MUONH_ROUTE, LAYOUT_KEY: QKV_LAYOUT, "wd_mult": 4.0},
            ParamKey(
                with_name_predicate=ParamWithNamePredicate(
                    name="grug_muonh_gate_up", fn=lambda parameter, name: ".linear_fc1.weight" in name
                )
            ): {ROUTE_KEY: MUONH_ROUTE, LAYOUT_KEY: GATE_UP_LAYOUT, "wd_mult": 5.0},
        },
    )


def init_megatron_optim_config(optim_config: dict, optimizer_config_kwargs: dict) -> OptimizerConfig:
    # megatron-core only recognizes 'adam' / 'sgd' as standard optimizers (anything
    # else routes to `_get_megatron_emerging_optimizer`, which raises
    # `ValueError: Unsupported emerging optimizer: AdamW`). megatron's 'adam' IS
    # AdamW (decoupled weight decay via weight_decay), so normalize the common
    # HF-style names to 'adam'.
    _optim_name = str(optim_config.get("optimizer", "adam")).lower()
    if _optim_name == "adamw":
        _optim_name = "adam"
    if _optim_name == "muonh":
        _optim_name = _GRUG_MUONH_KEY
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

    if _optim_name == _GRUG_MUONH_KEY:
        betas = tuple(float(value) for value in optim_config.get("adam_betas", DEFAULT_BETAS))
        if len(betas) != 2:
            raise ValueError("MuonH adam_betas must contain two values")
        extra = optim_config.get("optimizer_kwargs", {})
        if not isinstance(extra, Mapping):
            raise TypeError("MuonH optimizer_kwargs must be a mapping")
        optim_args.update(
            adam_beta1=betas[0],
            adam_beta2=betas[1],
            adam_eps=float(extra.get("epsilon", DEFAULT_EPSILON)),
        )

    optim_args.update(optimizer_config_kwargs)

    config = OptimizerConfig(**optim_args)
    if _optim_name == _GRUG_MUONH_KEY:
        config._grug_muonh_kwargs = _grug_muonh_kwargs(optim_config, config)
    return config


def get_megatron_optimizer(
    model,
    config: OptimizerConfig,
    no_weight_decay_cond=None,
    scale_lr_cond=None,
    lr_mult=1.0,
):
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
    if config.optimizer == _GRUG_MUONH_KEY:
        _register_grug_muonh()
        first_chunk = model[0]
        base_model = first_chunk.module if isinstance(first_chunk, DistributedDataParallel) else first_chunk
        model_config = base_model.config
        if model_config.num_attention_heads % model_config.num_query_groups:
            raise ValueError("Grug MuonH requires whole query heads per key/value group")
        config._grug_muonh_kwargs.update(
            qkv_num_query_groups=model_config.num_query_groups,
            qkv_heads_per_group=model_config.num_attention_heads // model_config.num_query_groups,
            qkv_head_dim=model_config.kv_channels,
            tensor_model_parallel_size=model_config.tensor_model_parallel_size,
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

    scheduler_cls = (
        _GrugMuonHParamScheduler
        if any(group.get(ROUTE_KEY) == ADAM_ROUTE for group in optimizer.param_groups)
        else OptimizerParamScheduler
    )
    opt_param_scheduler = scheduler_cls(
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
