"""FP16 gradient storage with BF16 parameters on ordinary MCore 0.18 DDP."""

from dataclasses import replace
from importlib.metadata import version
import math

import torch


def validate_fp16_gradient_config(config, *, bf16: bool) -> None:
    """Reject combinations outside the dense, synchronous-reduction qualification."""
    enabled = config.get("fp16_grad_reduce", False)
    if type(enabled) is not bool:
        raise ValueError("fp16_grad_reduce must be a boolean")
    if not enabled:
        return
    if not bf16:
        raise ValueError("FP16 gradient reduction requires BF16 model parameters")
    for key in ("pipeline_model_parallel_size", "expert_model_parallel_size", "context_parallel_size"):
        if config.get(key, 1) != 1:
            raise ValueError(f"FP16 gradient reduction requires {key}=1")
    ddp = config.get("ddp_config", {})
    if ddp.get("grad_reduce_in_fp32", True):
        raise ValueError("FP16 gradient reduction requires grad_reduce_in_fp32=false")
    for key in (
        "overlap_grad_reduce",
        "overlap_param_gather",
        "nccl_ub",
        "reuse_grad_buf_for_mxfp8_param_ag",
        "fp8_param_gather",
        "reduce_scatter_with_fp32_accumulation",
        "param_name_patterns_for_fp32_local_accumulation",
        "check_for_nan_in_grad",
        "check_for_large_grads",
    ):
        if ddp.get(key, False):
            raise ValueError(f"FP16 gradient reduction does not support {key}")
    if ddp.get("num_distributed_optimizer_instances", 1) != 1:
        raise ValueError("FP16 gradient reduction requires one distributed optimizer instance")
    optimizer = config.get("optimizer_config_kwargs", {})
    for key in ("use_precision_aware_optimizer", "optimizer_cpu_offload", "optimizer_cuda_graph"):
        if optimizer.get(key, False):
            raise ValueError(f"FP16 gradient reduction requires native Adam without {key}")
    for key in (
        "fp16",
        "bf16",
        "params_dtype",
        "loss_scale",
        "initial_loss_scale",
        "min_loss_scale",
        "loss_scale_window",
        "hysteresis",
    ):
        if key in optimizer:
            raise ValueError(f"Configure gradient loss scaling through dynamic_loss_scale, not optimizer {key}")
    scale = config["dynamic_loss_scale"]
    for key in ("initial_scale", "min_scale"):
        if not math.isfinite(scale[key]) or scale[key] <= 0:
            raise ValueError(f"dynamic_loss_scale.{key} must be finite and positive")
    if scale["initial_scale"] < scale["min_scale"]:
        raise ValueError("Initial loss scale must be at least min_scale")
    for key in ("growth_interval", "hysteresis"):
        if type(scale[key]) is not int or scale[key] < 1:
            raise ValueError(f"dynamic_loss_scale.{key} must be a positive integer")


def fp16_optimizer_overrides(config) -> dict:
    """Select native MCore dynamic scaling without changing the model provider dtype."""
    scale = config["dynamic_loss_scale"]
    return {
        # MCore's fp16 flag enables DynamicGradScaler. Keep bf16 true as well: it
        # selects the existing BF16 parameter-copy path (no FP16 multi-tensor kernel).
        # These optimizer flags do not configure the BF16 model provider.
        "fp16": True,
        "bf16": True,
        "params_dtype": torch.bfloat16,
        "loss_scale": None,
        "initial_loss_scale": scale["initial_scale"],
        "min_loss_scale": scale["min_scale"],
        "loss_scale_window": scale["growth_interval"],
        "hysteresis": scale["hysteresis"],
    }


def use_fp16_gradient_buffers(model_chunks: list) -> list:
    """Bridge post-wrap hook: retarget zeroed gradient views before building the optimizer.

    BF16 and FP16 both occupy two bytes, so this changes neither allocation nor offsets.
    Parameter data never changes. MCore buckets and autograd hooks read these tensor views
    at backward time; the optimizer constructs its typed range maps after this hook.
    """
    if version("megatron-core") != "0.18.0":
        raise ValueError("FP16 gradient buffer layout requires qualified MCore 0.18.0")
    for chunk in model_chunks:
        if chunk.expert_parallel_buffers:
            raise ValueError("FP16 gradient buffers are only qualified for dense models")
        if not chunk.buffers:
            raise ValueError("FP16 gradient reduction requires allocated DDP buffers")
        for buffer in chunk.buffers:
            if buffer.param_dtype != torch.bfloat16 or buffer.grad_dtype != torch.bfloat16:
                raise ValueError("FP16 gradient conversion requires BF16 parameters and initial BF16 gradients")
            if buffer.extra_main_grads or buffer.nccl_ub:
                raise ValueError("FP16 gradient conversion cannot retarget promoted or registered buffers")
            if torch.count_nonzero(buffer.grad_data).item():
                raise ValueError("FP16 gradient conversion must precede the first backward")
            buffer.grad_data = buffer.grad_data.view(torch.float16)
            buffer.grad_dtype = torch.float16
            for bucket in buffer.buckets:
                bucket.grad_data = bucket.grad_data.view(torch.float16)
            for parameter in buffer.params:
                if parameter.dtype != torch.bfloat16 or parameter.main_grad.dtype != torch.bfloat16:
                    raise ValueError("FP16 gradient conversion cannot change parameter dtype")
                parameter.main_grad = parameter.main_grad.view(torch.float16)
        layout = chunk.full_param_layout
        chunk.full_param_layout = replace(
            layout, layouts={replace(key, grad_dtype=torch.float16): value for key, value in layout.layouts.items()}
        )
    return model_chunks


def bind_fp16_gradient_optimizer(optimizer) -> None:
    """Make native overflow and norm decisions unanimous across the policy worker world."""
    if len(optimizer.chained_optimizers) != 1:
        raise ValueError("FP16 gradient reduction requires one dense native optimizer")
    component = optimizer.chained_optimizers[0]
    if component.grad_scaler is None or component.config.loss_scale is not None:
        raise ValueError("FP16 gradient reduction requires the native dynamic loss scaler")
    component.grad_stats_parallel_group = torch.distributed.group.WORLD


def gradient_precision_metrics(model_chunks, optimizer, scale_before: float, successful: bool) -> dict[str, float]:
    """Return per-rank buffer bytes and native scaling decisions before gradients are cleared."""
    scale_after = float(optimizer.get_loss_scale().item())
    buffers = [buffer for chunk in model_chunks for buffer in chunk.buffers]
    if not buffers or any(buffer.grad_data.dtype != torch.float16 for buffer in buffers):
        raise ValueError("FP16 gradient telemetry requires actual FP16 buffers")
    overflow = any(component.found_inf.item() != 0 for component in optimizer.chained_optimizers)
    return {
        "gradient_precision/buffer_bytes": float(
            sum(buffer.grad_data.numel() * buffer.grad_data.element_size() for buffer in buffers)
        ),
        "gradient_precision/buffer_dtype_fp16": 1.0,
        "gradient_precision/loss_scale_before": scale_before,
        "gradient_precision/loss_scale_after": scale_after,
        "gradient_precision/scale_growth": float(scale_after > scale_before),
        "gradient_precision/scale_backoff": float(scale_after < scale_before),
        "gradient_precision/overflow": float(overflow),
        "gradient_precision/optimizer_skipped": float(not successful),
    }


def bind_fp16_gradient_loss(model_configs, optimizer) -> None:
    """Use the native scheduler's single backward-loss scaling point."""
    for config in model_configs:
        if config.grad_scale_func is not None:
            raise ValueError("FP16 gradient loss already has a scaling callback")
        config.grad_scale_func = optimizer.scale_loss
