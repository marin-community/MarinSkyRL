from typing import Protocol

import torch


class HybridOptimizerState(Protocol):
    param_update_in_fp32: bool
    state: dict[torch.Tensor, dict[str, torch.Tensor]]
    param_to_inner_param: dict[torch.Tensor, torch.Tensor]


@torch.no_grad()
def restore_hybrid_master_params(optimizer: HybridOptimizerState) -> None:
    """Restore master weights for both FP32 and reduced-precision parameters."""
    if not optimizer.param_update_in_fp32:
        return
    # Megatron 0.18's param_to_fp32_param omits parameters already in FP32.
    # The inner map also covers their CPU-offloaded copies, which must receive
    # the saved master weights before the next optimizer update.
    for param, state in optimizer.state.items():
        optimizer.param_to_inner_param[param].copy_(state["master_param"])
