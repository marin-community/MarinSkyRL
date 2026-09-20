"""Weight extractor interface for extracting weights from training backends."""

from abc import ABC, abstractmethod
from typing import Iterator

import torch

from skyrl_train.models.grug_moe import GRUG_MOE_MODEL_TYPE, is_grug_router_bias

from .base import WeightChunk


def weight_sync_dtype(model_type: str | None, name: str, default: torch.dtype) -> torch.dtype:
    """Return the wire dtype for one canonical HF state entry."""

    if is_grug_router_bias(model_type, name):
        return torch.float32
    return default


def is_weight_sync_dtype_compatible(
    model_type: str | None,
    name: str,
    wire_dtype: torch.dtype,
    model_dtype: torch.dtype,
) -> bool:
    """Return whether one state entry has the dtype required by its wire contract."""

    return wire_dtype == weight_sync_dtype(model_type, name, model_dtype)


def prepare_weight_sync_tensor(
    model_type: str,
    name: str,
    tensor: torch.Tensor,
    target_dtype: torch.dtype,
) -> torch.Tensor:
    """Cast an extracted tensor and stage replicated CUDA-only state."""

    if is_grug_router_bias(model_type, name):
        return tensor.to(
            device=torch.cuda.current_device(),
            dtype=target_dtype,
            non_blocking=True,
        )
    return tensor.to(target_dtype)


def allocate_cuda_ipc_buffer(numel: int, *, device: int, dtype: torch.dtype) -> torch.Tensor:
    """Allocate a shareable buffer while preserving the training allocator settings."""
    # Torch 2.13 exports expandable segments as fabric handles on GB200, but
    # unmapHandles reads those handles as integer file descriptors. Releasing
    # a shared segment then throws std::bad_variant_access. Only the wire buffer
    # needs ordinary cudaMalloc storage; model and optimizer allocations can
    # keep using expandable segments. Do not yield while changing this setting.
    settings = torch._C._accelerator_getAllocatorSettings()
    ipc_settings = ",".join(filter(None, (settings, "expandable_segments:False")))
    torch._C._accelerator_setAllocatorSettings(ipc_settings)
    try:
        return torch.empty(numel, device=device, dtype=dtype, requires_grad=False)
    finally:
        torch._C._accelerator_setAllocatorSettings(settings)


def validate_weight_sync_mode(model_type: str, *, fuse_weights: bool) -> None:
    """Reject transport modes that cannot preserve a model's state contract."""

    if model_type == GRUG_MOE_MODEL_TYPE and fuse_weights:
        raise ValueError("Grug weight sync does not support fused weights")


class WeightExtractor(ABC):
    """Extracts weights from training backend models.

    Subclasses implement backend-specific logic to extract model weights,
    handle sharding, and prepare them for transfer to inference engines.
    """

    @abstractmethod
    def extract_weights(self, dtype: torch.dtype) -> Iterator[WeightChunk]:
        """Extract weights from the model as WeightChunk objects.

        Implementations should:
        - Gather sharded weights into full tensors
        - Convert tensors to the specified dtype for inference
        - Ensure tensors are contiguous in memory
        - Optionally group related parameters (e.g., QKV for efficiency)

        Args:
            dtype: Target dtype for inference (e.g., torch.bfloat16, torch.float16)

        Yields:
            WeightChunk objects containing model parameters ready for transfer
        """
        ...
