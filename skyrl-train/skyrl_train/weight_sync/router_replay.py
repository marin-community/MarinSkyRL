"""Bounded exact installed-byte proof for Grug's BF16-wire/FP32 router."""

import torch

from skyrl_train.weight_sync.byte_replay import ByteComparison, compare_installed_views


ROUTER_CONVERSION_ELEMENTS = 65536  # 256 KiB of FP32; the caller supplies bounded boolean scratch.


def compare_widened_router(source: torch.Tensor, installed: torch.Tensor, scratch: torch.Tensor) -> ByteComparison:
    if (
        source.dtype != torch.bfloat16
        or installed.dtype != torch.float32
        or source.shape != installed.shape
        or not source.numel()
        or not source.is_contiguous()
        or not installed.is_contiguous()
        or source.device != installed.device
        or scratch.device != source.device
        or scratch.dtype != torch.bool
        or scratch.ndim != 1
        or not scratch.is_contiguous()
        or not 0 < scratch.numel() <= 512 * 1024
    ):
        raise ValueError("Router replay requires matching BF16 source and FP32 installed views with bounded scratch")
    views = (source, installed, scratch)
    for index, left in enumerate(views):
        for right in views[index + 1 :]:
            if max(left.data_ptr(), right.data_ptr()) < min(
                left.data_ptr() + left.numel() * left.element_size(),
                right.data_ptr() + right.numel() * right.element_size(),
            ):
                raise ValueError("Router source, installed bytes and replay scratch must not overlap")
    source = source.view(-1)
    installed = installed.view(-1)
    compared = mismatches = 0
    for start in range(0, source.numel(), ROUTER_CONVERSION_ELEMENTS):
        count = min(ROUTER_CONVERSION_ELEMENTS, source.numel() - start)
        # Same numeric widening as the pinned loader's param.data.copy_(BF16).
        # Byte comparison covers every resulting FP32 byte, including low halves.
        converted = source.narrow(0, start, count).to(torch.float32)
        result = compare_installed_views(
            ((converted, installed.narrow(0, start, count)),), scratch, expected_bytes=count * 4
        )
        compared += result.compared_bytes
        mismatches += result.mismatches
        # Do not retain the old conversion while allocating the next chunk.
        del converted
    return ByteComparison(compared, mismatches)
