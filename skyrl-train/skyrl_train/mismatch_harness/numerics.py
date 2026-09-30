"""Byte-level comparison of trainer and vLLM tensors: byte-equal fraction and distance in units in the last place."""

from __future__ import annotations

from dataclasses import dataclass

import torch

_INTEGER_VIEW = {torch.bfloat16: torch.int16, torch.float16: torch.int16, torch.float32: torch.int32}


def ordered_bits(tensor: torch.Tensor) -> torch.Tensor:
    """Map floating-point bit patterns to int64 so adjacent representable values differ by one.

    Negative values are mirrored below zero, so -0.0 and +0.0 both map to 0 and the distance between
    the smallest positive and the smallest negative subnormal is two.
    """
    bits = tensor.contiguous().view(_INTEGER_VIEW[tensor.dtype]).to(torch.int64)
    width = torch.iinfo(_INTEGER_VIEW[tensor.dtype]).bits
    sign = 1 << (width - 1)
    return torch.where(bits < 0, -(bits + sign), bits)


def ulp_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Elementwise number of representable values between two same-dtype tensors."""
    if left.dtype != right.dtype or left.shape != right.shape:
        raise ValueError(f"cannot compare {left.dtype}{tuple(left.shape)} with {right.dtype}{tuple(right.shape)}")
    return (ordered_bits(left) - ordered_bits(right)).abs()


@dataclass(frozen=True)
class RegionStats:
    """Agreement of one tensor pair over the valid token rows."""

    elements: int
    byte_equal_fraction: float
    max_ulp: int
    mean_ulp: float
    rows_all_equal_fraction: float
    max_abs: float
    dtype: str

    def to_json(self) -> dict:
        return {
            "elements": self.elements,
            "byte_equal_fraction": self.byte_equal_fraction,
            "max_ulp": self.max_ulp,
            "mean_ulp": self.mean_ulp,
            "rows_all_equal_fraction": self.rows_all_equal_fraction,
            "max_abs": self.max_abs,
            "dtype": self.dtype,
        }


def compare(left: torch.Tensor, right: torch.Tensor) -> RegionStats:
    """Compare two tensors whose first dimension indexes tokens.

    NaNs compare equal only when their bit patterns are equal; a NaN on one side makes ``max_abs`` NaN.
    """
    distance = ulp_distance(left, right)
    rows = distance.reshape(distance.shape[0], -1) if distance.dim() > 0 else distance.reshape(1, 1)
    equal = distance == 0
    difference = (left.float() - right.float()).abs()
    return RegionStats(
        elements=distance.numel(),
        byte_equal_fraction=equal.float().mean().item() if distance.numel() else 1.0,
        max_ulp=int(distance.max().item()) if distance.numel() else 0,
        mean_ulp=distance.double().mean().item() if distance.numel() else 0.0,
        rows_all_equal_fraction=(rows == 0).all(dim=1).float().mean().item() if rows.numel() else 1.0,
        max_abs=difference.max().item() if difference.numel() else 0.0,
        dtype=str(left.dtype).removeprefix("torch."),
    )
