"""Rank-local, lossless state snapshots for opt-in checkpoint replay tests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from typing import Any

import numpy as np
import torch


def snapshot_value(value: Any) -> Any:
    """Copy all numerical leaves to CPU without retaining live model references.

    Megatron's sharded wrappers carry local tensor data plus placement metadata.
    The same rank and geometry are compared on both sides of the replay, so the
    local data is the numerical state that must match. Unknown types fail closed.
    """
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(np.array(value, copy=True))
    if isinstance(value, np.generic):
        return value.item()
    if (
        type(value).__module__.startswith("megatron.core.dist_checkpointing.")
        and type(value).__name__ == "LocalNonpersistentObject"
    ):
        return "<intentionally nonpersistent MCore object>"
    if type(value).__module__.startswith("megatron.core.dist_checkpointing.") and hasattr(value, "data"):
        return snapshot_value(value.data)
    if isinstance(value, Mapping):
        return {key: snapshot_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(snapshot_value(item) for item in value)
    if isinstance(value, list):
        return [snapshot_value(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: snapshot_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, (str, bytes, int, float, bool, type(None))):
        return value
    if isinstance(value, (torch.dtype, torch.device, np.dtype)):
        return str(value)
    raise TypeError(f"Cannot snapshot checkpoint-parity state of type {type(value).__module__}.{type(value).__name__}")


def assert_state_equal(expected: Any, actual: Any, path: str = "state") -> None:
    """Require exact dtype, shape, values, and tree structure with useful errors."""
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor):
            raise AssertionError(f"{path}: expected tensor, got {type(actual).__name__}")
        if expected.dtype != actual.dtype or expected.shape != actual.shape:
            raise AssertionError(
                f"{path}: dtype/shape {expected.dtype}/{tuple(expected.shape)} != {actual.dtype}/{tuple(actual.shape)}"
            )
        if not torch.equal(expected, actual):
            difference = expected.to(torch.float64) - actual.to(torch.float64)
            raise AssertionError(
                f"{path}: {torch.count_nonzero(difference).item()} elements differ; "
                f"max_abs={difference.abs().max().item():.9g}, "
                f"mean_abs={difference.abs().mean().item():.9g}"
            )
        return
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or expected.keys() != actual.keys():
            actual_keys = actual.keys() if isinstance(actual, Mapping) else type(actual).__name__
            raise AssertionError(f"{path}: keys differ: {expected.keys()} != {actual_keys}")
        for key in expected:
            assert_state_equal(expected[key], actual[key], f"{path}[{key!r}]")
        return
    if isinstance(expected, (tuple, list)):
        if type(expected) is not type(actual) or len(expected) != len(actual):
            raise AssertionError(f"{path}: sequence type or length differs")
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual, strict=True)):
            assert_state_equal(expected_item, actual_item, f"{path}[{index}]")
        return
    if type(expected) is not type(actual) or expected != actual:
        raise AssertionError(f"{path}: {expected!r} != {actual!r}")


def count_tensors(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return 1
    if isinstance(value, Mapping):
        return sum(count_tensors(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(count_tensors(item) for item in value)
    return 0
