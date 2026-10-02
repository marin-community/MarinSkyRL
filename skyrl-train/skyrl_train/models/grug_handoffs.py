"""What formed each bf16 tensor a Grug norm will read, handed from the module that formed it to that norm.

A layer's residual sum, the embedding gated norm's product and a pipeline stage's received statistic are registered on
the tensor the next norm reads (``hand_off``) and taken by that norm (``take_hand_off``). Each entry keeps a weak
reference to its tensor: under pipeline parallelism several micro-batches are in flight and a stage's last residual has
no reader, so a freed tensor's ``id()`` can come back for an unrelated tensor, which must not receive the entry.

Megatron's transformer block replaces a view entering it (the embedding gated norm's output under the vLLM numerics)
with a new tensor object on the same storage (``make_viewless_tensor``), so a norm whose input is not the registered
object takes the entry of a registered tensor with the same storage, offset and shape.
"""

from __future__ import annotations

import weakref
from typing import Any

import torch

_HAND_OFFS: dict[int, tuple[weakref.ref, Any]] = {}


def same_storage(left: torch.Tensor, right: torch.Tensor) -> bool:
    return (
        left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()
        and left.storage_offset() == right.storage_offset()
        and left.shape == right.shape
    )


def hand_off(receiver: torch.Tensor, parts: Any) -> None:
    """Register what formed ``receiver`` for the norm that will read it."""
    for key in [key for key, (ref, _) in _HAND_OFFS.items() if ref() is None]:
        del _HAND_OFFS[key]
    _HAND_OFFS[id(receiver)] = weakref.ref(receiver), parts


def take_hand_off(receiver: torch.Tensor) -> Any | None:
    """What formed ``receiver``, registered on it or on a tensor with the same storage; ``None`` when nothing was."""
    entry = _HAND_OFFS.pop(id(receiver), None)
    if entry is not None and entry[0]() is receiver:
        return entry[1]
    for key, (ref, parts) in list(_HAND_OFFS.items()):
        original = ref()
        if original is not None and same_storage(original, receiver):
            del _HAND_OFFS[key]
            return parts
    return None


def clear_hand_offs() -> None:
    """Drop every hand-off: each forward starts empty, as a stage's last residual has no reader."""
    _HAND_OFFS.clear()
