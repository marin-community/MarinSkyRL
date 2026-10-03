"""What formed each bf16 tensor a Grug norm will read, handed from the module that formed it to that norm, and the
values a checkpoint unit's first forward keeps for its recompute.

Entries match by tensor object or by a live tensor with the same storage, offset and shape, since Megatron's
``make_viewless_tensor`` replaces a view entering a transformer block with a new tensor object on the same storage, and
a recompute runs on detached copies of the unit's inputs. Entries hold weak references, so a tensor that reuses a freed
tensor's ``id()`` never receives the freed tensor's entry.
"""

from __future__ import annotations

import weakref
from typing import Protocol

import torch


class HandOff(Protocol):
    """What formed a tensor that a Grug norm reads."""

    def statistic(self) -> torch.Tensor:
        """The norm's per-row sum of squares of the unrounded value, as compiled vLLM computes it."""
        ...


_HAND_OFFS: dict[int, tuple[weakref.ref, HandOff]] = {}
# Values kept for recompute, keyed by the id of the module that computed them, each with its checkpoint unit's input.
_KEPT_FOR_RECOMPUTE: dict[int, list[tuple[weakref.ref, torch.Tensor]]] = {}
# Whether a backward, and so the recompute of each checkpoint unit, follows the running forward.
_BACKWARD_FOLLOWS = False


def same_storage(left: torch.Tensor, right: torch.Tensor) -> bool:
    return (
        left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()
        and left.storage_offset() == right.storage_offset()
        and left.shape == right.shape
    )


def hand_off(receiver: torch.Tensor, parts: HandOff) -> None:
    """Register what formed ``receiver`` for the norm that will read it."""
    for key in [key for key, (ref, _) in _HAND_OFFS.items() if ref() is None]:
        del _HAND_OFFS[key]
    _HAND_OFFS[id(receiver)] = weakref.ref(receiver), parts


def take_hand_off(receiver: torch.Tensor) -> HandOff | None:
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
    """Drop every hand-off, so that the forward that starts begins with none."""
    _HAND_OFFS.clear()


def expect_backward(follows: bool) -> None:
    """Whether the forward that starts is followed by a backward, whose recompute takes what the forward keeps."""
    global _BACKWARD_FOLLOWS
    _BACKWARD_FOLLOWS = follows


def keep_for_recompute(owner: torch.nn.Module, receiver: torch.Tensor, value: torch.Tensor) -> None:
    """Keep ``value``, computed by ``owner``, for the recompute of the checkpoint unit that ``receiver`` identifies; a
    forward without a backward recomputes nothing and keeps nothing."""
    if not _BACKWARD_FOLLOWS:
        return
    _KEPT_FOR_RECOMPUTE.setdefault(id(owner), []).append((weakref.ref(receiver), value))


def take_for_recompute(owner: torch.nn.Module, receiver: torch.Tensor) -> torch.Tensor:
    """The value ``owner`` kept for the checkpoint unit that ``receiver`` identifies."""
    entries = _KEPT_FOR_RECOMPUTE.get(id(owner), [])
    for index, (ref, value) in enumerate(entries):
        original = ref()
        if original is not None and same_storage(original, receiver):
            del entries[index]
            return value
    raise RuntimeError("a checkpoint unit's recompute found nothing kept by the unit's first forward")


def assert_recompute_drained() -> None:
    """Raise unless every value a checkpoint unit's first forward kept was taken by the unit's recompute."""
    kept = sum(len(entries) for entries in _KEPT_FOR_RECOMPUTE.values())
    if kept:
        raise RuntimeError(f"{kept} values kept by checkpoint units' first forwards were not taken by their recompute")
