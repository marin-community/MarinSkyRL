"""Disposable exact codecs for the routed expert-block experiment.

Positions address an existing flat destination view. Replacement values keep the source
dtype and bits; no arithmetic is applied to floating-point data. The receiver owns its
initial dense image and applies patches in place.
"""

from dataclasses import dataclass

import torch


INTEGER_VIEW = {2: torch.int16, 4: torch.int32}


@dataclass(frozen=True)
class Patch:
    encoding: str
    numel: int
    positions: torch.Tensor
    values: torch.Tensor

    @property
    def changed(self) -> int:
        return self.values.numel()

    @property
    def payload_bytes(self) -> int:
        return self.positions.numel() * self.positions.element_size() + self.values.numel() * self.values.element_size()


def bits(tensor: torch.Tensor) -> torch.Tensor:
    """View BF16 or FP32 values as signed integers without changing their bits."""
    if tensor.dtype not in (torch.bfloat16, torch.float32) or not tensor.is_contiguous():
        raise ValueError("Exact sparse encoding requires contiguous BF16 or FP32 tensors")
    return tensor.detach().view(-1).view(INTEGER_VIEW[tensor.element_size()])


def changed_mask(current: torch.Tensor, baseline: torch.Tensor) -> torch.Tensor:
    if current.shape != baseline.shape or current.dtype != baseline.dtype or current.device != baseline.device:
        raise ValueError("Source and baseline must have the same shape, dtype and device")
    return bits(current).ne(bits(baseline))


def pack_bitmap(mask: torch.Tensor) -> torch.Tensor:
    """Pack each run of eight flags with the lowest flat position in bit zero."""
    result = torch.zeros((mask.numel() + 7) // 8, dtype=torch.uint8, device=mask.device)
    for bit in range(8):
        part = mask[bit::8].to(torch.uint8)
        result[: part.numel()].bitwise_or_(part << bit)
    return result


def unpack_bitmap(bitmap: torch.Tensor, numel: int) -> torch.Tensor:
    if bitmap.dtype != torch.uint8 or bitmap.numel() != (numel + 7) // 8:
        raise ValueError("Bitmap length differs from the destination")
    mask = torch.empty(numel, dtype=torch.bool, device=bitmap.device)
    for bit in range(8):
        part = mask[bit::8]
        part.copy_(((bitmap[: part.numel()] >> bit) & 1).bool())
    return mask


def encode(current: torch.Tensor, baseline: torch.Tensor, encoding: str) -> Patch:
    mask = changed_mask(current, baseline)
    flat = current.detach().view(-1)
    if encoding == "indices":
        if flat.numel() >= 2**31:
            raise ValueError("A single indexed transfer must fit signed int32 positions")
        positions = mask.nonzero(as_tuple=False).view(-1).to(torch.int32)
        values = flat.index_select(0, positions.to(torch.int64))
    elif encoding == "bitmap":
        positions = pack_bitmap(mask)
        values = flat.masked_select(mask)
    else:
        raise ValueError(f"Unknown exact sparse encoding {encoding}")
    return Patch(encoding, flat.numel(), positions, values)


def apply(destination: torch.Tensor, patch: Patch) -> None:
    """Apply replacements into an already-bound receiver view."""
    if destination.numel() != patch.numel or destination.device != patch.values.device:
        raise ValueError("Patch and destination differ in length or device")
    if destination.dtype not in (patch.values.dtype, torch.float32):
        raise ValueError("Destination dtype differs from the source")
    flat = destination.view(-1)
    values = patch.values.to(destination.dtype)
    if patch.encoding == "indices":
        if patch.positions.dtype != torch.int32 or patch.positions.numel() != patch.changed:
            raise ValueError("Indexed patch has invalid positions")
        flat.index_copy_(0, patch.positions.to(torch.int64), values)
    elif patch.encoding == "bitmap":
        mask = unpack_bitmap(patch.positions, patch.numel)
        if int(mask.sum().item()) != patch.changed:
            raise ValueError("Bitmap population differs from packed values")
        flat.masked_scatter_(mask, values)
    else:
        raise ValueError(f"Unknown exact sparse encoding {patch.encoding}")
