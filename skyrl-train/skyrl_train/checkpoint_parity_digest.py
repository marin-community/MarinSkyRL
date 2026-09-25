"""Exact, bounded-memory rank-state fingerprints for opt-in checkpoint parity."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
import hashlib
import json
from typing import Any

import numpy as np
import torch


_DIGEST_CHUNK_BYTES = 64 * 1024 * 1024


def digest_value(value: Any) -> dict[str, int | str]:
    """Fingerprint exact state without sending tensor contents through Ray.

    Tensor bytes are copied to CPU in bounded chunks. The digest includes dtype,
    shape, structure, and scalar values. MCore sharded wrappers contribute their
    rank-local data, which is sufficient for same-geometry numerical parity.
    """
    digest = hashlib.sha256()
    tensor_count = 0
    tensor_bytes = 0

    def record(tag: str, payload: str | bytes) -> None:
        data = payload.encode("utf-8") if isinstance(payload, str) else payload
        digest.update(tag.encode("ascii"))
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)

    def visit(item: Any) -> None:
        nonlocal tensor_count, tensor_bytes
        if isinstance(item, torch.Tensor):
            tensor_count += 1
            record("tensor", json.dumps([str(item.dtype), list(item.shape)], separators=(",", ":")))
            raw = item.detach().contiguous().reshape(-1).view(torch.uint8)
            tensor_bytes += raw.numel()
            record("data_length", str(raw.numel()))
            for offset in range(0, raw.numel(), _DIGEST_CHUNK_BYTES):
                chunk = raw[offset : offset + _DIGEST_CHUNK_BYTES].cpu()
                digest.update(chunk.numpy().tobytes())
            return
        if isinstance(item, np.ndarray):
            visit(torch.from_numpy(np.ascontiguousarray(item)))
            return
        if isinstance(item, np.generic):
            visit(item.item())
            return
        if (
            type(item).__module__.startswith("megatron.core.dist_checkpointing.")
            and type(item).__name__ == "LocalNonpersistentObject"
        ):
            record("nonpersistent", "")
            return
        if type(item).__module__.startswith("megatron.core.dist_checkpointing.") and hasattr(item, "data"):
            visit(item.data)
            return
        if isinstance(item, Mapping):
            record("mapping", str(len(item)))
            for key in sorted(item, key=lambda key: (type(key).__name__, repr(key))):
                visit(key)
                visit(item[key])
            return
        if isinstance(item, (tuple, list)):
            record(type(item).__name__, str(len(item)))
            for child in item:
                visit(child)
            return
        if is_dataclass(item) and not isinstance(item, type):
            visit({field.name: getattr(item, field.name) for field in fields(item)})
            return
        if isinstance(item, (str, bytes, int, float, bool, type(None))):
            record(type(item).__name__, repr(item))
            return
        if isinstance(item, (torch.dtype, torch.device, np.dtype)):
            record(type(item).__name__, str(item))
            return
        raise TypeError(f"Cannot digest checkpoint-parity state of type {type(item).__module__}.{type(item).__name__}")

    visit(value)
    return {"sha256": digest.hexdigest(), "tensor_count": tensor_count, "tensor_bytes": tensor_bytes}
