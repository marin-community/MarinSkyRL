"""Bounded weight-sync manifests, including slices of stacked expert tensors.

Planning is CPU-only. Engine allocation and transport must separately qualify
headroom and the receiver's actual parameter layout before using this manifest.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import torch


_DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}
_SIZES = {"bfloat16": 2, "float32": 4}


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    wire_dtype: str
    split_experts: bool = False


@dataclass(frozen=True)
class ManifestEntry:
    hf_name: str
    full_shape: tuple[int, ...]
    shape: tuple[int, ...]
    wire_dtype: str
    numel: int
    bucket_id: int
    offset: int  # Bytes within the packed buffer.
    tensor_offset: int  # Elements within the complete HF tensor.
    expert_start: int | None

    @property
    def nbytes(self) -> int:
        return self.numel * _SIZES[self.wire_dtype]


@dataclass(frozen=True)
class PublicationManifest:
    bucket_bytes: int
    entries: tuple[ManifestEntry, ...]

    @property
    def manifest_id(self) -> str:
        canonical = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    @property
    def bucket_count(self) -> int:
        return self.entries[-1].bucket_id + 1 if self.entries else 0

    def bucket(self, bucket_id: int) -> tuple[ManifestEntry, ...]:
        if not 0 <= bucket_id < self.bucket_count:
            raise ValueError("Unknown bucket")
        return tuple(entry for entry in self.entries if entry.bucket_id == bucket_id)


def build_manifest(specs: Sequence[TensorSpec], bucket_bytes: int = 2**30) -> PublicationManifest:
    """Preserve HF order and expert identities without splitting any single expert.

    Complete bridge conversion groups must feed this plan. Passing incomplete
    expert task groups to the bridge exporter loses its pending grouped state.
    """
    if type(bucket_bytes) is not int or bucket_bytes <= 0:
        raise ValueError("bucket_bytes must be a positive integer")
    names = set()
    entries = []
    bucket_id, used = 0, 0
    for spec in specs:
        if not spec.name or spec.name in names:
            raise ValueError("Tensor names must be unique and nonempty")
        names.add(spec.name)
        if spec.wire_dtype not in _DTYPES or not spec.shape or any(type(d) is not int or d <= 0 for d in spec.shape):
            raise ValueError("Unsupported wire dtype or shape")
        if spec.split_experts and len(spec.shape) != 3:
            raise ValueError("Expert splitting requires a stacked rank-three tensor")
        width = _SIZES[spec.wire_dtype]
        unit_shape = spec.shape[1:] if spec.split_experts else spec.shape
        unit_numel = math.prod(unit_shape)
        unit_bytes = unit_numel * width
        if unit_bytes > bucket_bytes:
            raise ValueError("A whole tensor or single expert exceeds the bucket threshold")
        # Float32 entries are isolated; router biases must never enter a bf16 pack.
        separate = spec.wire_dtype == "float32"
        if separate and used:
            bucket_id, used = bucket_id + 1, 0
        units = spec.shape[0] if spec.split_experts else 1
        first = 0
        while first < units:
            available = (bucket_bytes - used) // unit_bytes
            if available == 0:
                bucket_id, used = bucket_id + 1, 0
                available = bucket_bytes // unit_bytes
            count = min(units - first, available)
            shape = (count, *unit_shape) if spec.split_experts else spec.shape
            entries.append(
                ManifestEntry(
                    spec.name,
                    spec.shape,
                    shape,
                    spec.wire_dtype,
                    count * unit_numel,
                    bucket_id,
                    used,
                    first * unit_numel,
                    first if spec.split_experts else None,
                )
            )
            used += count * unit_bytes
            first += count
        if separate:
            bucket_id, used = bucket_id + 1, 0
    return PublicationManifest(bucket_bytes, tuple(entries))


def parse_manifest(payload: Mapping[str, Any], expected_manifest_id: str) -> PublicationManifest:
    """Validate a transported manifest before the receiver allocates or writes.

    Rebuilding the deterministic plan checks complete tensor coverage, expert
    offsets, bucket bounds and dtype isolation, even if a malformed payload has
    an internally consistent digest. This function only handles CPU metadata.
    """
    if set(payload) != {"bucket_bytes", "entries"} or not payload["entries"]:
        raise ValueError("Expected a nonempty weight-sync manifest")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if hashlib.sha256(canonical.encode()).hexdigest() != expected_manifest_id:
        raise ValueError("Weight-sync manifest digest mismatch")
    required = set(ManifestEntry.__dataclass_fields__)
    specs = {}
    for entry in payload["entries"]:
        if not isinstance(entry, dict) or set(entry) != required:
            raise ValueError("Unexpected weight-sync manifest entry fields")
        name = entry["hf_name"]
        if not isinstance(name, str) or not name:
            raise ValueError("Manifest tensor name must be nonempty")
        if not isinstance(entry["full_shape"], (tuple, list)):
            raise ValueError("Manifest tensor shape must be a sequence")
        if name not in specs:
            specs[name] = TensorSpec(
                name, tuple(entry["full_shape"]), entry["wire_dtype"], entry["expert_start"] is not None
            )
    rebuilt = build_manifest(tuple(specs.values()), payload["bucket_bytes"])
    if json.dumps(asdict(rebuilt), sort_keys=True, separators=(",", ":")) != canonical:
        raise ValueError("Manifest slices do not match the complete canonical plan")
    return rebuilt


def pack_bucket(
    manifest: PublicationManifest, bucket_id: int, tensors: Mapping[str, torch.Tensor], buffer: torch.Tensor
) -> int:
    """Copy exact source slices; no implicit dtype conversion or allocation."""
    entries = manifest.bucket(bucket_id)
    nbytes = entries[-1].offset + entries[-1].nbytes
    if buffer.dtype != torch.uint8 or buffer.ndim != 1 or buffer.numel() < nbytes or not buffer.is_contiguous():
        raise ValueError("Invalid packed buffer")
    for entry in entries:
        source = tensors[entry.hf_name]
        if tuple(source.shape) != entry.full_shape or source.dtype != _DTYPES[entry.wire_dtype]:
            raise ValueError("Source shape/dtype differs from manifest")
        if source.device != buffer.device or not source.is_contiguous():
            raise ValueError("Source and buffer must be contiguous on the same device")
    for entry in entries:
        source = tensors[entry.hf_name].view(-1).narrow(0, entry.tensor_offset, entry.numel).view(torch.uint8)
        buffer.narrow(0, entry.offset, entry.nbytes).copy_(source)
    return nbytes


def unpack_bucket(
    manifest: PublicationManifest, bucket_id: int, buffer: torch.Tensor
) -> tuple[tuple[ManifestEntry, torch.Tensor], ...]:
    """Return storage-sharing views carrying their full-tensor/expert offsets."""
    entries = manifest.bucket(bucket_id)
    nbytes = entries[-1].offset + entries[-1].nbytes
    if buffer.dtype != torch.uint8 or buffer.ndim != 1 or buffer.numel() < nbytes or not buffer.is_contiguous():
        raise ValueError("Invalid packed buffer")
    return tuple(
        (entry, buffer.narrow(0, entry.offset, entry.nbytes).view(_DTYPES[entry.wire_dtype]).view(entry.shape))
        for entry in entries
    )
