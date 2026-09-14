"""Checksummed tensor manifests and bounded collective transport."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any

import torch


TENSOR_TRANSFER_FORMAT = "marinskyrl-tensor-transfer"
TENSOR_TRANSFER_VERSION = 1
DEFAULT_STAGING_BYTES = 256 * 1024 * 1024

_DTYPES_BY_NAME = {
    str(dtype): dtype
    for dtype in (
        torch.bool,
        torch.int8,
        torch.uint8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    )
}


def _group_rank(group: torch.distributed.ProcessGroup) -> int:
    """Read the standalone group's rank without consulting another default world."""
    rank = group.rank()
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
        raise RuntimeError(f"Tensor transfer group returned an invalid rank: {rank!r}")
    return rank


def _group_size(group: torch.distributed.ProcessGroup) -> int:
    size = group.size()
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise RuntimeError(f"Tensor transfer group returned an invalid size: {size!r}")
    return size


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().to(device="cpu").contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy()).hexdigest()


@dataclass(frozen=True)
class TensorTransferEntry:
    name: str
    shape: tuple[int, ...]
    dtype: str
    numel: int
    bytes: int
    sha256: str

    @classmethod
    def from_tensor(cls, name: str, tensor: torch.Tensor) -> TensorTransferEntry:
        if not name:
            raise ValueError("Tensor transfer names must be nonempty")
        value = tensor.detach()
        return cls(
            name=name,
            shape=tuple(value.shape),
            dtype=str(value.dtype),
            numel=value.numel(),
            bytes=value.numel() * value.element_size(),
            sha256=_tensor_sha256(value),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> TensorTransferEntry:
        expected = {"name", "shape", "dtype", "numel", "bytes", "sha256"}
        if set(value) != expected:
            raise ValueError(f"Tensor transfer entry fields must be {sorted(expected)}")
        name = value["name"]
        shape = value["shape"]
        dtype = value["dtype"]
        numel = value["numel"]
        byte_count = value["bytes"]
        digest = value["sha256"]
        if not isinstance(name, str) or not name:
            raise ValueError("Tensor transfer names must be nonempty")
        if not isinstance(shape, (list, tuple)) or any(
            isinstance(size, bool) or not isinstance(size, int) or size < 0 for size in shape
        ):
            raise ValueError(f"Tensor transfer shape is invalid for {name!r}")
        if dtype not in _DTYPES_BY_NAME:
            raise ValueError(f"Tensor transfer dtype is unsupported for {name!r}: {dtype!r}")
        if isinstance(numel, bool) or not isinstance(numel, int) or numel < 0:
            raise ValueError(f"Tensor transfer numel is invalid for {name!r}")
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
            raise ValueError(f"Tensor transfer byte count is invalid for {name!r}")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"Tensor transfer digest is invalid for {name!r}")
        resolved_shape = tuple(shape)
        expected_numel = 1
        for size in resolved_shape:
            expected_numel *= size
        element_size = torch.empty((), dtype=_DTYPES_BY_NAME[dtype]).element_size()
        if numel != expected_numel or byte_count != numel * element_size:
            raise ValueError(f"Tensor transfer size metadata is inconsistent for {name!r}")
        return cls(name=name, shape=resolved_shape, dtype=dtype, numel=numel, bytes=byte_count, sha256=digest)

    @property
    def torch_dtype(self) -> torch.dtype:
        return _DTYPES_BY_NAME[self.dtype]

    def to_mapping(self) -> dict[str, Any]:
        value = asdict(self)
        value["shape"] = list(self.shape)
        return value

    def validate_tensor(self, tensor: torch.Tensor) -> None:
        if tuple(tensor.shape) != self.shape or tensor.dtype != self.torch_dtype:
            raise ValueError(f"Tensor transfer metadata mismatch for {self.name!r}")
        if _tensor_sha256(tensor) != self.sha256:
            raise ValueError(f"Tensor transfer digest mismatch for {self.name!r}")


@dataclass(frozen=True)
class TensorTransferOperation:
    key: str
    source_rank: int
    tensor: TensorTransferEntry

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> TensorTransferOperation:
        if set(value) != {"key", "source_rank", "tensor"}:
            raise ValueError("Tensor transfer operation has unsupported fields")
        key = value["key"]
        source_rank = value["source_rank"]
        if not isinstance(key, str) or not key:
            raise ValueError("Tensor transfer operation key must be nonempty")
        if isinstance(source_rank, bool) or not isinstance(source_rank, int) or source_rank < 0:
            raise ValueError("Tensor transfer operation source_rank must be nonnegative")
        tensor = value["tensor"]
        if not isinstance(tensor, Mapping):
            raise ValueError("Tensor transfer operation tensor must be a mapping")
        return cls(key=key, source_rank=source_rank, tensor=TensorTransferEntry.from_mapping(tensor))

    def to_mapping(self) -> dict[str, Any]:
        return {"key": self.key, "source_rank": self.source_rank, "tensor": self.tensor.to_mapping()}


def _payload_digest(entries: tuple[TensorTransferEntry, ...]) -> str:
    payload = json.dumps(
        [entry.to_mapping() for entry in entries],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class TensorTransferManifest:
    transfer_id: str
    revision: str
    source_weights_sha256: str
    total_bytes: int
    payload_sha256: str
    tensors: tuple[TensorTransferEntry, ...]

    @classmethod
    def from_tensors(
        cls,
        *,
        transfer_id: str,
        revision: str,
        source_weights_sha256: str,
        tensors: Mapping[str, torch.Tensor],
    ) -> TensorTransferManifest:
        if not transfer_id or not revision or not source_weights_sha256:
            raise ValueError("Tensor transfer identity fields must be nonempty")
        if not tensors:
            raise ValueError("Tensor transfer must contain at least one tensor")
        entries = tuple(TensorTransferEntry.from_tensor(name, tensors[name]) for name in sorted(tensors))
        return cls(
            transfer_id=transfer_id,
            revision=revision,
            source_weights_sha256=source_weights_sha256,
            total_bytes=sum(entry.bytes for entry in entries),
            payload_sha256=_payload_digest(entries),
            tensors=entries,
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> TensorTransferManifest:
        expected = {
            "format",
            "format_version",
            "transfer_id",
            "revision",
            "source_weights_sha256",
            "total_bytes",
            "payload_sha256",
            "tensors",
        }
        if set(value) != expected:
            raise ValueError(f"Tensor transfer manifest fields must be {sorted(expected)}")
        if value["format"] != TENSOR_TRANSFER_FORMAT or value["format_version"] != TENSOR_TRANSFER_VERSION:
            raise ValueError("Unsupported tensor transfer manifest")
        entries_value = value["tensors"]
        if not isinstance(entries_value, list) or not entries_value:
            raise ValueError("Tensor transfer manifest must contain tensors")
        entries = tuple(TensorTransferEntry.from_mapping(entry) for entry in entries_value)
        names = [entry.name for entry in entries]
        if names != sorted(names) or len(names) != len(set(names)):
            raise ValueError("Tensor transfer entries must have unique, sorted names")
        for field_name in ("transfer_id", "revision", "source_weights_sha256", "payload_sha256"):
            if not isinstance(value[field_name], str) or not value[field_name]:
                raise ValueError(f"Tensor transfer {field_name} must be nonempty")
        total_bytes = value["total_bytes"]
        if isinstance(total_bytes, bool) or not isinstance(total_bytes, int) or total_bytes < 0:
            raise ValueError("Tensor transfer total_bytes must be a nonnegative integer")
        if total_bytes != sum(entry.bytes for entry in entries):
            raise ValueError("Tensor transfer byte count does not match its entries")
        if value["payload_sha256"] != _payload_digest(entries):
            raise ValueError("Tensor transfer payload digest does not match its entries")
        return cls(
            transfer_id=value["transfer_id"],
            revision=value["revision"],
            source_weights_sha256=value["source_weights_sha256"],
            total_bytes=total_bytes,
            payload_sha256=value["payload_sha256"],
            tensors=entries,
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "format": TENSOR_TRANSFER_FORMAT,
            "format_version": TENSOR_TRANSFER_VERSION,
            "transfer_id": self.transfer_id,
            "revision": self.revision,
            "source_weights_sha256": self.source_weights_sha256,
            "total_bytes": self.total_bytes,
            "payload_sha256": self.payload_sha256,
            "tensors": [entry.to_mapping() for entry in self.tensors],
        }

    def validate_tensors(self, tensors: Mapping[str, torch.Tensor]) -> None:
        if set(tensors) != {entry.name for entry in self.tensors}:
            raise ValueError("Tensor transfer payload names do not match its manifest")
        for entry in self.tensors:
            entry.validate_tensor(tensors[entry.name])


def broadcast_tensor_payload(
    manifest_value: Mapping[str, Any],
    *,
    tensors: Mapping[str, torch.Tensor] | None,
    group: torch.distributed.ProcessGroup,
    source_rank: int = 0,
    device: torch.device | None = None,
    staging_bytes: int = DEFAULT_STAGING_BYTES,
) -> dict[str, torch.Tensor] | None:
    """Broadcast an ordered payload with bounded device staging.

    Every group rank must call this function with the same manifest. Only the
    source supplies ``tensors``; receivers get pinned CPU tensors.
    """
    manifest = TensorTransferManifest.from_mapping(manifest_value)
    if isinstance(staging_bytes, bool) or not isinstance(staging_bytes, int) or staging_bytes <= 0:
        raise ValueError("Tensor transfer staging_bytes must be a positive integer")
    rank = _group_rank(group)
    group_size = _group_size(group)
    if isinstance(source_rank, bool) or not isinstance(source_rank, int) or not 0 <= source_rank < group_size:
        raise ValueError(f"Tensor transfer source rank is outside the group: {source_rank!r}")
    is_source = rank == source_rank
    if is_source:
        if tensors is None:
            raise ValueError("Tensor transfer source did not provide tensors")
        manifest.validate_tensors(tensors)
    elif tensors is not None:
        raise ValueError("Tensor transfer receivers must not provide tensors")
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())

    received: dict[str, torch.Tensor] | None = None
    if not is_source:
        received = {
            entry.name: torch.empty(
                entry.shape, dtype=entry.torch_dtype, device="cpu", pin_memory=device.type == "cuda"
            )
            for entry in manifest.tensors
        }

    stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
    stream_context = torch.cuda.stream(stream) if stream is not None else torch.no_grad()
    staging: dict[torch.dtype, torch.Tensor] = {}
    with stream_context:
        for entry in manifest.tensors:
            max_elements = max(1, staging_bytes // torch.empty((), dtype=entry.torch_dtype).element_size())
            buffer_elements = min(max_elements, max(1, entry.numel))
            buffer = staging.get(entry.torch_dtype)
            if buffer is None or buffer.numel() < buffer_elements:
                buffer = torch.empty(buffer_elements, dtype=entry.torch_dtype, device=device)
                staging[entry.torch_dtype] = buffer
            source_flat = None if tensors is None else tensors[entry.name].detach().contiguous().view(-1)
            destination_flat = None if received is None else received[entry.name].view(-1)
            for offset in range(0, entry.numel, max_elements):
                length = min(max_elements, entry.numel - offset)
                chunk = buffer[:length]
                if source_flat is not None:
                    chunk.copy_(source_flat[offset : offset + length], non_blocking=device.type == "cuda")
                torch.distributed.broadcast(chunk, src=source_rank, group=group)
                if destination_flat is not None:
                    destination_flat[offset : offset + length].copy_(chunk, non_blocking=device.type == "cuda")
    if stream is not None:
        stream.synchronize()
    if received is not None:
        manifest.validate_tensors(received)
    return received


def transfer_tensor_operations(
    operation_values: Sequence[Mapping[str, Any]],
    *,
    tensor_provider: Callable[[TensorTransferOperation], torch.Tensor] | None,
    group: torch.distributed.ProcessGroup,
    receiver_rank: int = 0,
    device: torch.device | None = None,
    staging_bytes: int = DEFAULT_STAGING_BYTES,
) -> dict[str, torch.Tensor] | None:
    """Move a canonical multi-source operation log to one receiver over P2P.

    All ranks join the two fences. A source participates only in its ordered
    sends; the receiver performs every ordered receive. Other ranks never
    allocate payload storage.
    """
    operations = tuple(TensorTransferOperation.from_mapping(value) for value in operation_values)
    keys = [operation.key for operation in operations]
    if not operations or len(keys) != len(set(keys)):
        raise ValueError("Tensor transfer operations must have unique keys")
    if isinstance(staging_bytes, bool) or not isinstance(staging_bytes, int) or staging_bytes <= 0:
        raise ValueError("Tensor transfer staging_bytes must be a positive integer")
    rank = _group_rank(group)
    group_size = _group_size(group)
    if isinstance(receiver_rank, bool) or not isinstance(receiver_rank, int) or not 0 <= receiver_rank < group_size:
        raise ValueError(f"Tensor transfer receiver rank is outside the group: {receiver_rank!r}")
    invalid_source_ranks = {
        operation.source_rank
        for operation in operations
        if operation.source_rank == receiver_rank or not 0 <= operation.source_rank < group_size
    }
    if invalid_source_ranks:
        raise ValueError(f"Tensor transfer operations have invalid source ranks: {sorted(invalid_source_ranks)}")
    if rank == receiver_rank and tensor_provider is not None:
        raise ValueError("Tensor transfer receiver must not provide source tensors")
    if rank != receiver_rank and any(operation.source_rank == rank for operation in operations):
        if tensor_provider is None:
            raise ValueError(f"Tensor transfer source rank {rank} did not provide tensors")
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())

    received: dict[str, torch.Tensor] | None = {} if rank == receiver_rank else None
    stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
    stream_context = torch.cuda.stream(stream) if stream is not None else torch.no_grad()
    staging: dict[torch.dtype, torch.Tensor] = {}
    torch.distributed.barrier(group=group)
    with stream_context:
        for operation in operations:
            if rank not in {receiver_rank, operation.source_rank}:
                continue
            entry = operation.tensor
            max_elements = max(1, staging_bytes // torch.empty((), dtype=entry.torch_dtype).element_size())
            buffer_elements = min(max_elements, max(1, entry.numel))
            buffer = staging.get(entry.torch_dtype)
            if buffer is None or buffer.numel() < buffer_elements:
                buffer = torch.empty(buffer_elements, dtype=entry.torch_dtype, device=device)
                staging[entry.torch_dtype] = buffer
            source_flat = None
            destination_flat = None
            if rank == operation.source_rank:
                assert tensor_provider is not None
                source = tensor_provider(operation).detach().contiguous()
                entry.validate_tensor(source)
                source_flat = source.view(-1)
            else:
                assert received is not None
                destination = torch.empty(
                    entry.shape,
                    dtype=entry.torch_dtype,
                    device="cpu",
                    pin_memory=device.type == "cuda",
                )
                received[operation.key] = destination
                destination_flat = destination.view(-1)
            for offset in range(0, entry.numel, max_elements):
                length = min(max_elements, entry.numel - offset)
                chunk = buffer[:length]
                if source_flat is not None:
                    chunk.copy_(source_flat[offset : offset + length], non_blocking=device.type == "cuda")
                    torch.distributed.send(chunk, dst=receiver_rank, group=group)
                else:
                    torch.distributed.recv(chunk, src=operation.source_rank, group=group)
                    assert destination_flat is not None
                    destination_flat[offset : offset + length].copy_(chunk, non_blocking=device.type == "cuda")
    if stream is not None:
        stream.synchronize()
    torch.distributed.barrier(group=group)
    if received is not None:
        for operation in operations:
            operation.tensor.validate_tensor(received[operation.key])
    return received
