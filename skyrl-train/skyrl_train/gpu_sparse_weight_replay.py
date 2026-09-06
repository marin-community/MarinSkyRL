"""Opt-in exact GPU wire replay; no production publisher imports this module.

Raw integer views preserve signed zero and NaN payloads. Integrity uses resident
replay fixtures and full device comparisons, not a production checksum protocol.
Only a count and validation/ACK metadata cross through the host. Payload buffers
are fixed-capacity and reused; dense fallback is chosen by actual payload bytes.
"""

from dataclasses import dataclass
from enum import StrEnum
import time

import torch

from skyrl_train.sparse_weight_replay import MAX_CHUNK_BYTES


BLOCK_ELEMENTS = 1024
RAW_DTYPES = {2: torch.int16, 4: torch.int32}


class Encoding(StrEnum):
    DENSE = "dense"
    INDEX32 = "index32"
    BLOCK_LOCAL16 = "block_local16"
    NOOP = "noop"


@dataclass(frozen=True)
class PatchHeader:
    elements: int
    width: int
    base_version: int
    target_version: int
    encoding: Encoding
    entries: int

    @property
    def blocks(self) -> int:
        return (self.elements + BLOCK_ELEMENTS - 1) // BLOCK_ELEMENTS

    @property
    def payload_bytes(self) -> int:
        if self.encoding == Encoding.DENSE:
            return self.elements * self.width
        if self.encoding == Encoding.NOOP:
            return 0
        index_width = 4 if self.encoding == Encoding.INDEX32 else 2
        offsets = 4 * (self.blocks + 1) if self.encoding == Encoding.BLOCK_LOCAL16 else 0
        return self.entries * (index_width + self.width) + offsets

    def validate(self, *, elements: int, width: int, installed_version: int) -> None:
        values = (self.elements, self.width, self.base_version, self.target_version, self.entries)
        if any(type(value) is not int for value in values):
            raise ValueError("Header integers must be exact integers")
        if self.width not in RAW_DTYPES or not 0 < self.elements * self.width <= MAX_CHUNK_BYTES:
            raise ValueError("Wire chunk must use two/four-byte elements within 16 MiB")
        if (self.elements, self.width) != (elements, width):
            raise ValueError("Wire shape differs from resident baseline")
        if not 0 <= self.base_version < self.target_version or self.base_version != installed_version:
            raise ValueError("Publication base/version differs")
        if not isinstance(self.encoding, Encoding) or not 0 <= self.entries <= self.elements:
            raise ValueError("Unknown encoding or invalid entry count")
        if self.encoding == Encoding.DENSE and self.entries != self.elements:
            raise ValueError("Dense payload must cover every element")
        if (self.encoding == Encoding.NOOP) != (self.entries == 0):
            raise ValueError("Empty patches must use noop")


def packet_header(
    *, elements: int, width: int, changed: int, preferred: Encoding, base_version: int, target_version: int
) -> PatchHeader:
    """Choose exact replacement payloads, falling back before sparse expansion."""
    if preferred not in (Encoding.DENSE, Encoding.INDEX32, Encoding.BLOCK_LOCAL16):
        raise ValueError("Select dense, index32 or block_local16")
    if type(changed) is not int or not 0 <= changed <= elements:
        raise ValueError("Changed count exceeds chunk")
    encoding = Encoding.DENSE if preferred == Encoding.DENSE else (preferred if changed else Encoding.NOOP)
    header = PatchHeader(
        elements, width, base_version, target_version, encoding, elements if encoding == Encoding.DENSE else changed
    )
    if header.payload_bytes >= elements * width:
        header = PatchHeader(elements, width, base_version, target_version, Encoding.DENSE, elements)
    header.validate(elements=elements, width=width, installed_version=base_version)
    return header


def validate_ack(header: PatchHeader, ack: dict) -> None:
    expected = {"accepted": True, "base_version": header.base_version, "target_version": header.target_version}
    if ack != expected or any(type(ack.get(key)) is not type(value) for key, value in expected.items()):
        raise ValueError("Receiver ACK missing, rejected or from another version")


@dataclass
class ReplayBaseline:
    """A versioned raw-bit buffer; failed checks never install candidate bytes."""

    values: torch.Tensor
    version: int = 0

    def validate_base(self, header: PatchHeader, expected_base: torch.Tensor) -> None:
        if self.values.dtype not in RAW_DTYPES.values() or self.values.ndim != 1 or not self.values.is_contiguous():
            raise ValueError("Baseline must be a contiguous raw integer vector")
        header.validate(elements=self.values.numel(), width=self.values.element_size(), installed_version=self.version)
        if (
            expected_base.dtype != self.values.dtype
            or expected_base.device != self.values.device
            or not torch.equal(self.values, expected_base)
        ):
            raise ValueError("Resident baseline differs from replay fixture")

    def install(self, header: PatchHeader, staged: torch.Tensor, expected_target: torch.Tensor) -> torch.Tensor:
        """Verify the full staging buffer before swapping it into the receiver."""
        header.validate(elements=self.values.numel(), width=self.values.element_size(), installed_version=self.version)
        if (
            staged.dtype != self.values.dtype
            or staged.shape != self.values.shape
            or staged.device != self.values.device
            or not staged.is_contiguous()
            or staged.untyped_storage().data_ptr() == self.values.untyped_storage().data_ptr()
            or expected_target.dtype != self.values.dtype
            or expected_target.device != self.values.device
            or not torch.equal(staged, expected_target)
        ):
            raise ValueError("Reconstructed target differs from replay fixture")
        previous = self.values
        self.values, self.version = staged, header.target_version
        return previous

    def commit_after_ack(self, header: PatchHeader, target: torch.Tensor, ack: dict) -> None:
        """Advance the sender predecessor only after an exact successful ACK.

        A missing/bad ACK fails closed. Receiver and sender commit are not atomic;
        this benchmark aborts that session and makes no reconnect/retry claim.
        """
        header.validate(elements=self.values.numel(), width=self.values.element_size(), installed_version=self.version)
        validate_ack(header, ack)
        if target.dtype != self.values.dtype or target.shape != self.values.shape:
            raise ValueError("Sender target layout differs")
        self.values.copy_(target)
        self.version = header.target_version


class CudaPhases:
    """Record stream intervals without synchronizing each phase separately."""

    def __init__(self):
        self.events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []

    def start(self, name: str) -> torch.cuda.Event:
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def end(self, name: str, start: torch.cuda.Event) -> None:
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self.events.append((name, start, end))

    def seconds(self) -> dict[str, float]:
        # Caller has already synchronized completion before ending wall timing.
        result = {}
        for name, start, end in self.events:
            result[name] = result.get(name, 0.0) + start.elapsed_time(end) / 1000
        return result


class GpuWorkspace:
    """One bounded reusable scan/compact/apply workspace on a single CUDA device."""

    def __init__(self, elements: int, width: int, device: torch.device):
        packet_header(
            elements=elements, width=width, changed=0, preferred=Encoding.DENSE, base_version=0, target_version=1
        )
        if device.type != "cuda":
            raise ValueError("GPU replay workspace requires CUDA")
        # Optional dependency belongs to the already-pinned CUDA runtime only.
        from skyrl_train import gpu_sparse_weight_kernels as kernels

        self.kernels = kernels
        self.elements, self.width = elements, width
        self.blocks = (elements + BLOCK_ELEMENTS - 1) // BLOCK_ELEMENTS
        self.counts = torch.empty(self.blocks, dtype=torch.int32, device=device)
        self.offsets = torch.empty(self.blocks + 1, dtype=torch.int32, device=device)
        self.offsets[0] = 0
        self.indices32 = torch.empty(elements, dtype=torch.int32, device=device)
        self.indices16 = self.indices32.view(torch.int16)[:elements]
        self.values = torch.empty(elements, dtype=RAW_DTYPES[width], device=device)
        self.errors = torch.empty(self.blocks, dtype=torch.int32, device=device)
        self.host_count = torch.empty(1, dtype=torch.int32, pin_memory=True)

    def validate_vectors(self, *values: torch.Tensor) -> None:
        if any(
            value.shape != self.values.shape
            or value.dtype != self.values.dtype
            or value.device != self.values.device
            or not value.is_contiguous()
            for value in values
        ):
            raise ValueError("GPU wire vector differs from workspace layout")

    def pack(self, base: torch.Tensor, target: torch.Tensor, *, preferred: Encoding, version: int, phases: CudaPhases):
        """Scan and compact on device; synchronize only the four-byte total count."""
        self.validate_vectors(base, target)
        if preferred == Encoding.DENSE:
            return packet_header(
                elements=self.elements,
                width=self.width,
                changed=self.elements,
                preferred=preferred,
                base_version=version,
                target_version=version + 1,
            ), {"count_sync_seconds": 0.0, "count_d2h_bytes": 0}
        started = phases.start("compare_scan_gpu_seconds")
        self.kernels.count_changes[(self.blocks,)](base, target, self.counts, self.elements, BLOCK=BLOCK_ELEMENTS)
        torch.cumsum(self.counts, dim=0, dtype=torch.int32, out=self.offsets[1:])
        phases.end("compare_scan_gpu_seconds", started)
        started_wall = time.perf_counter()
        self.host_count.copy_(self.offsets[-1:], non_blocking=True)
        torch.cuda.current_stream().synchronize()
        changed = int(self.host_count.item())
        stats = {
            "count_sync_seconds": time.perf_counter() - started_wall,
            "count_d2h_bytes": 4,
            "changed_elements": changed,
        }
        header = packet_header(
            elements=self.elements,
            width=self.width,
            changed=changed,
            preferred=preferred,
            base_version=version,
            target_version=version + 1,
        )
        if header.encoding in (Encoding.INDEX32, Encoding.BLOCK_LOCAL16):
            started = phases.start("pack_gpu_seconds")
            indices = self.indices16 if header.encoding == Encoding.BLOCK_LOCAL16 else self.indices32
            self.kernels.pack_changes[(self.blocks,)](
                base,
                target,
                self.offsets,
                indices,
                self.values,
                self.elements,
                LOCAL=header.encoding == Encoding.BLOCK_LOCAL16,
                BLOCK=BLOCK_ELEMENTS,
            )
            phases.end("pack_gpu_seconds", started)
        return header, stats

    def payload(self, header: PatchHeader, dense: torch.Tensor) -> list[torch.Tensor]:
        """Expose resident byte views; NCCL has no native signed-int16 wire type."""
        header.validate(elements=self.elements, width=self.width, installed_version=header.base_version)
        self.validate_vectors(dense)
        if header.encoding == Encoding.DENSE:
            return [dense.view(torch.uint8)]
        if header.encoding == Encoding.NOOP:
            return []
        indices = self.indices16 if header.encoding == Encoding.BLOCK_LOCAL16 else self.indices32
        views = [indices[: header.entries].view(torch.uint8), self.values[: header.entries].view(torch.uint8)]
        return [self.offsets.view(torch.uint8), *views] if header.encoding == Encoding.BLOCK_LOCAL16 else views

    def apply(self, header: PatchHeader, base: torch.Tensor, staged: torch.Tensor, phases: CudaPhases) -> dict:
        """Validate sparse addressing before any scatter into the staging buffer."""
        header.validate(elements=self.elements, width=self.width, installed_version=header.base_version)
        self.validate_vectors(base, staged)
        if header.encoding == Encoding.DENSE:
            return {"validation_status_d2h_bytes": 0}
        started = phases.start("receiver_base_copy_gpu_seconds")
        staged.copy_(base)
        phases.end("receiver_base_copy_gpu_seconds", started)
        if header.encoding == Encoding.NOOP:
            return {"validation_status_d2h_bytes": 0}
        local = header.encoding == Encoding.BLOCK_LOCAL16
        indices = self.indices16 if local else self.indices32
        grid = self.blocks if local else (header.entries + BLOCK_ELEMENTS - 1) // BLOCK_ELEMENTS
        started = phases.start("payload_validate_gpu_seconds")
        self.kernels.validate_indices[(grid,)](
            indices, self.offsets, self.errors, self.elements, header.entries, LOCAL=local, BLOCK=BLOCK_ELEMENTS
        )
        phases.end("payload_validate_gpu_seconds", started)
        status_started = time.perf_counter()
        invalid = int(self.errors[:grid].sum().item())
        stats = {
            "validation_status_sync_seconds": time.perf_counter() - status_started,
            "validation_status_d2h_bytes": 8,
        }
        if invalid:
            raise ValueError("Sparse addressing is malformed")
        started = phases.start("receiver_apply_gpu_seconds")
        self.kernels.apply_changes[(grid,)](
            indices, self.offsets, self.values, staged, header.entries, LOCAL=local, BLOCK=BLOCK_ELEMENTS
        )
        phases.end("receiver_apply_gpu_seconds", started)
        return stats
