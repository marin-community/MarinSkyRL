"""Bounded exact wire-bit replay primitives; unused by production publication.

Indices are little-endian uint32 element offsets. Values are replacement bytes,
never arithmetic deltas. This is a chunk experiment, not a publication protocol.
"""

from dataclasses import dataclass
import hashlib
import time

import numpy as np


MAX_CACHE_BYTES = 256 * 1024**2
MAX_CHUNK_BYTES = 16 * 1024**2
WIRE_BYTES = {"bfloat16": 2, "float32": 4}


@dataclass(frozen=True)
class EncodedChunk:
    dtype: str
    elements: int
    base_version: int
    target_version: int
    base_sha256: str
    target_sha256: str
    encoding: str
    indices: bytes
    values: bytes

    @property
    def payload_bytes(self) -> int:
        return len(self.indices) + len(self.values)


def _element_bytes(dtype: str) -> int:
    if dtype not in WIRE_BYTES:
        raise ValueError("Only raw bfloat16 and float32 wire layouts are supported")
    return WIRE_BYTES[dtype]


def _validate_bytes(value: bytes, element_bytes: int) -> None:
    if not isinstance(value, bytes) or len(value) > MAX_CHUNK_BYTES or len(value) % element_bytes:
        raise ValueError("Wire chunk must be aligned immutable bytes within the 16 MiB bound")


def _validate_versions(base_version: int, target_version: int) -> None:
    if type(base_version) is not int or type(target_version) is not int or not 0 <= base_version < target_version:
        raise ValueError("Expected nonnegative base version and strictly newer target version")


def encode_chunk(
    base: bytes,
    target: bytes,
    *,
    dtype: str,
    base_version: int,
    target_version: int,
    mode: str = "auto",
) -> tuple[EncodedChunk, dict[str, float | int]]:
    """Encode one bounded range and report scan/hash/packing CPU wall times."""
    width = _element_bytes(dtype)
    _validate_bytes(base, width)
    _validate_bytes(target, width)
    _validate_versions(base_version, target_version)
    if len(base) != len(target) or mode not in {"auto", "dense", "indexed"}:
        raise ValueError("Equal-sized chunks and a supported encoding mode are required")
    started = time.perf_counter()
    base_hash, target_hash = hashlib.sha256(base).hexdigest(), hashlib.sha256(target).hexdigest()
    hash_seconds = time.perf_counter() - started
    started = time.perf_counter()
    before = np.frombuffer(base, dtype=f"<u{width}")
    after = np.frombuffer(target, dtype=f"<u{width}")
    mask = before != after
    changed = int(np.count_nonzero(mask))
    scan_seconds = time.perf_counter() - started
    encoding = mode if mode != "auto" else ("indexed" if changed * (4 + width) < len(target) else "dense")
    started = time.perf_counter()
    indices = np.flatnonzero(mask).astype("<u4").tobytes() if encoding == "indexed" else b""
    values = after[mask].tobytes() if encoding == "indexed" else target
    pack_seconds = time.perf_counter() - started
    return (
        EncodedChunk(
            dtype, len(after), base_version, target_version, base_hash, target_hash, encoding, indices, values
        ),
        {
            "changed_elements": changed,
            "hash_seconds": hash_seconds,
            "scan_seconds": scan_seconds,
            "pack_seconds": pack_seconds,
        },
    )


def decode_chunk(base: bytes, chunk: EncodedChunk, *, installed_version: int) -> tuple[bytes, dict[str, float]]:
    """Validate a patch against its predecessor and reconstruct exact target bits."""
    width = _element_bytes(chunk.dtype)
    _validate_bytes(base, width)
    _validate_versions(chunk.base_version, chunk.target_version)
    if type(installed_version) is not int or installed_version != chunk.base_version:
        raise ValueError("Installed base version differs")
    if type(chunk.elements) is not int or chunk.elements != len(base) // width:
        raise ValueError("Chunk element count differs from baseline")
    if not isinstance(chunk.indices, bytes) or not isinstance(chunk.values, bytes):
        raise ValueError("Encoded payload must contain immutable bytes")
    if chunk.payload_bytes > 3 * MAX_CHUNK_BYTES:
        raise ValueError("Encoded payload exceeds chunk bound")
    started = time.perf_counter()
    if hashlib.sha256(base).hexdigest() != chunk.base_sha256:
        raise ValueError("Base checksum differs")
    base_hash_seconds = time.perf_counter() - started
    started = time.perf_counter()
    if chunk.encoding == "dense":
        if chunk.indices or len(chunk.values) != len(base):
            raise ValueError("Malformed dense payload")
        target = chunk.values
    elif chunk.encoding == "indexed":
        if len(chunk.indices) % 4 or len(chunk.values) != len(chunk.indices) // 4 * width:
            raise ValueError("Malformed indexed payload lengths")
        indices = np.frombuffer(chunk.indices, dtype="<u4")
        if len(indices) and (int(indices[-1]) >= chunk.elements or np.any(indices[1:] <= indices[:-1])):
            raise ValueError("Indices must be strictly increasing, unique and in range")
        target_buffer = bytearray(base)
        np.frombuffer(target_buffer, dtype=f"<u{width}")[indices] = np.frombuffer(chunk.values, dtype=f"<u{width}")
        target = bytes(target_buffer)
    else:
        raise ValueError("Unknown chunk encoding")
    apply_seconds = time.perf_counter() - started
    started = time.perf_counter()
    if hashlib.sha256(target).hexdigest() != chunk.target_sha256:
        raise ValueError("Target checksum differs")
    return target, {
        "base_hash_seconds": base_hash_seconds,
        "apply_seconds": apply_seconds,
        "target_hash_seconds": time.perf_counter() - started,
    }


class HostBaselineCache:
    """Own bounded immutable chunks for a frozen replay, with explicit admission."""

    def __init__(self, max_bytes: int = MAX_CACHE_BYTES):
        if type(max_bytes) is not int or not 0 < max_bytes <= MAX_CACHE_BYTES:
            raise ValueError("Cache budget must be positive and at most 256 MiB")
        self.max_bytes = max_bytes
        self.retained_bytes = 0
        self._chunks: dict[str, bytes] = {}

    def add(self, key: str, data: bytes) -> None:
        _validate_bytes(data, 1)
        if key in self._chunks:
            raise ValueError("Duplicate baseline key")
        if self.retained_bytes + len(data) > self.max_bytes:
            raise ValueError("Baseline cache budget exceeded")
        self._chunks[key] = data
        self.retained_bytes += len(data)

    def get(self, key: str) -> bytes:
        return self._chunks[key]
