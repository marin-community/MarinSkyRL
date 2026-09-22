"""Range-backed safetensors reads for immutable object-store model exports."""

from __future__ import annotations

from collections import defaultdict
import fnmatch
import json
import math
from pathlib import Path, PurePosixPath
import struct
from typing import Iterable

import torch

from marinskyrl.resource_locator import join_resource_path
from marinskyrl.model_manifest import HF_WEIGHT_INDEX_FILENAME
from skyrl_train.io import io


_TORCH_DTYPES = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
}


def _safe_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Invalid safetensors shard path: {value!r}")
    return path.as_posix()


class RemoteSafetensorsTensorStore:
    """Load requested tensors with object-store range reads and no weight files on disk."""

    def __init__(
        self,
        source_uri: str,
        metadata_dir: str | Path,
        *,
        lazy_first_dim_patterns: Iterable[str] = (),
    ) -> None:
        self.source_uri = source_uri.rstrip("/")
        index_path = Path(metadata_dir) / HF_WEIGHT_INDEX_FILENAME
        try:
            index = json.loads(index_path.read_text())
            weight_map = index["weight_map"]
        except (FileNotFoundError, KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid or missing safetensors weight index: {index_path}") from error
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Safetensors weight index has an empty weight_map: {index_path}")
        self._weight_map = {str(key): _safe_relative_path(str(value)) for key, value in weight_map.items()}
        self._lazy_first_dim_keys = {
            key for key in self._weight_map if any(fnmatch.fnmatch(key, pattern) for pattern in lazy_first_dim_patterns)
        }
        self._headers: dict[str, tuple[int, dict[str, object]]] = {}
        self.bytes_read = 0

    def get_all_keys(self) -> list[str]:
        return sorted(self._weight_map)

    def _header(self, shard: str, source) -> tuple[int, dict[str, object]]:
        cached = self._headers.get(shard)
        if cached is not None:
            return cached
        source.seek(0)
        prefix = source.read(8)
        if len(prefix) != 8:
            raise ValueError(f"Truncated safetensors header: {join_resource_path(self.source_uri, shard)}")
        header_size = struct.unpack("<Q", prefix)[0]
        header_bytes = source.read(header_size)
        if len(header_bytes) != header_size:
            raise ValueError(f"Truncated safetensors metadata: {join_resource_path(self.source_uri, shard)}")
        header = json.loads(header_bytes)
        if not isinstance(header, dict):
            raise ValueError(f"Invalid safetensors metadata: {join_resource_path(self.source_uri, shard)}")
        self.bytes_read += 8 + header_size
        cached = 8 + header_size, header
        self._headers[shard] = cached
        return cached

    def _tensor_descriptor(
        self, key: str, shard: str, header: dict[str, object]
    ) -> tuple[torch.dtype, tuple[int, ...], int, int]:
        descriptor = header.get(key)
        if not isinstance(descriptor, dict):
            raise ValueError(f"Tensor {key!r} is absent from {join_resource_path(self.source_uri, shard)}")
        try:
            dtype = _TORCH_DTYPES[str(descriptor["dtype"])]
            shape = tuple(int(size) for size in descriptor["shape"])
            start, end = (int(offset) for offset in descriptor["data_offsets"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid safetensors descriptor for {key!r} in {join_resource_path(self.source_uri, shard)}"
            ) from error
        expected_size = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
        if start < 0 or end < start or end - start != expected_size:
            raise ValueError(
                f"Invalid safetensors byte range for {key!r} in {join_resource_path(self.source_uri, shard)}"
            )
        return dtype, shape, start, end

    def load_first_dim_slice(self, key: str, index: int) -> torch.Tensor:
        """Read one contiguous first-dimension slice of a stacked tensor."""
        if key not in self._weight_map:
            raise KeyError(f"Tensor is absent from the safetensors index: {key!r}")
        shard = self._weight_map[key]
        shard_uri = join_resource_path(self.source_uri, shard)
        with io.open_file(shard_uri, "rb") as source:
            data_offset, header = self._header(shard, source)
            dtype, shape, start, _end = self._tensor_descriptor(key, shard, header)
            if not shape:
                raise IndexError(f"Cannot slice scalar safetensors tensor {key!r}")
            normalized_index = index + shape[0] if index < 0 else index
            if normalized_index < 0 or normalized_index >= shape[0]:
                raise IndexError(f"First-dimension index {index} is out of bounds for {key!r} with shape {shape}")
            slice_shape = shape[1:]
            slice_size = math.prod(slice_shape) * torch.empty((), dtype=dtype).element_size()
            slice_start = start + normalized_index * slice_size
            source.seek(data_offset + slice_start)
            payload = bytearray(source.read(slice_size))
            if len(payload) != slice_size:
                raise ValueError(f"Truncated tensor slice {key!r}[{index}] in {shard_uri}")
            self.bytes_read += len(payload)
            return torch.frombuffer(payload, dtype=dtype).reshape(slice_shape)

    def load_tensors(self, keys: list[str]) -> dict[str, torch.Tensor]:
        missing = [key for key in keys if key not in self._weight_map]
        if missing:
            raise KeyError(f"Tensors are absent from the safetensors index: {missing}")
        tensors: dict[str, torch.Tensor] = {
            key: _LazyFirstDimensionTensor(self, key) for key in keys if key in self._lazy_first_dim_keys
        }
        by_shard: dict[str, list[str]] = defaultdict(list)
        for key in keys:
            if key not in self._lazy_first_dim_keys:
                by_shard[self._weight_map[key]].append(key)

        for shard, shard_keys in by_shard.items():
            shard_uri = join_resource_path(self.source_uri, shard)
            with io.open_file(shard_uri, "rb") as source:
                data_offset, header = self._header(shard, source)
                for key in shard_keys:
                    dtype, shape, start, end = self._tensor_descriptor(key, shard, header)
                    source.seek(data_offset + start)
                    payload = bytearray(source.read(end - start))
                    if len(payload) != end - start:
                        raise ValueError(f"Truncated tensor {key!r} in {shard_uri}")
                    self.bytes_read += len(payload)
                    tensors[key] = torch.frombuffer(payload, dtype=dtype).reshape(shape)
        return tensors


class _LazyFirstDimensionTensor:
    """Defer a stacked tensor read until a mapping selects its rank-owned slice."""

    def __init__(self, store: RemoteSafetensorsTensorStore, key: str) -> None:
        self._store = store
        self._key = key

    def __getitem__(self, index: int) -> torch.Tensor:
        if not isinstance(index, int):
            raise TypeError(f"Remote stacked tensor slices require an integer index, got {type(index).__name__}")
        return self._store.load_first_dim_slice(self._key, index)
