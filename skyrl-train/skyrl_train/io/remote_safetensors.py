"""Range-backed safetensors reads for immutable object-store model exports."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, replace
import fnmatch
import json
import math
import time
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Iterable

import torch
from fsspec.utils import merge_offset_ranges

from marinskyrl.resource_locator import join_resource_path
from marinskyrl.model_manifest import (
    HF_WEIGHT_INDEX_FILENAME,
    SAFETENSORS_LENGTH_PREFIX_BYTES,
    read_safetensors_header,
)
from skyrl_train.hf_model_io import HF_WEIGHT_FILENAME
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

_PREFETCH_WINDOW_BYTES = 1024**3
_PREFETCH_MAX_BLOCK = 64 * 1024**2
_PREFETCH_MAX_GAP = 1024**2


def prefetch_items_from_tasks(
    tasks: Iterable[Any], lazy_keys: set[str], expert_index: Callable[[str], int]
) -> list[tuple[str, int | None]]:
    """Return rank-owned HF keys and expert slices in conversion-task order."""
    items = dict.fromkeys(
        (key, expert_index(task.mapping.megatron_param) if key in lazy_keys else None)
        for task in tasks
        if task.megatron_module is not None
        for key in (
            [task.mapping.hf_param] if isinstance(task.mapping.hf_param, str) else task.mapping.hf_param.values()
        )
    )
    return list(items)


def lazy_first_dim_patterns_for_bridge(bridge: object) -> tuple[str, ...]:
    """Return remote slice patterns from an AutoBridge's registered model bridge."""
    registered_bridge = getattr(bridge, "_model_bridge", bridge)
    return tuple(getattr(registered_bridge, "REMOTE_FIRST_DIM_SLICE_PATTERNS", ()))


def _safe_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Invalid safetensors shard path: {value!r}")
    return path.as_posix()


@dataclass(frozen=True)
class ReadStats:
    """Count shard opens and logical header or tensor reads with their bytes and elapsed time."""

    opens: int = 0
    reads: int = 0
    bytes_read: int = 0
    read_seconds: float = 0.0
    gets: int = 0
    prefetch_misses: int = 0
    prefetch_unused: int = 0


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
        self._headers: dict[str, tuple[int, dict[str, object]]] = {}
        self.read_stats = ReadStats()
        self._handles: dict[str, BinaryIO] = {}
        self._prefetch_plan: deque[tuple[str, int | None]] | None = None
        self._prefetched: dict[tuple[str, int | None], bytearray] = {}
        if index_path.exists():
            try:
                index = json.loads(index_path.read_text())
                weight_map = index["weight_map"]
            except (KeyError, TypeError, json.JSONDecodeError) as error:
                raise ValueError(f"Invalid safetensors weight index: {index_path}") from error
        else:
            # A single-file export maps every header key to model.safetensors.
            shard = HF_WEIGHT_FILENAME
            shard_uri = join_resource_path(self.source_uri, shard)
            if not io.exists(shard_uri):
                raise ValueError(f"Missing safetensors weight index and single-file shard: {index_path}")
            _offset, header = self._header(shard)
            weight_map = {key: shard for key in header if key != "__metadata__"}
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Safetensors weight index has an empty weight_map: {index_path}")
        self._weight_map = {str(key): _safe_relative_path(str(value)) for key, value in weight_map.items()}
        self._lazy_first_dim_keys = {
            key for key in self._weight_map if any(fnmatch.fnmatch(key, pattern) for pattern in lazy_first_dim_patterns)
        }

    def get_all_keys(self) -> list[str]:
        return sorted(self._weight_map)

    def plan_prefetch(self, items: Iterable[tuple[str, int | None]]) -> None:
        """Fetch requested tensor ranges in bounded windows as the reader consumes them."""
        self._prefetch_plan = deque(dict.fromkeys(items))

    def _source(self, shard: str) -> BinaryIO:
        source = self._handles.get(shard)
        if source is None:
            started = time.monotonic()
            source = io.open_file(join_resource_path(self.source_uri, shard), "rb", cache_type="none")
            self._handles[shard] = source
            self.read_stats = replace(
                self.read_stats,
                opens=self.read_stats.opens + 1,
                read_seconds=self.read_stats.read_seconds + time.monotonic() - started,
            )
        return source

    def _fill_window(self) -> None:
        items = []
        budget = 0
        while self._prefetch_plan:
            key, index = self._prefetch_plan[0]
            shard = self._weight_map[key]
            offset, header = self._header(shard)
            dtype, shape, start, end = self._tensor_descriptor(key, shard, header)
            if index is not None:
                if not shape or index < 0 or index >= shape[0]:
                    raise IndexError(f"Invalid prefetch expert index {index} for {key!r} with shape {shape}")
                slice_size = math.prod(shape[1:]) * torch.empty((), dtype=dtype).element_size()
                start += index * slice_size
                end = start + slice_size
            cost = end - start + _PREFETCH_MAX_GAP
            if cost > _PREFETCH_WINDOW_BYTES:
                self._prefetch_plan.popleft()
                continue
            if budget + cost > _PREFETCH_WINDOW_BYTES:
                break
            self._prefetch_plan.popleft()
            budget += cost
            items.append((key, index, shard, offset + start, offset + end))
        if not items:
            return
        filesystem = getattr(self._source(items[0][2]), "fs")
        paths, starts, ends = [], [], []
        for _key, _index, shard, start, end in items:
            path = io._normalized_path(filesystem, join_resource_path(self.source_uri, shard))
            for part_start in range(start, end, _PREFETCH_MAX_BLOCK):
                paths.append(path)
                starts.append(part_start)
                ends.append(min(part_start + _PREFETCH_MAX_BLOCK, end))
        paths, starts, ends = merge_offset_ranges(
            paths, starts, ends, max_gap=_PREFETCH_MAX_GAP, max_block=_PREFETCH_MAX_BLOCK
        )
        started = time.monotonic()
        payloads = filesystem.cat_ranges(paths, starts, ends, batch_size=16)
        self.read_stats = replace(
            self.read_stats,
            gets=self.read_stats.gets + len(paths),
            read_seconds=self.read_stats.read_seconds + time.monotonic() - started,
        )
        for start, end, payload in zip(starts, ends, payloads, strict=True):
            if isinstance(payload, Exception):
                raise payload
            if len(payload) != end - start:
                raise ValueError("Truncated prefetched safetensors range")
        for key, index, shard, start, end in items:
            path = io._normalized_path(filesystem, join_resource_path(self.source_uri, shard))
            value = bytearray()
            for part_start in range(start, end, _PREFETCH_MAX_BLOCK):
                part_end = min(part_start + _PREFETCH_MAX_BLOCK, end)
                for fetched_path, low, high, payload in zip(paths, starts, ends, payloads, strict=True):
                    if fetched_path == path and low <= part_start and part_end <= high:
                        value.extend(payload[part_start - low : part_end - low])
                        break
                else:
                    raise ValueError(f"Prefetch did not cover {key!r}")
            self._prefetched[key, index] = value

    def _pop_prefetched(self, key: str, index: int | None = None) -> bytearray | None:
        if self._prefetch_plan is None:
            return None
        if not self._prefetched:
            self._fill_window()
        started = time.monotonic()
        payload = self._prefetched.pop((key, index), None)
        if payload is None:
            self.read_stats = replace(self.read_stats, prefetch_misses=self.read_stats.prefetch_misses + 1)
        else:
            self._record_read(len(payload), started)
        return payload

    def close(self) -> None:
        """Release buffered ranges and source handles, recording unconsumed plan items."""
        self.read_stats = replace(
            self.read_stats,
            prefetch_unused=self.read_stats.prefetch_unused + len(self._prefetched) + len(self._prefetch_plan or ()),
        )
        self._prefetched.clear()
        self._prefetch_plan = None
        for source in self._handles.values():
            source.close()
        self._handles.clear()
        self._headers.clear()

    def _header(self, shard: str) -> tuple[int, dict[str, object]]:
        cached = self._headers.get(shard)
        if cached is not None:
            return cached
        source = self._source(shard)
        source.seek(0)
        started = time.monotonic()
        header_bytes, _keys = read_safetensors_header(source, join_resource_path(self.source_uri, shard))
        self._record_read(len(header_bytes), started)
        self.read_stats = replace(self.read_stats, gets=self.read_stats.gets + 2)
        cached = len(header_bytes), json.loads(header_bytes[SAFETENSORS_LENGTH_PREFIX_BYTES:])
        self._headers[shard] = cached
        return cached

    def _record_read(self, size: int, started: float) -> None:
        self.read_stats = replace(
            self.read_stats,
            reads=self.read_stats.reads + 1,
            bytes_read=self.read_stats.bytes_read + size,
            read_seconds=self.read_stats.read_seconds + time.monotonic() - started,
        )

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
        data_offset, header = self._header(shard)
        source = self._handles[shard]
        dtype, shape, start, _end = self._tensor_descriptor(key, shard, header)
        if not shape:
            raise IndexError(f"Cannot slice scalar safetensors tensor {key!r}")
        normalized_index = index + shape[0] if index < 0 else index
        if normalized_index < 0 or normalized_index >= shape[0]:
            raise IndexError(f"First-dimension index {index} is out of bounds for {key!r} with shape {shape}")
        slice_shape = shape[1:]
        slice_size = math.prod(slice_shape) * torch.empty((), dtype=dtype).element_size()
        slice_start = start + normalized_index * slice_size
        payload = self._pop_prefetched(key, normalized_index)
        if payload is None:
            source.seek(data_offset + slice_start)
            started = time.monotonic()
            payload = bytearray(source.read(slice_size))
            self._record_read(len(payload), started)
            self.read_stats = replace(self.read_stats, gets=self.read_stats.gets + 1)
        if len(payload) != slice_size:
            raise ValueError(f"Truncated tensor slice {key!r}[{index}] in {shard_uri}")
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
            data_offset, header = self._header(shard)
            source = self._handles[shard]
            for key in shard_keys:
                dtype, shape, start, end = self._tensor_descriptor(key, shard, header)
                payload = self._pop_prefetched(key)
                if payload is None:
                    source.seek(data_offset + start)
                    started = time.monotonic()
                    payload = bytearray(source.read(end - start))
                    self._record_read(len(payload), started)
                    self.read_stats = replace(self.read_stats, gets=self.read_stats.gets + 1)
                if len(payload) != end - start:
                    raise ValueError(f"Truncated tensor {key!r} in {shard_uri}")
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
