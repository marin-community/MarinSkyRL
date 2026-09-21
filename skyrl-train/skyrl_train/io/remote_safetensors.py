"""Range-backed safetensors reads for immutable object-store model exports."""

from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path, PurePosixPath
import struct

import torch

from marinskyrl.resource_locator import join_resource_path
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

    def __init__(self, source_uri: str, metadata_dir: str | Path) -> None:
        self.source_uri = source_uri.rstrip("/")
        index_path = Path(metadata_dir) / "model.safetensors.index.json"
        try:
            index = json.loads(index_path.read_text())
            weight_map = index["weight_map"]
        except (FileNotFoundError, KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid or missing safetensors weight index: {index_path}") from error
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Safetensors weight index has an empty weight_map: {index_path}")
        self._weight_map = {str(key): _safe_relative_path(str(value)) for key, value in weight_map.items()}
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

    def load_tensors(self, keys: list[str]) -> dict[str, torch.Tensor]:
        missing = [key for key in keys if key not in self._weight_map]
        if missing:
            raise KeyError(f"Tensors are absent from the safetensors index: {missing}")
        by_shard: dict[str, list[str]] = defaultdict(list)
        for key in keys:
            by_shard[self._weight_map[key]].append(key)

        tensors: dict[str, torch.Tensor] = {}
        for shard, shard_keys in by_shard.items():
            shard_uri = join_resource_path(self.source_uri, shard)
            with io.open_file(shard_uri, "rb") as source:
                data_offset, header = self._header(shard, source)
                for key in shard_keys:
                    descriptor = header.get(key)
                    if not isinstance(descriptor, dict):
                        raise ValueError(f"Tensor {key!r} is absent from {shard_uri}")
                    try:
                        dtype = _TORCH_DTYPES[str(descriptor["dtype"])]
                        shape = tuple(int(size) for size in descriptor["shape"])
                        start, end = (int(offset) for offset in descriptor["data_offsets"])
                    except (KeyError, TypeError, ValueError) as error:
                        raise ValueError(f"Invalid safetensors descriptor for {key!r} in {shard_uri}") from error
                    expected_size = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
                    if start < 0 or end < start or end - start != expected_size:
                        raise ValueError(f"Invalid safetensors byte range for {key!r} in {shard_uri}")
                    source.seek(data_offset + start)
                    payload = bytearray(source.read(end - start))
                    if len(payload) != end - start:
                        raise ValueError(f"Truncated tensor {key!r} in {shard_uri}")
                    self.bytes_read += len(payload)
                    tensors[key] = torch.frombuffer(payload, dtype=dtype).reshape(shape)
        return tensors
