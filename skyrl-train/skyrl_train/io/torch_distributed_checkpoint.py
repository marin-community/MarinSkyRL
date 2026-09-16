from collections.abc import Generator
from contextlib import contextmanager
import io
import math
import os

from fsspec import AbstractFileSystem
from fsspec.core import url_to_fs
from loguru import logger
import torch
from torch.distributed.checkpoint import FileSystemWriter, SavePlan, WriteItem
from torch.distributed.checkpoint._fsspec_filesystem import FileSystem as FsspecFileSystem


DEFAULT_TENSOR_COPY_AHEAD_BYTES = 2**30


class _AbortableFsspecFileSystem(FsspecFileSystem):
    def __init__(self, filesystem: AbstractFileSystem | None = None) -> None:
        super().__init__()
        self.fs = filesystem

    def init_path(self, path: str | os.PathLike, **kwargs) -> str | os.PathLike:
        if self.fs is None:
            self.fs, _ = url_to_fs(path, **kwargs)
        return path

    @contextmanager
    def create_stream(self, path: str | os.PathLike, mode: str) -> Generator[io.IOBase, None, None]:
        if self.fs is None:
            raise AssertionError("filesystem has not been initialized")

        stream = self.fs.open(os.fspath(path), mode)
        try:
            yield stream
            stream.close()
        except BaseException as error:
            if any(character in mode for character in "w+a") and hasattr(stream, "discard"):
                try:
                    stream.discard()
                    stream.closed = True
                except Exception as cleanup_error:
                    error.add_note(f"Failed to abort multipart upload for {path}: {cleanup_error}")
            error.add_note(f"Object write failed for {path}")
            raise


def _tensor_write_item_bytes(item: WriteItem) -> int | None:
    if item.tensor_data is None:
        return None
    return math.prod(item.tensor_data.size) * torch._utils._element_size(item.tensor_data.properties.dtype)


class StreamingFsspecWriter(FileSystemWriter):
    """Write one DCP item per remote object with bounded tensor copy-ahead."""

    def __init__(
        self,
        path: str,
        *,
        filesystem: AbstractFileSystem | None = None,
        tensor_copy_ahead_bytes: int = DEFAULT_TENSOR_COPY_AHEAD_BYTES,
        **filesystem_kwargs,
    ) -> None:
        if tensor_copy_ahead_bytes <= 0:
            raise ValueError("tensor_copy_ahead_bytes must be positive")
        super().__init__(
            path,
            single_file_per_rank=False,
            sync_files=False,
            thread_count=1,
            per_thread_copy_ahead=tensor_copy_ahead_bytes,
        )
        self.fs = _AbortableFsspecFileSystem(filesystem)
        self.path = self.fs.init_path(path, **filesystem_kwargs)
        self.tensor_copy_ahead_bytes = tensor_copy_ahead_bytes

    def prepare_local_plan(self, plan: SavePlan) -> SavePlan:
        plan = super().prepare_local_plan(plan)
        tensor_items: list[tuple[WriteItem, int]] = []
        for item in plan.items:
            size = _tensor_write_item_bytes(item)
            if size is not None:
                tensor_items.append((item, size))

        total_bytes = sum(size for _, size in tensor_items)
        largest_bytes = max((size for _, size in tensor_items), default=0)
        logger.info(
            "DCP direct-write plan rank={} items={} tensor_bytes={} largest_tensor_bytes={} copy_ahead_bytes={}",
            self.rank,
            len(plan.items),
            total_bytes,
            largest_bytes,
            self.tensor_copy_ahead_bytes,
        )
        for item, size in tensor_items:
            if size > self.tensor_copy_ahead_bytes:
                logger.warning(
                    "DCP item exceeds copy-ahead target rank={} key={} tensor_bytes={} copy_ahead_bytes={}",
                    self.rank,
                    item.index.fqn,
                    size,
                    self.tensor_copy_ahead_bytes,
                )
        return plan
