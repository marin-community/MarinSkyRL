from collections.abc import Generator
from contextlib import contextmanager
import io
import os
from typing import cast, Protocol, runtime_checkable

from fsspec import AbstractFileSystem
from loguru import logger
import torch
from torch.distributed.checkpoint import FileSystemWriter, SavePlan, SavePlanner
from torch.distributed.checkpoint._fsspec_filesystem import FileSystem as FsspecFileSystem
from torch.distributed.checkpoint.filesystem import (
    DEFAULT_SUFFIX,
    _item_size,
    _OverlappingCpuLoader,
    _SerialCpuLoader,
    _write_item,
)
from torch.distributed.checkpoint.planner import WriteItem, WriteItemType
from torch.distributed.checkpoint.storage import WriteResult
from torch.futures import Future


DEFAULT_TENSOR_COPY_AHEAD_BYTES = 2**31


@runtime_checkable
class _AbortableWriteStream(Protocol):
    closed: bool

    def discard(self) -> None: ...


class _AbortableFsspecFileSystem(FsspecFileSystem):
    def __init__(self, filesystem: AbstractFileSystem) -> None:
        super().__init__()
        self.fs = filesystem

    def init_path(self, path: str | os.PathLike, **_kwargs) -> str | os.PathLike:
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
            if any(character in mode for character in "w+a") and isinstance(stream, _AbortableWriteStream):
                try:
                    stream.discard()
                    stream.closed = True
                except Exception as cleanup_error:
                    error.add_note(f"Failed to abort multipart upload for {path}: {cleanup_error}")
            error.add_note(f"Object write failed for {path}")
            raise


class StreamingFsspecWriter(FileSystemWriter):
    """Stream one aggregated DCP object per rank with bounded tensor copy-ahead."""

    def __init__(
        self,
        path: str,
        *,
        filesystem: AbstractFileSystem,
        tensor_copy_ahead_bytes: int = DEFAULT_TENSOR_COPY_AHEAD_BYTES,
    ) -> None:
        if tensor_copy_ahead_bytes <= 0:
            raise ValueError("tensor_copy_ahead_bytes must be positive")
        super().__init__(
            path,
            single_file_per_rank=True,
            sync_files=False,
            thread_count=1,
            per_thread_copy_ahead=tensor_copy_ahead_bytes,
        )
        self.fs = _AbortableFsspecFileSystem(filesystem)
        self.path = self.fs.init_path(path)
        self.tensor_copy_ahead_bytes = tensor_copy_ahead_bytes

    def prepare_local_plan(self, plan: SavePlan) -> SavePlan:
        plan = super().prepare_local_plan(plan)
        logger.info(
            "DCP direct-write plan rank={} items={} files=1 copy_ahead_bytes={}",
            self.rank,
            len(plan.items),
            self.tensor_copy_ahead_bytes,
        )
        return plan

    def write_data(self, plan: SavePlan, planner: SavePlanner) -> Future[list[WriteResult]]:
        """Write a rank shard without retaining serialized tensors until close."""
        storage_plan = plan.storage_data
        if storage_plan is None:
            raise AssertionError("DCP storage plan is missing its rank prefix")
        file_name = f"{storage_plan.prefix}0{DEFAULT_SUFFIX}"
        path = self.fs.concat_path(self.path, file_name)

        tensor_items = [item for item in plan.items if item.type != WriteItemType.BYTE_IO]
        if torch.cuda.is_available():
            loader = _OverlappingCpuLoader(
                planner.resolve_data,
                inflight_threshhold=self.tensor_copy_ahead_bytes,
            )
        else:
            loader = _SerialCpuLoader(planner.resolve_data)
        for item in tensor_items:
            loader.add(_item_size(item), item)
        loader.start_loading()

        results: list[WriteResult] = []
        with self.fs.create_stream(path, "wb") as stream:
            for item in plan.items:
                if item.type == WriteItemType.BYTE_IO:
                    results.append(
                        _write_item(
                            self.transforms,
                            stream,
                            planner.resolve_data(item),
                            item,
                            file_name,
                            self.serialization_format,
                        )
                    )

            # PyTorch's stock one-file writer fills a safetensors dictionary even
            # for torch.save, retaining every GPU-to-CPU copy until the rank closes.
            # Emit each result immediately so only the loader's bounded window lives.
            for tensor, item_object in loader.values():
                item = cast(WriteItem, item_object)
                if not tensor.is_cpu:
                    raise AssertionError("DCP tensor must be on CPU before serialization")
                results.append(
                    _write_item(
                        self.transforms,
                        stream,
                        tensor,
                        item,
                        file_name,
                        self.serialization_format,
                    )
                )

        future: Future[list[WriteResult]] = Future()
        future.set_result(results)
        return future
