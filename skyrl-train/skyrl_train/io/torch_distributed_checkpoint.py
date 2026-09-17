from collections.abc import Generator
from contextlib import contextmanager
import io
import os

from fsspec import AbstractFileSystem
from loguru import logger
from torch.distributed.checkpoint import FileSystemWriter, SavePlan
from torch.distributed.checkpoint._fsspec_filesystem import FileSystem as FsspecFileSystem


DEFAULT_TENSOR_COPY_AHEAD_BYTES = 2**31


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
            if any(character in mode for character in "w+a") and hasattr(stream, "discard"):
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
