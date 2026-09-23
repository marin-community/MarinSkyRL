from collections.abc import Generator
from collections import deque
from concurrent.futures import Future as ConcurrentFuture, ThreadPoolExecutor
from contextlib import contextmanager
import io
import os
import time
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

from skyrl_train.io.s3fs import call_with_s3_retry, get_s3_fs, s3_refresh_if_expiring
from skyrl_train.checkpoint_listing import extract_step_from_path
from skyrl_train.timing_observability import CheckpointPhaseSample, checkpoint_phase


DEFAULT_TENSOR_COPY_AHEAD_BYTES = 2**30
DEFAULT_S3_MULTIPART_PART_BYTES = 128 * 2**20
DEFAULT_S3_MULTIPART_CONCURRENCY = 4
_MINIMUM_S3_MULTIPART_PART_BYTES = 5 * 2**20
_S3_MULTIPART_PART_MAX_ATTEMPTS = 2


@runtime_checkable
class _AbortableWriteStream(Protocol):
    closed: bool

    def discard(self) -> None: ...


@runtime_checkable
class _MultipartS3FileSystem(Protocol):
    protocol: str | tuple[str, ...]

    def split_path(self, path: str) -> tuple[str, str, str | None]: ...

    def call_s3(self, method: str, *args, **kwargs): ...


class _ConcurrentS3WriteStream:
    """Bounded file-like multipart writer with concurrent UploadPart calls."""

    def __init__(
        self,
        filesystem: _MultipartS3FileSystem,
        path: str,
        *,
        part_bytes: int,
        concurrency: int,
    ) -> None:
        if part_bytes < _MINIMUM_S3_MULTIPART_PART_BYTES:
            raise ValueError(f"part_bytes must be at least {_MINIMUM_S3_MULTIPART_PART_BYTES}")
        if concurrency <= 0:
            raise ValueError("concurrency must be positive")

        bucket, key, _version_id = filesystem.split_path(path)
        self.filesystem = filesystem
        self.bucket = bucket
        self.key = key
        self.path = path
        self.part_bytes = part_bytes
        self.concurrency = concurrency
        self.closed = False
        self._discarded = False
        self._position = 0
        self._buffer = bytearray()
        self._upload_id: str | None = None
        self._next_part_number = 1
        self._completed_parts: list[dict[str, int | str]] = []
        self._pending: deque[tuple[int, ConcurrentFuture[tuple[dict[str, int | str], float]]]] = deque()
        self._executor = ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="checkpoint-s3")
        self._write_error: BaseException | None = None
        self.create_upload_seconds = 0.0
        self.upload_part_seconds_total = 0.0
        self.upload_part_seconds_max = 0.0
        self.upload_queue_wait_seconds = 0.0
        self.complete_upload_seconds = 0.0
        self.uploaded_part_count = 0

    def writable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def flush(self) -> None:
        if self.closed:
            raise ValueError("flush of closed checkpoint stream")

    def write(self, payload) -> int:
        if self.closed:
            raise ValueError("write to closed checkpoint stream")
        view = memoryview(payload).cast("B")
        payload_bytes = len(view)
        if self._write_error is None:
            try:
                while view:
                    chunk_bytes = min(self.part_bytes - len(self._buffer), len(view))
                    self._buffer.extend(view[:chunk_bytes])
                    view = view[chunk_bytes:]
                    if len(self._buffer) == self.part_bytes:
                        self._submit_part(bytes(self._buffer))
                        self._buffer.clear()
            except BaseException as error:
                # torch.save replaces exceptions from a Python write callback with
                # an `unexpected pos` assertion while finalizing its zip stream.
                # Accept the remaining serialization callbacks, then raise the
                # initiating storage error from close(), outside the C++ writer.
                self._write_error = error
                self._buffer.clear()
        self._position += payload_bytes
        return payload_bytes

    def close(self) -> None:
        if self.closed:
            return
        if self._discarded:
            self.closed = True
            return
        if self._write_error is not None:
            error = self._write_error
            try:
                self.discard()
            except Exception as cleanup_error:
                error.add_note(f"Failed to abort multipart upload for {self.path}: {cleanup_error}")
            raise error

        if self._upload_id is None:
            call_with_s3_retry(
                self.filesystem,
                self.filesystem.call_s3,
                "put_object",
                max_attempts=1,
                Bucket=self.bucket,
                Key=self.key,
                Body=bytes(self._buffer),
            )
            self._buffer.clear()
            self._executor.shutdown(wait=True, cancel_futures=True)
            self.closed = True
            return

        if self._buffer:
            self._submit_part(bytes(self._buffer))
            self._buffer.clear()
        while self._pending:
            self._finish_oldest_part()
        self._executor.shutdown(wait=True)
        started = time.perf_counter()
        call_with_s3_retry(
            self.filesystem,
            self.filesystem.call_s3,
            "complete_multipart_upload",
            max_attempts=1,
            Bucket=self.bucket,
            Key=self.key,
            UploadId=self._upload_id,
            MultipartUpload={"Parts": sorted(self._completed_parts, key=lambda part: int(part["PartNumber"]))},
        )
        self.complete_upload_seconds = time.perf_counter() - started
        self.closed = True

    def discard(self) -> None:
        if self.closed or self._discarded:
            return
        self._discarded = True
        for _part_number, future in self._pending:
            future.cancel()
        if self._upload_id is not None:
            call_with_s3_retry(
                self.filesystem,
                self.filesystem.call_s3,
                "abort_multipart_upload",
                max_attempts=1,
                Bucket=self.bucket,
                Key=self.key,
                UploadId=self._upload_id,
            )
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._buffer.clear()
        self.closed = True

    def _submit_part(self, payload: bytes) -> None:
        if self._upload_id is None:
            started = time.perf_counter()
            response = call_with_s3_retry(
                self.filesystem,
                self.filesystem.call_s3,
                "create_multipart_upload",
                max_attempts=1,
                Bucket=self.bucket,
                Key=self.key,
            )
            self.create_upload_seconds = time.perf_counter() - started
            self._upload_id = str(response["UploadId"])

        part_number = self._next_part_number
        self._next_part_number += 1
        future = self._executor.submit(self._upload_part, part_number, payload)
        self._pending.append((part_number, future))
        if len(self._pending) >= self.concurrency:
            self._finish_oldest_part()

    def _upload_part(self, part_number: int, payload: bytes) -> tuple[dict[str, int | str], float]:
        started = time.perf_counter()
        try:
            response = call_with_s3_retry(
                self.filesystem,
                self.filesystem.call_s3,
                "upload_part",
                max_attempts=_S3_MULTIPART_PART_MAX_ATTEMPTS,
                Bucket=self.bucket,
                Key=self.key,
                UploadId=self._upload_id,
                PartNumber=part_number,
                Body=payload,
            )
        except BaseException as error:
            error.add_note(f"Multipart upload failed for {self.path}: upload_id={self._upload_id} part={part_number}")
            raise
        return {"PartNumber": part_number, "ETag": str(response["ETag"])}, time.perf_counter() - started

    def _finish_oldest_part(self) -> None:
        _part_number, future = self._pending.popleft()
        started = time.perf_counter()
        part, upload_seconds = future.result()
        self.upload_queue_wait_seconds += time.perf_counter() - started
        self.upload_part_seconds_total += upload_seconds
        self.upload_part_seconds_max = max(self.upload_part_seconds_max, upload_seconds)
        self.uploaded_part_count += 1
        self._completed_parts.append(part)


class _AbortableFsspecFileSystem(FsspecFileSystem):
    def __init__(
        self,
        filesystem: AbstractFileSystem,
        *,
        multipart_part_bytes: int,
        multipart_concurrency: int,
    ) -> None:
        super().__init__()
        self.fs = filesystem
        self.multipart_part_bytes = multipart_part_bytes
        self.multipart_concurrency = multipart_concurrency

    def init_path(self, path: str | os.PathLike, **_kwargs) -> str | os.PathLike:
        return path

    @contextmanager
    def create_stream(self, path: str | os.PathLike, mode: str) -> Generator[io.IOBase, None, None]:
        if self.fs is None:
            raise AssertionError("filesystem has not been initialized")

        object_path = os.fspath(path)
        protocol = self.fs.protocol
        protocols = (protocol,) if isinstance(protocol, str) else protocol
        if mode == "wb" and object_path.endswith((DEFAULT_SUFFIX, ".pt")) and "s3" in protocols:
            if not isinstance(self.fs, _MultipartS3FileSystem):
                raise TypeError("S3 checkpoint filesystem does not provide multipart operations")
            stream = _ConcurrentS3WriteStream(
                self.fs,
                object_path,
                part_bytes=self.multipart_part_bytes,
                concurrency=self.multipart_concurrency,
            )
        else:
            stream = self.fs.open(object_path, mode)
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


@contextmanager
def open_s3_checkpoint_write_stream(path: str):
    """Write one checkpoint object with bounded multipart staging and abort on failure."""
    if not path.startswith("s3://"):
        raise ValueError(f"Expected an S3 checkpoint object path, got {path!r}")
    filesystem = get_s3_fs()
    s3_refresh_if_expiring(filesystem)
    adapter = _AbortableFsspecFileSystem(
        filesystem,
        multipart_part_bytes=DEFAULT_S3_MULTIPART_PART_BYTES,
        multipart_concurrency=DEFAULT_S3_MULTIPART_CONCURRENCY,
    )
    with adapter.create_stream(path, "wb") as stream:
        yield stream


class StreamingFsspecWriter(FileSystemWriter):
    """Stream one aggregated DCP object per rank with bounded tensor copy-ahead."""

    def __init__(
        self,
        path: str,
        *,
        filesystem: AbstractFileSystem,
        tensor_copy_ahead_bytes: int = DEFAULT_TENSOR_COPY_AHEAD_BYTES,
        multipart_part_bytes: int = DEFAULT_S3_MULTIPART_PART_BYTES,
        multipart_concurrency: int = DEFAULT_S3_MULTIPART_CONCURRENCY,
    ) -> None:
        if tensor_copy_ahead_bytes <= 0:
            raise ValueError("tensor_copy_ahead_bytes must be positive")
        if multipart_part_bytes < _MINIMUM_S3_MULTIPART_PART_BYTES:
            raise ValueError(f"multipart_part_bytes must be at least {_MINIMUM_S3_MULTIPART_PART_BYTES}")
        if multipart_concurrency <= 0:
            raise ValueError("multipart_concurrency must be positive")
        super().__init__(
            path,
            single_file_per_rank=True,
            sync_files=False,
            thread_count=1,
            per_thread_copy_ahead=tensor_copy_ahead_bytes,
        )
        self.fs = _AbortableFsspecFileSystem(
            filesystem,
            multipart_part_bytes=multipart_part_bytes,
            multipart_concurrency=multipart_concurrency,
        )
        self.path = self.fs.init_path(path)
        self.tensor_copy_ahead_bytes = tensor_copy_ahead_bytes
        self.multipart_part_bytes = multipart_part_bytes
        self.multipart_concurrency = multipart_concurrency
        self.checkpoint_step = extract_step_from_path(os.path.dirname(path.rstrip("/")))

    def prepare_local_plan(self, plan: SavePlan) -> SavePlan:
        with checkpoint_phase("megatron", "save", "dcp_local_plan", rank=self.rank, step=self.checkpoint_step):
            plan = super().prepare_local_plan(plan)
        logger.info(
            "DCP direct-write plan rank={} items={} files=1 copy_ahead_bytes={} multipart_part_bytes={} "
            "multipart_concurrency={} max_staged_bytes={}",
            self.rank,
            len(plan.items),
            self.tensor_copy_ahead_bytes,
            self.multipart_part_bytes,
            self.multipart_concurrency,
            self.tensor_copy_ahead_bytes + self.multipart_part_bytes * (self.multipart_concurrency + 1),
        )
        return plan

    def prepare_global_plan(self, plans: list[SavePlan]) -> list[SavePlan]:
        with checkpoint_phase("megatron", "save", "dcp_global_plan", rank=self.rank, step=self.checkpoint_step):
            return super().prepare_global_plan(plans)

    def finish(self, metadata, results: list[list[WriteResult]]) -> None:
        with checkpoint_phase("megatron", "save", "dcp_finish", rank=self.rank, step=self.checkpoint_step):
            return super().finish(metadata, results)

    def write_data(self, plan: SavePlan, planner: SavePlanner) -> Future[list[WriteResult]]:
        """Write a rank shard without retaining serialized tensors until close."""
        with checkpoint_phase("megatron", "save", "stream_shard", rank=self.rank, step=self.checkpoint_step) as phase:
            result = self._write_data(plan, planner, phase)
            phase.bytes_written = sum(item.size_in_bytes for item in result.wait())
            return result

    def _write_data(
        self, plan: SavePlan, planner: SavePlanner, phase: CheckpointPhaseSample
    ) -> Future[list[WriteResult]]:
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
        started = time.perf_counter()
        loader.start_loading()
        phase.counters["loader_start_seconds"] = time.perf_counter() - started
        phase.counters["tensor_item_count"] = len(tensor_items)

        results: list[WriteResult] = []
        byte_item_seconds = 0.0
        tensor_item_seconds = 0.0
        loader_next_seconds = 0.0
        with self.fs.create_stream(path, "wb") as stream:
            for item in plan.items:
                if item.type == WriteItemType.BYTE_IO:
                    started = time.perf_counter()
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
                    byte_item_seconds += time.perf_counter() - started

            # PyTorch's stock one-file writer fills a safetensors dictionary even
            # for torch.save, retaining every GPU-to-CPU copy until the rank closes.
            # Emit each result immediately so only the loader's bounded window lives.
            values = iter(loader.values())
            while True:
                started = time.perf_counter()
                try:
                    tensor, item_object = next(values)
                except StopIteration:
                    loader_next_seconds += time.perf_counter() - started
                    break
                loader_next_seconds += time.perf_counter() - started
                item = cast(WriteItem, item_object)
                if not tensor.is_cpu:
                    raise AssertionError("DCP tensor must be on CPU before serialization")
                started = time.perf_counter()
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
                tensor_item_seconds += time.perf_counter() - started

        phase.counters.update(
            {
                "byte_item_seconds": byte_item_seconds,
                "tensor_item_seconds": tensor_item_seconds,
                "loader_next_seconds": loader_next_seconds,
            }
        )
        if isinstance(stream, _ConcurrentS3WriteStream):
            phase.counters.update(
                {
                    "create_upload_seconds": stream.create_upload_seconds,
                    "upload_part_seconds_total": stream.upload_part_seconds_total,
                    "upload_part_seconds_max": stream.upload_part_seconds_max,
                    "upload_queue_wait_seconds": stream.upload_queue_wait_seconds,
                    "complete_upload_seconds": stream.complete_upload_seconds,
                    "uploaded_part_count": stream.uploaded_part_count,
                }
            )

        future: Future[list[WriteResult]] = Future()
        future.set_result(results)
        return future
