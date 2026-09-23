import io
import threading
import warnings

import fsspec
from fsspec import AbstractFileSystem
from fsspec.exceptions import FSTimeoutError
import pytest
import torch
from torch.distributed import checkpoint
from torch.distributed.checkpoint.api import CheckpointException

from skyrl_train.io.torch_distributed_checkpoint import (
    _ConcurrentS3WriteStream,
    _MINIMUM_S3_MULTIPART_PART_BYTES,
    StreamingFsspecWriter,
    open_s3_checkpoint_write_stream,
)


def test_streaming_fsspec_writer_round_trips_one_aggregated_object_per_rank():
    checkpoint_uri = "memory://streaming-checkpoint/step"
    filesystem = fsspec.filesystem("memory")
    if filesystem.exists("/streaming-checkpoint"):
        filesystem.rm("/streaming-checkpoint", recursive=True)
    state = {
        "first": torch.arange(8),
        "second": torch.arange(6).reshape(2, 3),
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        checkpoint.save(state, storage_writer=StreamingFsspecWriter(checkpoint_uri, filesystem=filesystem))

    files = filesystem.find("/streaming-checkpoint/step")
    assert len([path for path in files if path.endswith(".distcp")]) == 1
    assert "/streaming-checkpoint/step/.metadata" in files

    restored = {
        "first": torch.zeros_like(state["first"]),
        "second": torch.zeros_like(state["second"]),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        checkpoint.load(restored, checkpoint_id=checkpoint_uri)
    assert torch.equal(restored["first"], state["first"])
    assert torch.equal(restored["second"], state["second"])


class _FailingWriteStream:
    def __init__(self) -> None:
        self.discarded = False
        self.closed = False

    def write(self, _payload) -> int:
        raise OSError("injected object-store failure")

    def tell(self) -> int:
        return 0

    def flush(self) -> None:
        pass

    def discard(self) -> None:
        self.discarded = True


class _FailingFilesystem(AbstractFileSystem):
    def __init__(self) -> None:
        self.streams: list[_FailingWriteStream] = []
        self.opened_paths: list[str] = []

    def open(self, path: str, _mode: str) -> _FailingWriteStream:
        self.opened_paths.append(path)
        stream = _FailingWriteStream()
        self.streams.append(stream)
        return stream

    def makedirs(self, _path: str, exist_ok: bool = False) -> None:
        pass

    def exists(self, _path: str) -> bool:
        return False

    def rm(self, _path: str) -> None:
        pass

    def rename(self, _path: str, _new_path: str) -> None:
        pass


def test_streaming_fsspec_writer_aborts_failed_object():
    filesystem = _FailingFilesystem()
    writer = StreamingFsspecWriter("memory://failed-checkpoint/step", filesystem=filesystem)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with pytest.raises(CheckpointException):
            checkpoint.save({"tensor": torch.arange(8)}, storage_writer=writer)

    assert filesystem.streams
    assert all(stream.discarded for stream in filesystem.streams)
    assert not any(path.endswith(".metadata") for path in filesystem.opened_paths)


class _RecordingMultipartFilesystem:
    protocol = "s3"

    def __init__(self, fail_part: int | None = None, transient_failures: int = 0) -> None:
        self.fail_part = fail_part
        self.transient_failures = transient_failures
        self.upload_attempts = 0
        self.active_uploads = 0
        self.peak_active_uploads = 0
        self.uploaded_parts: dict[int, bytes] = {}
        self.completed_parts: list[dict[str, int | str]] | None = None
        self.aborted = False
        self._concurrent_uploads = threading.Event()
        self._lock = threading.Lock()

    def split_path(self, path: str) -> tuple[str, str, None]:
        _protocol, location = path.split("://", maxsplit=1)
        bucket, key = location.split("/", maxsplit=1)
        return bucket, key, None

    def makedirs(self, _path: str, exist_ok: bool = False) -> None:
        pass

    def exists(self, _path: str) -> bool:
        return False

    def call_s3(self, method: str, **kwargs):
        if method == "create_multipart_upload":
            return {"UploadId": "upload-1"}
        if method == "upload_part":
            part_number = int(kwargs["PartNumber"])
            self.upload_attempts += 1
            if self.transient_failures:
                self.transient_failures -= 1
                raise FSTimeoutError("injected transient UploadPart timeout")
            if part_number == self.fail_part:
                raise OSError("injected UploadPart failure")
            with self._lock:
                self.active_uploads += 1
                self.peak_active_uploads = max(self.peak_active_uploads, self.active_uploads)
                if self.active_uploads == 2:
                    self._concurrent_uploads.set()
            if self.fail_part is None:
                assert self._concurrent_uploads.wait(timeout=5)
            self.uploaded_parts[part_number] = bytes(kwargs["Body"])
            with self._lock:
                self.active_uploads -= 1
            return {"ETag": f"etag-{part_number}"}
        if method == "complete_multipart_upload":
            self.completed_parts = kwargs["MultipartUpload"]["Parts"]
            return {}
        if method == "abort_multipart_upload":
            self.aborted = True
            self._concurrent_uploads.set()
            return {}
        if method == "put_object":
            raise AssertionError("multipart test unexpectedly used PutObject")
        raise AssertionError(f"unexpected S3 method: {method}")


def test_concurrent_s3_stream_bounds_and_parallelizes_parts():
    filesystem = _RecordingMultipartFilesystem()
    part_bytes = _MINIMUM_S3_MULTIPART_PART_BYTES
    stream = _ConcurrentS3WriteStream(
        filesystem,
        "s3://bucket/checkpoint/__0_0.distcp",
        part_bytes=part_bytes,
        concurrency=2,
    )

    stream.write(b"a" * part_bytes + b"b" * part_bytes + b"tail")
    stream.close()

    assert filesystem.peak_active_uploads == 2
    assert filesystem.uploaded_parts == {1: b"a" * part_bytes, 2: b"b" * part_bytes, 3: b"tail"}
    assert filesystem.completed_parts == [
        {"PartNumber": 1, "ETag": "etag-1"},
        {"PartNumber": 2, "ETag": "etag-2"},
        {"PartNumber": 3, "ETag": "etag-3"},
    ]
    assert not filesystem.aborted


def test_concurrent_s3_stream_aborts_failed_part():
    filesystem = _RecordingMultipartFilesystem(fail_part=1)
    stream = _ConcurrentS3WriteStream(
        filesystem,
        "s3://bucket/checkpoint/__0_0.distcp",
        part_bytes=_MINIMUM_S3_MULTIPART_PART_BYTES,
        concurrency=1,
    )

    stream.write(b"a" * _MINIMUM_S3_MULTIPART_PART_BYTES)
    with pytest.raises(OSError, match="injected UploadPart failure"):
        stream.close()

    assert filesystem.aborted
    assert stream.closed


def test_concurrent_s3_stream_retries_transient_part(monkeypatch):
    monkeypatch.setattr("skyrl_train.io.s3fs.time.sleep", lambda _delay: None)
    filesystem = _RecordingMultipartFilesystem(transient_failures=1)
    stream = _ConcurrentS3WriteStream(
        filesystem,
        "s3://bucket/checkpoint/__0_0.distcp",
        part_bytes=_MINIMUM_S3_MULTIPART_PART_BYTES,
        concurrency=2,
    )

    stream.write(b"a" * _MINIMUM_S3_MULTIPART_PART_BYTES + b"b" * _MINIMUM_S3_MULTIPART_PART_BYTES)
    stream.close()

    assert filesystem.upload_attempts == 3
    assert filesystem.uploaded_parts == {
        1: b"a" * _MINIMUM_S3_MULTIPART_PART_BYTES,
        2: b"b" * _MINIMUM_S3_MULTIPART_PART_BYTES,
    }
    assert not filesystem.aborted


def test_streaming_fsspec_writer_preserves_upload_part_failure():
    filesystem = _RecordingMultipartFilesystem(fail_part=1)
    writer = StreamingFsspecWriter(
        "s3://bucket/checkpoint",
        filesystem=filesystem,
        tensor_copy_ahead_bytes=2**20,
        multipart_part_bytes=_MINIMUM_S3_MULTIPART_PART_BYTES,
        multipart_concurrency=1,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with pytest.raises(CheckpointException, match="injected UploadPart failure"):
            checkpoint.save(
                {"tensor": torch.arange(_MINIMUM_S3_MULTIPART_PART_BYTES + 1, dtype=torch.uint8)},
                storage_writer=writer,
            )

    assert filesystem.aborted


def test_s3_checkpoint_stream_round_trips_torch_save_and_aborts_failed_write(monkeypatch):
    class Filesystem:
        protocol = "s3"

        def __init__(self):
            self.objects = {}

        def split_path(self, path):
            bucket, key = path.removeprefix("s3://").split("/", 1)
            return bucket, key, None

        def call_s3(self, method, **kwargs):
            if method != "put_object":
                raise AssertionError(f"Unexpected S3 operation: {method}")
            self.objects[kwargs["Key"]] = kwargs["Body"]

    filesystem = Filesystem()
    monkeypatch.setattr("skyrl_train.io.torch_distributed_checkpoint.get_s3_fs", lambda: filesystem)
    monkeypatch.setattr("skyrl_train.io.torch_distributed_checkpoint.s3_refresh_if_expiring", lambda _fs: None)
    state = {"weight": torch.arange(8)}
    path = "s3://bucket/checkpoint/model_world_size_1_rank_0.pt"

    with open_s3_checkpoint_write_stream(path) as stream:
        torch.save(state, stream)
    restored = torch.load(io.BytesIO(filesystem.objects["checkpoint/model_world_size_1_rank_0.pt"]))
    assert torch.equal(restored["weight"], state["weight"])

    failed_path = "s3://bucket/checkpoint/optim_world_size_1_rank_0.pt"
    with pytest.raises(OSError, match="injected serialization failure"):
        with open_s3_checkpoint_write_stream(failed_path) as stream:
            stream.write(b"partial")
            raise OSError("injected serialization failure")
    assert "checkpoint/optim_world_size_1_rank_0.pt" not in filesystem.objects


def test_s3_checkpoint_stream_round_trips_multipart_torch_save(monkeypatch):
    filesystem = _RecordingMultipartFilesystem()
    monkeypatch.setattr("skyrl_train.io.torch_distributed_checkpoint.get_s3_fs", lambda: filesystem)
    monkeypatch.setattr("skyrl_train.io.torch_distributed_checkpoint.s3_refresh_if_expiring", lambda _fs: None)
    monkeypatch.setattr(
        "skyrl_train.io.torch_distributed_checkpoint.DEFAULT_S3_MULTIPART_PART_BYTES",
        _MINIMUM_S3_MULTIPART_PART_BYTES,
    )
    state = {"weight": torch.arange(12 * 2**20, dtype=torch.uint8)}

    with open_s3_checkpoint_write_stream("s3://bucket/checkpoint/model_world_size_1_rank_0.pt") as stream:
        torch.save(state, stream)

    uploaded = b"".join(filesystem.uploaded_parts[number] for number in sorted(filesystem.uploaded_parts))
    restored = torch.load(io.BytesIO(uploaded))
    assert torch.equal(restored["weight"], state["weight"])
    assert filesystem.completed_parts is not None
    assert not filesystem.aborted


def test_s3_checkpoint_stream_aborts_multipart_after_serialization_failure(monkeypatch):
    filesystem = _RecordingMultipartFilesystem()
    monkeypatch.setattr("skyrl_train.io.torch_distributed_checkpoint.get_s3_fs", lambda: filesystem)
    monkeypatch.setattr("skyrl_train.io.torch_distributed_checkpoint.s3_refresh_if_expiring", lambda _fs: None)
    monkeypatch.setattr(
        "skyrl_train.io.torch_distributed_checkpoint.DEFAULT_S3_MULTIPART_PART_BYTES",
        _MINIMUM_S3_MULTIPART_PART_BYTES,
    )

    with pytest.raises(OSError, match="injected serialization failure"):
        with open_s3_checkpoint_write_stream("s3://bucket/checkpoint/optim_world_size_1_rank_0.pt") as stream:
            stream.write(b"a" * _MINIMUM_S3_MULTIPART_PART_BYTES)
            raise OSError("injected serialization failure")

    assert filesystem.aborted
    assert filesystem.completed_parts is None
