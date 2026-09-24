import threading
import warnings

import fsspec
from fsspec import AbstractFileSystem
import pytest
import torch
from torch.distributed import checkpoint
from torch.distributed.checkpoint.api import CheckpointException

from marinskyrl.remote_io import S3MultipartWriteStream
from skyrl_train.io.torch_distributed_checkpoint import StreamingFsspecWriter


_TEST_PART_BYTES = 5 * 2**20


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
        self.closed = False

    def write(self, _payload) -> int:
        raise OSError("injected object-store failure")

    def tell(self) -> int:
        return 0

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


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


def test_streaming_fsspec_writer_closes_failed_object():
    filesystem = _FailingFilesystem()
    writer = StreamingFsspecWriter("memory://failed-checkpoint/step", filesystem=filesystem)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with pytest.raises(CheckpointException):
            checkpoint.save({"tensor": torch.arange(8)}, storage_writer=writer)

    assert filesystem.streams
    assert all(stream.closed for stream in filesystem.streams)
    assert not any(path.endswith(".metadata") for path in filesystem.opened_paths)


class _RecordingMultipartFilesystem:
    protocol = "s3"

    def __init__(self, fail_part: int | None = None) -> None:
        self.fail_part = fail_part
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


def test_concurrent_s3_stream_bounds_and_parallelizes_parts(monkeypatch):
    monkeypatch.setattr(S3MultipartWriteStream, "part_bytes", _TEST_PART_BYTES)
    monkeypatch.setattr(S3MultipartWriteStream, "concurrency", 2)
    filesystem = _RecordingMultipartFilesystem()
    stream = S3MultipartWriteStream(filesystem, "s3://bucket/checkpoint/__0_0.distcp")

    stream.write(b"a" * _TEST_PART_BYTES + b"b" * _TEST_PART_BYTES + b"tail")
    stream.commit()

    assert filesystem.peak_active_uploads == 2
    assert filesystem.uploaded_parts == {1: b"a" * _TEST_PART_BYTES, 2: b"b" * _TEST_PART_BYTES, 3: b"tail"}
    assert filesystem.completed_parts == [
        {"PartNumber": 1, "ETag": "etag-1"},
        {"PartNumber": 2, "ETag": "etag-2"},
        {"PartNumber": 3, "ETag": "etag-3"},
    ]
    assert not filesystem.aborted


def test_concurrent_s3_stream_aborts_failed_part(monkeypatch):
    monkeypatch.setattr(S3MultipartWriteStream, "part_bytes", _TEST_PART_BYTES)
    monkeypatch.setattr(S3MultipartWriteStream, "concurrency", 1)
    filesystem = _RecordingMultipartFilesystem(fail_part=1)
    stream = S3MultipartWriteStream(filesystem, "s3://bucket/checkpoint/__0_0.distcp")

    with pytest.raises(OSError, match="injected UploadPart failure"):
        stream.write(b"a" * _TEST_PART_BYTES)

    assert filesystem.aborted
    assert stream.closed


def test_concurrent_s3_stream_close_aborts_uncommitted_upload(monkeypatch):
    monkeypatch.setattr(S3MultipartWriteStream, "part_bytes", _TEST_PART_BYTES)
    monkeypatch.setattr(S3MultipartWriteStream, "concurrency", 2)
    filesystem = _RecordingMultipartFilesystem()
    stream = S3MultipartWriteStream(filesystem, "s3://bucket/checkpoint/__0_0.distcp")

    stream.write(b"a" * _TEST_PART_BYTES + b"b" * _TEST_PART_BYTES)
    stream.close()

    assert filesystem.aborted
    assert filesystem.completed_parts is None
    assert stream.closed


def test_streaming_fsspec_writer_preserves_upload_part_failure(monkeypatch):
    monkeypatch.setattr(S3MultipartWriteStream, "part_bytes", _TEST_PART_BYTES)
    monkeypatch.setattr(S3MultipartWriteStream, "concurrency", 1)
    filesystem = _RecordingMultipartFilesystem(fail_part=1)
    writer = StreamingFsspecWriter(
        "s3://bucket/checkpoint",
        filesystem=filesystem,
        tensor_copy_ahead_bytes=2**20,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with pytest.raises(CheckpointException, match="injected UploadPart failure"):
            checkpoint.save(
                {"tensor": torch.arange(_TEST_PART_BYTES + 1, dtype=torch.uint8)},
                storage_writer=writer,
            )

    assert filesystem.aborted


def test_checkpoint_stream_progresses_past_a_slow_first_part(monkeypatch):
    monkeypatch.setattr(S3MultipartWriteStream, "part_bytes", _TEST_PART_BYTES)

    class _SlowFirstPartFilesystem(_RecordingMultipartFilesystem):
        def __init__(self):
            super().__init__()
            self.third_part_started = threading.Event()

        def call_s3(self, method: str, **kwargs):
            if method == "upload_part":
                part = int(kwargs["PartNumber"])
                if part == 1 and not self.third_part_started.wait(timeout=3):
                    raise TimeoutError("part three never started while part one was slow")
                if part == 3:
                    self.third_part_started.set()
                self.uploaded_parts[part] = bytes(kwargs["Body"])
                return {"ETag": f"etag-{part}"}
            return super().call_s3(method, **kwargs)

    filesystem = _SlowFirstPartFilesystem()
    stream = S3MultipartWriteStream(
        filesystem,
        "s3://bucket/checkpoint/__0_0.distcp",
        concurrency=2,
        complete_out_of_order=True,
        wait_before_abort=True,
    )

    stream.write(b"a" * _TEST_PART_BYTES + b"b" * _TEST_PART_BYTES + b"c" * _TEST_PART_BYTES)
    stream.commit()

    assert filesystem.uploaded_parts == {
        1: b"a" * _TEST_PART_BYTES,
        2: b"b" * _TEST_PART_BYTES,
        3: b"c" * _TEST_PART_BYTES,
    }
    assert filesystem.completed_parts == [
        {"PartNumber": 1, "ETag": "etag-1"},
        {"PartNumber": 2, "ETag": "etag-2"},
        {"PartNumber": 3, "ETag": "etag-3"},
    ]


def test_checkpoint_stream_waits_for_inflight_part_before_abort(monkeypatch):
    monkeypatch.setattr(S3MultipartWriteStream, "part_bytes", _TEST_PART_BYTES)

    class _FailedPartFilesystem(_RecordingMultipartFilesystem):
        def __init__(self):
            super().__init__()
            self.first_part_started = threading.Event()
            self.second_part_failed = threading.Event()
            self.first_part_finished = threading.Event()

        def call_s3(self, method: str, **kwargs):
            if method == "upload_part":
                part = int(kwargs["PartNumber"])
                if part == 1:
                    self.first_part_started.set()
                    if not self.second_part_failed.wait(timeout=3):
                        raise TimeoutError("second part never failed")
                    self.first_part_finished.set()
                    return {"ETag": "etag-1"}
                if not self.first_part_started.wait(timeout=3):
                    raise TimeoutError("first part never started")
                self.second_part_failed.set()
                raise OSError("injected second-part failure")
            if method == "abort_multipart_upload":
                assert self.first_part_finished.is_set(), "abort raced an in-flight UploadPart"
            return super().call_s3(method, **kwargs)

    filesystem = _FailedPartFilesystem()
    stream = S3MultipartWriteStream(
        filesystem,
        "s3://bucket/checkpoint/__0_0.distcp",
        concurrency=2,
        complete_out_of_order=True,
        wait_before_abort=True,
    )

    with pytest.raises(OSError, match="injected second-part failure"):
        stream.write(b"a" * _TEST_PART_BYTES + b"b" * _TEST_PART_BYTES)

    assert filesystem.aborted
    assert filesystem.completed_parts is None
    assert stream.closed


def test_dcp_writer_uses_eight_concurrent_parts(monkeypatch):
    monkeypatch.setattr(S3MultipartWriteStream, "part_bytes", _TEST_PART_BYTES)

    class _EightWayFilesystem(_RecordingMultipartFilesystem):
        def __init__(self):
            super().__init__()
            self.all_parts_started = threading.Event()
            self.started = 0

        def call_s3(self, method: str, **kwargs):
            if method == "upload_part":
                with self._lock:
                    self.started += 1
                    if self.started == 8:
                        self.all_parts_started.set()
                if not self.all_parts_started.wait(timeout=3):
                    raise TimeoutError("fewer than eight checkpoint parts started")
                part = int(kwargs["PartNumber"])
                return {"ETag": f"etag-{part}"}
            return super().call_s3(method, **kwargs)

    filesystem = _EightWayFilesystem()
    writer = StreamingFsspecWriter("s3://bucket/checkpoint", filesystem=filesystem)
    with writer.fs.create_stream("s3://bucket/checkpoint/__0_0.distcp", "wb") as stream:
        stream.write(b"x" * (_TEST_PART_BYTES * 8))

    assert filesystem.started == 8
    assert filesystem.completed_parts is not None
    assert len(filesystem.completed_parts) == 8
    assert [part["PartNumber"] for part in filesystem.completed_parts] == list(range(1, 9))
