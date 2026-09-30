from contextlib import contextmanager
import copy
import multiprocessing
from pathlib import Path
import threading
import warnings

import fsspec
from fsspec import AbstractFileSystem
import pytest
import torch
from torch.distributed import checkpoint
from torch.distributed.checkpoint.api import CheckpointException
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.planner import LoadItemType, LoadPlan, ReadItem
from torch.distributed.checkpoint.metadata import MetadataIndex

from marinskyrl.remote_io import S3MultipartWriteStream
from skyrl_train.io.torch_distributed_checkpoint import StreamingFsspecWriter
from skyrl_train.io.checkpoint_reader import BudgetedCheckpointReader, PodCheckpointReadBudget


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


def test_budgeted_checkpoint_restores_adam_and_continues_the_same_update(tmp_path):
    model = torch.nn.Linear(5, 3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    inputs = torch.arange(10, dtype=torch.float32).reshape(2, 5) / 10

    def update(model, optimizer):
        optimizer.zero_grad()
        model(inputs).square().sum().backward()
        optimizer.step()

    for _ in range(8):
        update(model, optimizer)
    saved = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": 8, "rng": torch.get_rng_state()}
    checkpoint.save(saved, checkpoint_id=tmp_path / "checkpoint", no_dist=True)
    restored = copy.deepcopy(saved)
    restored["step"] = 0
    for tensor in restored["model"].values():
        tensor.zero_()
    for state in restored["optimizer"]["state"].values():
        for tensor in state.values():
            tensor.zero_()
    restored["rng"].zero_()
    checkpoint.load(
        restored,
        storage_reader=BudgetedCheckpointReader(
            str(tmp_path / "checkpoint"), PodCheckpointReadBudget(2**20, tmp_path / "budget")
        ),
        no_dist=True,
    )
    assert restored["step"] == 8
    assert torch.equal(restored["rng"], saved["rng"])
    for key, value in saved["model"].items():
        assert torch.equal(restored["model"][key], value)
    for key, state in saved["optimizer"]["state"].items():
        for field, value in state.items():
            assert torch.equal(restored["optimizer"]["state"][key][field], value)

    resumed = torch.nn.Linear(5, 3)
    resumed.load_state_dict(restored["model"])
    resumed_optimizer = torch.optim.Adam(resumed.parameters(), lr=0.01)
    resumed_optimizer.load_state_dict(restored["optimizer"])
    update(model, optimizer)
    update(resumed, resumed_optimizer)
    for expected, actual in zip(model.parameters(), resumed.parameters(), strict=True):
        assert torch.equal(expected, actual)


def test_budgeted_checkpoint_slice_accounts_for_the_whole_saved_tensor(tmp_path):
    source = torch.arange(8 * 2**20, dtype=torch.float32)
    checkpoint.save({"tensor": source}, checkpoint_id=tmp_path / "checkpoint", no_dist=True)
    offset = 3 * 2**20
    restored = {"tensor": torch.full((2**20,), -1.0)}
    planner = DefaultLoadPlanner()
    plan = LoadPlan(
        items=[
            ReadItem(
                type=LoadItemType.TENSOR,
                dest_index=MetadataIndex("tensor"),
                dest_offsets=torch.Size([0]),
                storage_index=MetadataIndex("tensor", torch.Size([0])),
                storage_offsets=torch.Size([offset]),
                lengths=restored["tensor"].size(),
            )
        ]
    )
    # The requested slice is 4 MiB, but deserializing the saved 32 MiB tensor
    # needs its full storage plus read and copy buffers. A 64 MiB budget must fail.
    for budget_mib in (64, 128):
        reader = BudgetedCheckpointReader(
            str(tmp_path / "checkpoint"), PodCheckpointReadBudget(budget_mib * 2**20, tmp_path / "budget")
        )
        metadata = reader.read_metadata()
        reader.set_up_storage_reader(metadata, is_coordinator=True)
        planner.set_up_planner(restored, metadata, is_coordinator=True)
        if budget_mib == 64:
            with pytest.raises(ValueError):
                reader.read_data(plan, planner).wait()
            assert torch.equal(restored["tensor"], torch.full_like(restored["tensor"], -1.0))
        else:
            reader.read_data(plan, planner).wait()
            assert torch.equal(restored["tensor"], source[offset : offset + 2**20])


def _hold_checkpoint_reservation(directory, amount, connection):
    budget = PodCheckpointReadBudget(100, Path(directory))
    connection.send("attempting")
    try:
        with budget.reserve(amount):
            connection.send("admitted")
            if connection.recv() == "fail":
                raise OSError("injected read failure")
    except OSError:
        connection.send("failed")
    else:
        connection.send("released")


@contextmanager
def _checkpoint_reservation_process(directory, amount):
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=_hold_checkpoint_reservation, args=(str(directory), amount, child))
    process.start()
    child.close()
    try:
        assert parent.poll(30), "reservation worker did not start"
        assert parent.recv() == "attempting"
        yield process, parent
    finally:
        if process.is_alive():
            process.terminate()
        process.join(10)
        if process.is_alive():
            process.kill()
            process.join(10)
        parent.close()
        assert not process.is_alive()


def test_pod_checkpoint_budget_admits_unequal_records_without_exceeding_limit(tmp_path):
    with _checkpoint_reservation_process(tmp_path, 60) as (_, large):
        assert large.poll(30) and large.recv() == "admitted"
        with _checkpoint_reservation_process(tmp_path, 60) as (_, waiting):
            # Holding the first record is the test input; two large records must
            # not be admitted together, while a smaller peer can still fit.
            assert not waiting.poll(0.2)
            with _checkpoint_reservation_process(tmp_path, 30) as (_, small):
                assert small.poll(30) and small.recv() == "admitted"
                large.send("release")
                assert large.poll(30) and large.recv() == "released"
                assert waiting.poll(30) and waiting.recv() == "admitted"
                small.send("release")
                waiting.send("release")
                assert small.poll(30) and small.recv() == "released"
                assert waiting.poll(30) and waiting.recv() == "released"


@pytest.mark.parametrize("end", ["fail", "kill"])
def test_pod_checkpoint_budget_releases_after_read_failure_or_process_exit(tmp_path, end):
    with _checkpoint_reservation_process(tmp_path, 100) as (holder_process, holder):
        assert holder.poll(30) and holder.recv() == "admitted"
        with _checkpoint_reservation_process(tmp_path, 100) as (_, peer):
            assert not peer.poll(0.2)
            if end == "kill":
                holder_process.kill()
                holder_process.join(10)
                assert holder_process.exitcode != 0
            else:
                holder.send("fail")
                assert holder.poll(30) and holder.recv() == "failed"
            assert peer.poll(30) and peer.recv() == "admitted"
            peer.send("release")
            assert peer.poll(30) and peer.recv() == "released"
