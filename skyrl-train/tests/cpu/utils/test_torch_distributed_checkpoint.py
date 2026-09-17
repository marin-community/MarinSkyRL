import warnings

import fsspec
from fsspec import AbstractFileSystem
import pytest
import torch
from torch.distributed import checkpoint
from torch.distributed.checkpoint.api import CheckpointException

from skyrl_train.io.torch_distributed_checkpoint import StreamingFsspecWriter


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
