from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import multiprocessing
from multiprocessing.connection import Connection
import os
from pathlib import Path
import threading
import time

from botocore.exceptions import ReadTimeoutError
from fsspec.exceptions import FSTimeoutError
import pytest
import rigging.filesystem.s3_compat as rigging_s3_compat
import torch
from torch.distributed import checkpoint
from torch.distributed.checkpoint.api import CheckpointException

from marinskyrl import remote_io
from skyrl_train.io import io
from skyrl_train.io.torch_distributed_checkpoint import StreamingFsspecWriter


_MULTIPART_TEST_FILE_SIZE = 100 * 2**20
_UPLOAD_PROCESS_TIMEOUT = 10
_UPLOAD_REQUEST_TIMEOUT = 0.2


@dataclass(frozen=True)
class _WithholdingEndpoint:
    url: str
    headers_seen: threading.Event
    aborts_seen: threading.Event


@dataclass(frozen=True)
class _UploadResult:
    error_type: str | None
    elapsed: float
    error_notes: tuple[str, ...] = ()
    failed_ranks: tuple[int, ...] = ()


@contextmanager
def _withholding_s3_endpoint() -> Generator[_WithholdingEndpoint, None, None]:
    headers_seen = threading.Event()
    aborts_seen = threading.Event()

    class WithholdingS3Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, _format, *args):
            pass

        def handle_expect_100(self):
            headers_seen.set()
            return False

        def do_POST(self):
            payload = (
                b'<?xml version="1.0" encoding="UTF-8"?>'
                b'<CreateMultipartUploadResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
                b"<Bucket>bucket</Bucket><Key>checkpoint.distcp</Key>"
                b"<UploadId>withheld-upload</UploadId></CreateMultipartUploadResult>"
            )
            self.send_response(200)
            self.send_header("Content-Type", "application/xml")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            payload = (
                b'<?xml version="1.0" encoding="UTF-8"?>'
                b'<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
                b"<Name>bucket</Name><Prefix>checkpoint.distcp/</Prefix><KeyCount>0</KeyCount>"
                b"<MaxKeys>1</MaxKeys><IsTruncated>false</IsTruncated></ListBucketResult>"
            )
            self.send_response(200)
            self.send_header("Content-Type", "application/xml")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_HEAD(self):
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_DELETE(self):
            aborts_seen.set()
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()

    server = ThreadingHTTPServer(("127.0.0.1", 0), WithholdingS3Handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    try:
        yield _WithholdingEndpoint(
            url=f"http://127.0.0.1:{server.server_port}",
            headers_seen=headers_seen,
            aborts_seen=aborts_seen,
        )
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)


def _configure_withholding_s3(endpoint: str):
    rigging_s3_compat._S3_TOTAL_TIMEOUT = _UPLOAD_REQUEST_TIMEOUT
    os.environ.update(
        {
            "AWS_ACCESS_KEY_ID": "test",
            "AWS_SECRET_ACCESS_KEY": "test",
            "AWS_DEFAULT_REGION": "us-east-1",
            "AWS_ENDPOINT_URL": endpoint,
            "NO_PROXY": "127.0.0.1",
            "no_proxy": "127.0.0.1",
        }
    )
    return remote_io.create_s3_filesystem(
        endpoint_url=endpoint,
        client_kwargs={"region_name": "us-east-1"},
        config_kwargs={
            "retries": {"total_max_attempts": 1, "mode": "standard"},
            "s3": {"addressing_style": "path"},
        },
    )


def _upload_to_withholding_endpoint(endpoint: str, checkpoint_shard: str, result_sender: Connection) -> None:
    filesystem = _configure_withholding_s3(endpoint)
    filesystem.retries = 1
    io._get_filesystem = lambda _path: filesystem

    started = time.monotonic()
    try:
        io.upload_file(checkpoint_shard, "s3://bucket/checkpoint.distcp")
    except (FSTimeoutError, ReadTimeoutError, TimeoutError) as error:
        result_sender.send(
            _UploadResult(
                error_type=type(error).__name__,
                elapsed=time.monotonic() - started,
                error_notes=tuple(getattr(error, "__notes__", ())),
            )
        )
    else:
        result_sender.send(_UploadResult(error_type=None, elapsed=time.monotonic() - started))
    finally:
        result_sender.close()


def _stream_dcp_to_withholding_endpoint(endpoint: str, result_sender: Connection) -> None:
    filesystem = _configure_withholding_s3(endpoint)
    tensor = torch.zeros(_MULTIPART_TEST_FILE_SIZE, dtype=torch.uint8)

    started = time.monotonic()
    try:
        checkpoint.save(
            {"shard": tensor},
            storage_writer=StreamingFsspecWriter(
                "s3://bucket/checkpoint",
                filesystem=filesystem,
            ),
        )
    except CheckpointException as error:
        notes = tuple(note for failure, _trace in error.failures.values() for note in getattr(failure, "__notes__", ()))
        result_sender.send(
            _UploadResult(
                error_type=type(error).__name__,
                elapsed=time.monotonic() - started,
                error_notes=notes,
                failed_ranks=tuple(sorted(error.failures)),
            )
        )
    else:
        result_sender.send(_UploadResult(error_type=None, elapsed=time.monotonic() - started))
    finally:
        result_sender.close()


def _run_upload_process(target: Callable[..., None], *args: str) -> _UploadResult:
    context = multiprocessing.get_context("spawn")
    result_receiver, result_sender = context.Pipe(duplex=False)
    process = context.Process(target=target, args=(*args, result_sender), daemon=True)
    process.start()
    result_sender.close()
    process.join(timeout=_UPLOAD_PROCESS_TIMEOUT)
    if process.is_alive():
        process.terminate()
        process.join(timeout=1)
        if process.is_alive():
            process.kill()
            process.join(timeout=1)
        pytest.fail(f"upload exceeded the {_UPLOAD_PROCESS_TIMEOUT}-second process deadline")

    assert process.exitcode == 0
    assert result_receiver.poll(), "upload process exited without a result"
    result: _UploadResult = result_receiver.recv()
    result_receiver.close()
    return result


def test_s3_multipart_upload_fails_when_peer_withholds_continue(tmp_path):
    checkpoint_shard = tmp_path / "checkpoint.distcp"
    with checkpoint_shard.open("wb") as shard_file:
        shard_file.truncate(_MULTIPART_TEST_FILE_SIZE)

    with _withholding_s3_endpoint() as endpoint:
        result = _run_upload_process(_upload_to_withholding_endpoint, endpoint.url, str(checkpoint_shard))

    assert endpoint.headers_seen.is_set(), "the peer did not receive an Expect request"
    assert result.error_type in {"FSTimeoutError", "ReadTimeoutError", "TimeoutError"}
    assert any(str(checkpoint_shard) in note for note in result.error_notes)
    assert any("s3://bucket/checkpoint.distcp" in note for note in result.error_notes)
    assert 0.1 < result.elapsed < _UPLOAD_PROCESS_TIMEOUT


def test_streaming_dcp_upload_fails_and_aborts_when_peer_withholds_continue():
    with _withholding_s3_endpoint() as endpoint:
        result = _run_upload_process(_stream_dcp_to_withholding_endpoint, endpoint.url)

    assert endpoint.headers_seen.is_set(), "the peer did not receive an Expect request"
    assert endpoint.aborts_seen.is_set(), f"the failed multipart upload was not aborted: {result}"
    assert result.error_type == "CheckpointException"
    assert result.failed_ranks == (0,)
    assert any("s3://bucket/checkpoint/__0_0.distcp" in note for note in result.error_notes)
    assert any("upload_id=withheld-upload part=" in note for note in result.error_notes)
    assert 0.1 < result.elapsed < _UPLOAD_PROCESS_TIMEOUT


def test_abort_multipart_uploads_limits_cleanup_to_checkpoint_prefix(monkeypatch):
    class RecordingFilesystem:
        def __init__(self):
            self.calls = []

        def call_s3(self, operation, **kwargs):
            self.calls.append((operation, kwargs))
            if operation == "list_multipart_uploads":
                return {
                    "Uploads": [
                        {
                            "Key": "checkpoints/global_step_4/policy/__0_0.distcp",
                            "UploadId": "upload-1",
                        },
                        {
                            "Key": "checkpoints/global_step_4/policy/__1_0.distcp",
                            "UploadId": "upload-2",
                        },
                    ]
                }
            return {}

    filesystem = RecordingFilesystem()
    monkeypatch.setattr(remote_io, "create_s3_filesystem", lambda: filesystem)

    assert remote_io.abort_multipart_uploads("s3://bucket/checkpoints/global_step_4/policy") == 2
    assert filesystem.calls == [
        (
            "list_multipart_uploads",
            {"Bucket": "bucket", "Prefix": "checkpoints/global_step_4/policy/"},
        ),
        (
            "abort_multipart_upload",
            {
                "Bucket": "bucket",
                "Key": "checkpoints/global_step_4/policy/__0_0.distcp",
                "UploadId": "upload-1",
            },
        ),
        (
            "abort_multipart_upload",
            {
                "Bucket": "bucket",
                "Key": "checkpoints/global_step_4/policy/__1_0.distcp",
                "UploadId": "upload-2",
            },
        ),
    ]


def test_abort_multipart_uploads_rejects_noncanonical_checkpoint_path():
    with pytest.raises(ValueError):
        remote_io.abort_multipart_uploads("s3://bucket/checkpoints/global_step_4/policy//")


def test_local_read_files_downloads_only_requested_objects(monkeypatch):
    class RecordingFilesystem:
        def __init__(self):
            self.downloads = []

        def _strip_protocol(self, path):
            return path.removeprefix("s3://")

        def get(self, source, destination, recursive=False):
            self.downloads.append((source, recursive))
            Path(destination).write_text(source)

    filesystem = RecordingFilesystem()
    monkeypatch.setattr(io, "_get_filesystem", lambda path: filesystem)
    requested = [
        "s3://bucket/checkpoint/model_rank_2.pt",
        "s3://bucket/checkpoint/optim_rank_2.pt",
        "s3://bucket/checkpoint/extra_rank_2.pt",
    ]

    with io.local_read_files(requested) as local_paths:
        assert [Path(path).read_text() for path in local_paths] == [path.removeprefix("s3://") for path in requested]

    assert filesystem.downloads == [(path.removeprefix("s3://"), False) for path in requested]
