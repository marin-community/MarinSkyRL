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
from types import SimpleNamespace

from botocore.exceptions import ClientError, ReadTimeoutError
from fsspec.exceptions import FSTimeoutError
import pytest
import rigging.filesystem.s3_compat as rigging_s3_compat
import torch
from torch.distributed import checkpoint
from torch.distributed.checkpoint.api import CheckpointException

from skyrl_train.io import io, s3fs
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
    request_bounds = rigging_s3_compat.s3_python_config_kwargs()
    request_bounds["retries"] = {"total_max_attempts": 1, "mode": "standard"}
    s3fs.s3_python_config_kwargs = request_bounds.copy
    s3fs._S3_FS = None
    os.environ.update(
        {
            "AWS_ACCESS_KEY_ID": "test",
            "AWS_SECRET_ACCESS_KEY": "test",
            "AWS_DEFAULT_REGION": "us-east-1",
            "AWS_ENDPOINT_URL": endpoint,
            "OT_AGENT_S3_ADDRESSING_STYLE": "path",
            "NO_PROXY": "127.0.0.1",
            "no_proxy": "127.0.0.1",
        }
    )
    return s3fs.get_s3_fs()


def _upload_to_withholding_endpoint(endpoint: str, checkpoint_shard: str, result_sender: Connection) -> None:
    filesystem = _configure_withholding_s3(endpoint)
    filesystem.retries = 1

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


def test_s3_client_uses_shared_request_bounds_and_bounded_retries(monkeypatch):
    sentinel = SimpleNamespace(retries=None)
    shared_request_bounds = {
        "connect_timeout": object(),
        "read_timeout": object(),
        "max_pool_connections": object(),
        "http_session_cls": object(),
    }
    calls = []
    monkeypatch.setattr(s3fs, "_S3_FS", None)
    monkeypatch.delenv("OT_AGENT_S3_ADDRESSING_STYLE", raising=False)
    monkeypatch.setattr(s3fs, "s3_python_config_kwargs", lambda: shared_request_bounds.copy())
    monkeypatch.setattr(
        s3fs.fsspec,
        "filesystem",
        lambda protocol, **kwargs: calls.append((protocol, kwargs)) or sentinel,
    )

    assert s3fs.get_s3_fs() is sentinel
    assert calls == [
        (
            "s3",
            {
                "config_kwargs": {
                    **shared_request_bounds,
                    "retries": {"total_max_attempts": 2, "mode": "standard"},
                    "s3": {"addressing_style": "virtual"},
                }
            },
        )
    ]
    assert sentinel.retries == 1


def test_s3_client_allows_addressing_style_override(monkeypatch):
    calls = []
    sentinel = SimpleNamespace(retries=None)
    monkeypatch.setattr(s3fs, "_S3_FS", None)
    monkeypatch.setenv("OT_AGENT_S3_ADDRESSING_STYLE", "path")
    monkeypatch.setattr(
        s3fs.fsspec,
        "filesystem",
        lambda protocol, **kwargs: calls.append((protocol, kwargs)) or sentinel,
    )

    s3fs.get_s3_fs()

    assert calls[0][1]["config_kwargs"]["s3"] == {"addressing_style": "path"}


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
    assert endpoint.aborts_seen.is_set(), "the failed multipart upload was not aborted"
    assert result.error_type == "CheckpointException"
    assert result.failed_ranks == (0,)
    assert any("s3://bucket/checkpoint/__0_0.distcp" in note for note in result.error_notes)
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
    monkeypatch.setattr(s3fs, "get_s3_fs", lambda: filesystem)
    monkeypatch.setattr(s3fs, "s3_refresh_if_expiring", lambda _filesystem: None)

    assert s3fs.abort_multipart_uploads("s3://bucket/checkpoints/global_step_4/policy") == 2
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
        s3fs.abort_multipart_uploads("s3://bucket/checkpoints/global_step_4/policy//")


@pytest.mark.parametrize(
    ("transfer_error", "expected_refreshes"),
    [
        (FSTimeoutError("injected fsspec timeout"), 0),
        (ReadTimeoutError(endpoint_url="https://bucket.invalid/shard.pt", error="injected SDK timeout"), 0),
        (OSError(5, "An error occurred (Forbidden) when calling ListObjectsV2: AccessDenied"), 2),
        (
            ClientError(
                {"Error": {"Code": "AccessDenied"}, "ResponseMetadata": {"HTTPStatusCode": 403}},
                "ListObjectsV2",
            ),
            2,
        ),
    ],
)
def test_s3_transfer_retries_retryable_failure_with_backoff(
    monkeypatch,
    transfer_error,
    expected_refreshes,
):
    class RefreshableFilesystem:
        def __init__(self):
            self.refreshes = 0

        def connect(self, *, refresh):
            assert refresh is True
            self.refreshes += 1

    filesystem = RefreshableFilesystem()
    attempts = 0
    delays = []

    def flaky_transfer():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise transfer_error
        return "complete"

    monkeypatch.setattr(s3fs.time, "sleep", delays.append)
    monkeypatch.setattr(s3fs.random, "uniform", lambda low, high: 1.0)

    assert s3fs.call_with_s3_retry(filesystem, flaky_transfer) == "complete"
    assert attempts == 3
    assert filesystem.refreshes == expected_refreshes
    assert delays == [1.0, 2.0]


def test_s3_transfer_raises_after_access_denied_retry_budget(monkeypatch):
    attempts = 0

    def denied_transfer():
        nonlocal attempts
        attempts += 1
        raise OSError(5, "Forbidden: AccessDenied")

    monkeypatch.setattr(s3fs.time, "sleep", lambda _delay: None)

    with pytest.raises(OSError, match="AccessDenied"):
        s3fs.call_with_s3_retry(object(), denied_transfer)

    assert attempts == 5


def test_s3_transfer_does_not_retry_unrelated_oserror(monkeypatch):
    attempts = 0

    def denied_local_operation():
        nonlocal attempts
        attempts += 1
        raise PermissionError("Forbidden local path")

    with pytest.raises(PermissionError, match="Forbidden local path"):
        s3fs.call_with_s3_retry(object(), denied_local_operation)

    assert attempts == 1


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
