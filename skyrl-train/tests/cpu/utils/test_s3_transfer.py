from contextlib import contextmanager
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

from skyrl_train.io import io, s3fs


_MULTIPART_TEST_FILE_SIZE = 100 * 2**20
_UPLOAD_PROCESS_TIMEOUT = 10
_UPLOAD_REQUEST_TIMEOUT = 0.2


@contextmanager
def _withholding_s3_endpoint():
    headers_seen = threading.Event()

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

        def do_DELETE(self):
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()

    server = ThreadingHTTPServer(("127.0.0.1", 0), WithholdingS3Handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", headers_seen
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)


def _upload_to_withholding_endpoint(endpoint: str, checkpoint_shard: str, result_sender: Connection) -> None:
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
    filesystem = s3fs.get_s3_fs()
    filesystem.retries = 1

    started = time.monotonic()
    try:
        io.upload_file(checkpoint_shard, "s3://bucket/checkpoint.distcp")
    except (FSTimeoutError, ReadTimeoutError, TimeoutError) as error:
        result_sender.send((type(error).__name__, time.monotonic() - started, getattr(error, "__notes__", [])))
    else:
        result_sender.send((None, time.monotonic() - started, None))
    finally:
        result_sender.close()


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

    with _withholding_s3_endpoint() as (endpoint, headers_seen):
        context = multiprocessing.get_context("spawn")
        result_receiver, result_sender = context.Pipe(duplex=False)
        process = context.Process(
            target=_upload_to_withholding_endpoint,
            args=(endpoint, str(checkpoint_shard), result_sender),
            daemon=True,
        )
        process.start()
        result_sender.close()
        process.join(timeout=_UPLOAD_PROCESS_TIMEOUT)
        if process.is_alive():
            process.terminate()
            process.join(timeout=1)
            if process.is_alive():
                process.kill()
                process.join(timeout=1)
            pytest.fail(f"multipart upload exceeded the {_UPLOAD_PROCESS_TIMEOUT}-second process deadline")

        assert process.exitcode == 0
        assert result_receiver.poll(), "upload process exited without a result"
        error_type, elapsed, error_notes = result_receiver.recv()
        result_receiver.close()

    assert headers_seen.is_set(), "the peer did not receive an Expect request"
    assert error_type in {"FSTimeoutError", "ReadTimeoutError", "TimeoutError"}
    assert any(str(checkpoint_shard) in note for note in error_notes)
    assert any("s3://bucket/checkpoint.distcp" in note for note in error_notes)
    assert 0.1 < elapsed < 10


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
