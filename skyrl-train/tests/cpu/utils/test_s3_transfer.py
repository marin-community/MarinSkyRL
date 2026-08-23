from pathlib import Path

from botocore.exceptions import ClientError, ReadTimeoutError
from fsspec.exceptions import FSTimeoutError
import pytest

from skyrl_train.utils.io import io, s3fs


def test_s3_client_has_explicit_transfer_timeouts_and_retries(monkeypatch):
    sentinel = object()
    calls = []
    monkeypatch.setattr(s3fs, "_S3_FS", None)
    monkeypatch.delenv("OT_AGENT_S3_ADDRESSING_STYLE", raising=False)
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
                    "connect_timeout": 60,
                    "read_timeout": 300,
                    "retries": {"max_attempts": 10, "mode": "adaptive"},
                    "s3": {"addressing_style": "virtual"},
                }
            },
        )
    ]


def test_s3_client_allows_addressing_style_override(monkeypatch):
    calls = []
    monkeypatch.setattr(s3fs, "_S3_FS", None)
    monkeypatch.setenv("OT_AGENT_S3_ADDRESSING_STYLE", "path")
    monkeypatch.setattr(
        s3fs.fsspec,
        "filesystem",
        lambda protocol, **kwargs: calls.append((protocol, kwargs)) or object(),
    )

    s3fs.get_s3_fs()

    assert calls[0][1]["config_kwargs"]["s3"] == {"addressing_style": "path"}


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
