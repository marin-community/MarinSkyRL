"""Shared, bounded remote I/O for Hugging Face and object storage."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Generator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import errno
import os
from typing import Any, Protocol, TypeVar, runtime_checkable

from fsspec.spec import AbstractFileSystem
import httpcore
import httpx
from loguru import logger
import requests
from rigging.filesystem.factory import filesystem as guarded_filesystem
from rigging.filesystem.factory import url_to_fs as guarded_url_to_fs
from rigging.filesystem.s3_compat import s3_python_config_kwargs
from rigging.filesystem.s3_errors import is_transient_s3_error
from rigging.filesystem.storage_path import StoragePath
from rigging.timing import ExponentialBackoff, retry_with_backoff
import urllib3


T = TypeVar("T")

MINIMUM_S3_MULTIPART_PART_BYTES = 5 * 2**20
DEFAULT_S3_MULTIPART_PART_BYTES = 64 * 2**20
DEFAULT_S3_MULTIPART_CONCURRENCY = 4
DEFAULT_HF_MAX_RETRIES = 5
DEFAULT_HF_BACKOFF_BASE_SECONDS = 2.0
DEFAULT_HF_BACKOFF_CAP_SECONDS = 32.0

_S3_FILESYSTEM_RETRIES = 1
_S3_REQUEST_TOTAL_ATTEMPTS = 2
_S3_TRANSFER_MAX_ATTEMPTS = 5
_S3_MULTIPART_PART_MAX_ATTEMPTS = 2
_S3_ADDRESSING_STYLE_ENV = "OT_AGENT_S3_ADDRESSING_STYLE"
_RETRYABLE_AUTH_CODES = {
    "403",
    "AccessDenied",
    "ExpiredToken",
    "ExpiredTokenException",
    "Forbidden",
    "RequestExpired",
}
_RETRYABLE_TRANSLATED_ERROR_MARKERS = (
    "accessdenied",
    "forbidden",
    "internalerror",
    "requesttimeout",
    "serviceunavailable",
    "slowdown",
)
_HF_FATAL_ERRORS: tuple[type[BaseException], ...] = ()
_HF_TRANSIENT_ERRORS = (
    OSError,
    httpx.TransportError,
    httpcore.ProtocolError,
    requests.exceptions.RequestException,
    urllib3.exceptions.HTTPError,
)
try:
    from huggingface_hub.errors import (
        EntryNotFoundError,
        GatedRepoError,
        HfHubHTTPError,
        LocalEntryNotFoundError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    _HF_FATAL_ERRORS = (RepositoryNotFoundError, RevisionNotFoundError, GatedRepoError)
    _HF_TRANSIENT_ERRORS += (HfHubHTTPError, EntryNotFoundError, LocalEntryNotFoundError)
except ImportError:
    pass
_HF_TRANSIENT_MESSAGE_FRAGMENTS = (
    "incompleteread",
    "incomplete read",
    "eof",
    "connection reset",
    "connection aborted",
    "connection broken",
    "remote end closed",
    "broken pipe",
    "server disconnected",
    "disconnected without sending",
    "peer closed connection",
)

_S3_FILESYSTEM: AbstractFileSystem | None = None


def create_s3_filesystem(**storage_options: Any) -> AbstractFileSystem:
    """Create an uncached S3 filesystem through Rigging's guarded factory."""
    return guarded_filesystem("s3", **storage_options)


def get_s3_filesystem() -> AbstractFileSystem:
    """Return the shared guarded S3 filesystem with bounded request attempts."""
    global _S3_FILESYSTEM
    if _S3_FILESYSTEM is None:
        config_kwargs = s3_python_config_kwargs()
        config_kwargs["retries"] = {"total_max_attempts": _S3_REQUEST_TOTAL_ATTEMPTS, "mode": "standard"}
        config_kwargs["s3"] = {
            "addressing_style": os.environ.get(_S3_ADDRESSING_STYLE_ENV, "virtual"),
        }
        filesystem = create_s3_filesystem(config_kwargs=config_kwargs)
        filesystem.retries = _S3_FILESYSTEM_RETRIES
        _S3_FILESYSTEM = filesystem
    return _S3_FILESYSTEM


def filesystem_and_path(uri: str) -> tuple[AbstractFileSystem, str]:
    """Resolve a URI through Rigging's guarded fsspec factory."""
    if uri.startswith(("s3://", "s3a://")):
        filesystem = get_s3_filesystem()
        return filesystem, filesystem._strip_protocol(uri)
    return guarded_url_to_fs(uri)


def _s3_expiry_time() -> datetime | None:
    try:
        import botocore.session

        credentials = botocore.session.get_session().get_credentials()
        if credentials is None:
            return None
        return getattr(credentials, "expiry_time", None) or getattr(credentials, "_expiry_time", None)
    except Exception:
        return None


def _refresh_s3_credentials(filesystem: AbstractFileSystem) -> None:
    if not hasattr(filesystem, "connect"):
        return
    try:
        filesystem.connect(refresh=True)
    except Exception:
        logger.opt(exception=True).warning("Failed to refresh S3 credentials before retry")


def refresh_s3_credentials_if_expiring(filesystem: AbstractFileSystem) -> None:
    """Refresh temporary S3 credentials when they expire within five minutes."""
    expiry = _s3_expiry_time()
    if expiry is not None and datetime.now(timezone.utc) >= expiry - timedelta(minutes=5):
        _refresh_s3_credentials(filesystem)


def _classify_s3_error(error: Exception) -> tuple[bool, bool]:
    """Return whether to retry and whether a retry should refresh credentials."""
    response = getattr(error, "response", None)
    if isinstance(response, dict):
        code = str(response.get("Error", {}).get("Code", ""))
        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        auth_error = code in _RETRYABLE_AUTH_CODES or status == 403
        return auth_error or is_transient_s3_error(error), auth_error
    if is_transient_s3_error(error):
        return True, False
    if isinstance(error, OSError):
        message = str(error).lower()
        auth_error = "accessdenied" in message or "forbidden" in message
        translated = error.errno == errno.EIO and any(
            marker in message for marker in _RETRYABLE_TRANSLATED_ERROR_MARKERS
        )
        return translated, auth_error and translated
    return False, False


def call_with_s3_retry(
    filesystem: AbstractFileSystem,
    call: Callable[..., T],
    *args: Any,
    max_attempts: int = _S3_TRANSFER_MAX_ATTEMPTS,
    **kwargs: Any,
) -> T:
    """Call one S3 operation using the shared timeout, retry, and credential policy."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")

    def invoke() -> T:
        return call(*args, **kwargs)

    def retryable(error: Exception) -> bool:
        should_retry, _refresh = _classify_s3_error(error)
        return should_retry

    def on_retry(error: Exception, _attempt: int) -> None:
        _retryable, refresh = _classify_s3_error(error)
        if refresh:
            _refresh_s3_credentials(filesystem)

    return retry_with_backoff(
        invoke,
        retryable=retryable,
        max_attempts=max_attempts,
        backoff=ExponentialBackoff(initial=1.0, maximum=16.0, factor=2.0, jitter=0.2),
        on_retry=on_retry,
        operation="S3 operation",
    )


def call_with_filesystem_retry(
    filesystem: AbstractFileSystem,
    call: Callable[..., T],
    *args: Any,
    max_attempts: int = _S3_TRANSFER_MAX_ATTEMPTS,
    **kwargs: Any,
) -> T:
    """Apply the shared retry policy when ``filesystem`` is backed by S3."""
    protocol = getattr(filesystem, "protocol", ())
    protocols = protocol if isinstance(protocol, tuple) else (protocol,)
    if any(protocol in {"s3", "s3a"} for protocol in protocols):
        return call_with_s3_retry(filesystem, call, *args, max_attempts=max_attempts, **kwargs)
    return call(*args, **kwargs)


def is_transient_hugging_face_error(error: BaseException) -> bool:
    """Return whether a Hugging Face failure is safe to retry."""
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, _HF_FATAL_ERRORS):
            return False
        if isinstance(current, _HF_TRANSIENT_ERRORS):
            return True
        message = str(current).lower()
        if any(fragment in message for fragment in _HF_TRANSIENT_MESSAGE_FRAGMENTS):
            return True
        if "safetensors" in message and any(
            fragment in message
            for fragment in ("no ", "not find", "cannot find", "couldn't find", "could not find", "does not")
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def call_with_hugging_face_retry(
    call: Callable[[], T],
    *,
    operation: str,
    max_retries: int = DEFAULT_HF_MAX_RETRIES,
    backoff_base: float = DEFAULT_HF_BACKOFF_BASE_SECONDS,
    backoff_cap: float = DEFAULT_HF_BACKOFF_CAP_SECONDS,
) -> T:
    """Call a Hub operation with the shared transient-failure policy."""
    return retry_with_backoff(
        call,
        retryable=is_transient_hugging_face_error,
        max_attempts=max_retries + 1,
        backoff=ExponentialBackoff(initial=backoff_base, maximum=backoff_cap, factor=2.0, jitter=0.0),
        operation=operation,
    )


def load_hugging_face_with_retry(
    call: Callable[[], T],
    *,
    model_id: str,
    max_retries: int = DEFAULT_HF_MAX_RETRIES,
    backoff_base: float = DEFAULT_HF_BACKOFF_BASE_SECONDS,
    backoff_cap: float = DEFAULT_HF_BACKOFF_CAP_SECONDS,
) -> T:
    """Load one model through the shared Hugging Face retry policy."""
    rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "?"))
    return call_with_hugging_face_retry(
        call,
        operation=f"load Hugging Face model {model_id} on rank {rank}",
        max_retries=max_retries,
        backoff_base=backoff_base,
        backoff_cap=backoff_cap,
    )


@runtime_checkable
class MultipartS3FileSystem(Protocol):
    protocol: str | tuple[str, ...]

    def split_path(self, path: str) -> tuple[str, str, str | None]: ...

    def call_s3(self, method: str, *args: Any, **kwargs: Any) -> Any: ...


class S3MultipartWriteStream:
    """Bounded file-like S3 writer with concurrent, individually retried parts."""

    def __init__(
        self,
        filesystem: MultipartS3FileSystem,
        path: str,
        *,
        part_bytes: int = DEFAULT_S3_MULTIPART_PART_BYTES,
        concurrency: int = DEFAULT_S3_MULTIPART_CONCURRENCY,
    ) -> None:
        if part_bytes < MINIMUM_S3_MULTIPART_PART_BYTES:
            raise ValueError(f"part_bytes must be at least {MINIMUM_S3_MULTIPART_PART_BYTES}")
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
        self._pending: deque[tuple[int, Future[dict[str, int | str]]]] = deque()
        self._executor = ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="marinskyrl-s3")

    def writable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def flush(self) -> None:
        if self.closed:
            raise ValueError("flush of closed remote stream")

    def write(self, payload: Any) -> int:
        if self.closed:
            raise ValueError("write to closed remote stream")
        view = memoryview(payload).cast("B")
        payload_bytes = len(view)
        try:
            while view:
                chunk_bytes = min(self.part_bytes - len(self._buffer), len(view))
                self._buffer.extend(view[:chunk_bytes])
                view = view[chunk_bytes:]
                if len(self._buffer) == self.part_bytes:
                    self._submit_part(bytes(self._buffer))
                    self._buffer.clear()
        except BaseException as error:
            try:
                self.discard()
            except Exception as cleanup_error:
                error.add_note(f"Failed to abort multipart upload for {self.path}: {cleanup_error}")
            raise
        self._position += payload_bytes
        return payload_bytes

    def close(self) -> None:
        if self.closed:
            return
        if self._discarded:
            self.closed = True
            return
        try:
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
            self.closed = True
        except BaseException as error:
            try:
                self.discard()
            except Exception as cleanup_error:
                error.add_note(f"Failed to abort multipart upload for {self.path}: {cleanup_error}")
            raise

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
            response = call_with_s3_retry(
                self.filesystem,
                self.filesystem.call_s3,
                "create_multipart_upload",
                max_attempts=1,
                Bucket=self.bucket,
                Key=self.key,
            )
            self._upload_id = str(response["UploadId"])

        part_number = self._next_part_number
        self._next_part_number += 1
        future = self._executor.submit(self._upload_part, part_number, payload)
        self._pending.append((part_number, future))
        if len(self._pending) >= self.concurrency:
            self._finish_oldest_part()

    def _upload_part(self, part_number: int, payload: bytes) -> dict[str, int | str]:
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
        return {"PartNumber": part_number, "ETag": str(response["ETag"])}

    def _finish_oldest_part(self) -> None:
        _part_number, future = self._pending.popleft()
        self._completed_parts.append(future.result())


@contextmanager
def open_output_stream(
    filesystem: AbstractFileSystem,
    path: str,
    *,
    part_bytes: int = DEFAULT_S3_MULTIPART_PART_BYTES,
    concurrency: int = DEFAULT_S3_MULTIPART_CONCURRENCY,
) -> Generator[Any, None, None]:
    """Open a write stream, using bounded multipart transfer for S3."""
    protocol = getattr(filesystem, "protocol", ())
    protocols = protocol if isinstance(protocol, tuple) else (protocol,)
    if "s3" in protocols or "s3a" in protocols:
        if not isinstance(filesystem, MultipartS3FileSystem):
            raise TypeError("S3 filesystem does not provide multipart operations")
        stream = S3MultipartWriteStream(filesystem, path, part_bytes=part_bytes, concurrency=concurrency)
    else:
        stream = filesystem.open(path, "wb")
    try:
        yield stream
        stream.close()
    except BaseException as error:
        if hasattr(stream, "discard"):
            try:
                stream.discard()
            except Exception as cleanup_error:
                error.add_note(f"Failed to discard remote write for {path}: {cleanup_error}")
        else:
            stream.close()
        error.add_note(f"Object write failed for {path}")
        raise


def abort_multipart_uploads(path: str) -> int:
    """Abort incomplete uploads below a canonical S3 checkpoint prefix."""
    checkpoint_path = StoragePath(path)
    if checkpoint_path.scheme != "s3" or str(checkpoint_path) != path or not checkpoint_path.key:
        raise ValueError(f"Expected a canonical S3 checkpoint path, got: {path}")

    filesystem = get_s3_filesystem()
    refresh_s3_credentials_if_expiring(filesystem)
    response = call_with_s3_retry(
        filesystem,
        filesystem.call_s3,
        "list_multipart_uploads",
        max_attempts=1,
        Bucket=checkpoint_path.bucket,
        Prefix=f"{checkpoint_path.key}/",
    )
    uploads = response.get("Uploads", [])
    for upload in uploads:
        call_with_s3_retry(
            filesystem,
            filesystem.call_s3,
            "abort_multipart_upload",
            max_attempts=1,
            Bucket=checkpoint_path.bucket,
            Key=upload["Key"],
            UploadId=upload["UploadId"],
        )
    if uploads:
        logger.warning("Aborted {} stale multipart uploads below {}", len(uploads), path)
    return len(uploads)
