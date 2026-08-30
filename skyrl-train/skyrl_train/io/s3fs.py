from datetime import datetime, timezone, timedelta
import errno
import os
import random
import time

import fsspec
from fsspec.exceptions import FSTimeoutError
from loguru import logger

# Optional AWS deps (present when s3fs is installed)
try:
    import botocore.session as _botocore_session
    from botocore.exceptions import ClientError, ConnectionError as BotocoreConnectionError, HTTPClientError

    _HAS_BOTOCORE = True
    _TRANSIENT_S3_ERRORS = (FSTimeoutError, TimeoutError, BotocoreConnectionError, HTTPClientError)
except Exception:
    _HAS_BOTOCORE = False
    _TRANSIENT_S3_ERRORS = (FSTimeoutError, TimeoutError)

    class ClientError(Exception):  # fallback type
        pass


_S3_FS = None  # type: ignore
_S3_CONNECT_TIMEOUT_SECONDS = 60
_S3_READ_TIMEOUT_SECONDS = 300
_S3_SDK_MAX_ATTEMPTS = 10
_S3_TRANSFER_MAX_ATTEMPTS = 5
_S3_RETRY_BASE_SECONDS = 1.0
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


def get_s3_fs():
    """Return a cached S3 filesystem instance, creating it once."""
    global _S3_FS
    if _S3_FS is None:
        addressing_style = os.environ.get(_S3_ADDRESSING_STYLE_ENV, "virtual")
        _S3_FS = fsspec.filesystem(
            "s3",
            config_kwargs={
                "connect_timeout": _S3_CONNECT_TIMEOUT_SECONDS,
                "read_timeout": _S3_READ_TIMEOUT_SECONDS,
                "retries": {"max_attempts": _S3_SDK_MAX_ATTEMPTS, "mode": "adaptive"},
                "s3": {"addressing_style": addressing_style},
            },
        )
    return _S3_FS


def s3_expiry_time():
    """Return botocore credential expiry (datetime in UTC) or None."""
    if not _HAS_BOTOCORE:
        return None
    try:
        sess = _botocore_session.get_session()
        creds = sess.get_credentials()
        if not creds:
            return None
        return getattr(creds, "expiry_time", None) or getattr(creds, "_expiry_time", None)
    except Exception:
        return None


def s3_refresh_if_expiring(fs) -> None:
    """
    Simple refresh:
    - If expiry exists and is within 300s (or past), refresh with fs.connect(refresh=True).
    - Otherwise, do nothing.
    """
    exp = s3_expiry_time()
    if not exp:
        return
    now = datetime.now(timezone.utc)
    if now >= exp - timedelta(seconds=300):
        _refresh_s3_credentials(fs)


def _retryable_s3_error(error: Exception) -> tuple[bool, bool]:
    """Return whether an error is retryable and whether credentials should refresh."""
    if isinstance(error, ClientError):
        response = getattr(error, "response", {})
        code = str(response.get("Error", {}).get("Code", ""))
        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        auth_error = code in _RETRYABLE_AUTH_CODES or status == 403
        return auth_error or (isinstance(status, int) and status >= 500), auth_error
    if isinstance(error, _TRANSIENT_S3_ERRORS):
        return True, False
    if isinstance(error, OSError):
        message = str(error).lower()
        auth_error = "accessdenied" in message or "forbidden" in message
        translated_s3_error = error.errno == errno.EIO and any(
            marker in message for marker in _RETRYABLE_TRANSLATED_ERROR_MARKERS
        )
        return translated_s3_error, auth_error and translated_s3_error
    return False, False


def _refresh_s3_credentials(fs) -> None:
    if not hasattr(fs, "connect"):
        return
    try:
        fs.connect(refresh=True)
    except Exception:
        logger.opt(exception=True).warning("Failed to refresh S3 credentials before retry")


def call_with_s3_retry(fs, fn, *args, **kwargs):
    """Call an S3 operation with bounded retries for credentials and transient transport failures."""
    for attempt in range(1, _S3_TRANSFER_MAX_ATTEMPTS + 1):
        try:
            return fn(*args, **kwargs)
        except (ClientError, *_TRANSIENT_S3_ERRORS, OSError) as error:
            retryable, refresh_credentials = _retryable_s3_error(error)
            if not retryable:
                raise
            if refresh_credentials:
                _refresh_s3_credentials(fs)
            retry_error = error

        if attempt == _S3_TRANSFER_MAX_ATTEMPTS:
            raise retry_error
        delay = _S3_RETRY_BASE_SECONDS * (2 ** (attempt - 1)) * random.uniform(0.8, 1.2)
        logger.warning(
            "S3 operation failed with {}; retrying attempt {}/{} in {:.1f}s",
            type(retry_error).__name__,
            attempt + 1,
            _S3_TRANSFER_MAX_ATTEMPTS,
            delay,
        )
        time.sleep(delay)

    raise AssertionError("unreachable")
