"""Atomically reserve a measurement boundary while permitting earlier startup retries."""

import hashlib
import json
import os
from typing import Protocol, cast
from urllib.parse import urlsplit

from cloud.iris.artifacts import fs_and_path


class ConditionalObjectStore(Protocol):
    def call_s3(self, method: str, **kwargs: object) -> object: ...

    def cat_file(self, path: str) -> bytes: ...


def claim_measurement(uri: str) -> dict[str, str]:
    """Create one run-scoped marker before any initial evaluation or training.

    S3 must implement conditional PutObject. A pre-existing marker always fails,
    including one from this attempt: this operation is deliberately not resumable.
    Permission, transport and unsupported conditional-write errors also fail closed.
    """
    parsed = urlsplit(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/") or parsed.query or parsed.fragment:
        raise ValueError("Measurement guard requires an explicit S3 object URI")
    task_id, attempt_uid = os.environ.get("IRIS_TASK_ID"), os.environ.get("IRIS_ATTEMPT_UID")
    if not task_id or not attempt_uid:
        raise ValueError("Measurement guard requires native task and attempt identities")
    if len(task_id) > 2048 or len(attempt_uid) > 128:
        raise ValueError("Measurement guard native identity exceeds its bound")
    payload = {
        "schema_version": 1,
        "boundary": "before_initial_evaluation_or_training",
        "task_id": task_id,
        "attempt_uid": attempt_uid,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    filesystem, path = fs_and_path(uri)
    if not callable(getattr(filesystem, "call_s3", None)):
        raise ValueError("Measurement guard requires conditional S3 PutObject support")
    store = cast(ConditionalObjectStore, filesystem)
    store.call_s3(
        "put_object",
        Bucket=parsed.netloc,
        Key=parsed.path.lstrip("/"),
        Body=encoded,
        IfNoneMatch="*",
        ContentType="application/json",
    )
    if store.cat_file(path) != encoded:
        raise RuntimeError("Measurement guard readback differs from the claimed native identity")
    return {"uri": uri, "sha256": hashlib.sha256(encoded).hexdigest(), "attempt_uid": attempt_uid}
