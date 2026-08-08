"""Durable requests for exporting immutable training checkpoints to Hugging Face format."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from pathlib import Path

from loguru import logger

from skyrl_train.utils.io import io
from skyrl_train.utils.trainer_utils import extract_step_from_path, list_checkpoint_dirs

HF_EXPORT_REQUEST_FILENAME = "hf_export_request.json"
HF_EXPORT_REQUEST_SCHEMA_VERSION = 1


class HFExportStatus(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"


@dataclass(frozen=True)
class HFExportRequest:
    schema_version: int
    status: HFExportStatus
    step: int
    checkpoint_path: str
    export_path: str
    model_path: str
    strategy: str
    num_nodes: int
    gpus_per_node: int
    hf_hub_repo_id: str | None = None
    hf_hub_private: bool = False
    hf_hub_revision: str = "main"
    hf_upload_mode: str = "latest"
    attempts: int = 0
    timeout_seconds: int | None = None
    last_exit_code: int | None = None

    def with_status(
        self,
        status: HFExportStatus,
        *,
        timeout_seconds: int | None = None,
        last_exit_code: int | None = None,
        increment_attempts: bool = False,
    ) -> "HFExportRequest":
        return replace(
            self,
            status=status,
            timeout_seconds=timeout_seconds if timeout_seconds is not None else self.timeout_seconds,
            last_exit_code=last_exit_code,
            attempts=self.attempts + int(increment_attempts),
        )


def hf_export_request_path(checkpoint_path: str) -> str:
    return os.path.join(checkpoint_path, HF_EXPORT_REQUEST_FILENAME)


def new_hf_export_request(
    *,
    step: int,
    checkpoint_path: str,
    export_path: str,
    model_path: str,
    strategy: str,
    num_nodes: int,
    gpus_per_node: int,
    hf_hub_repo_id: str | None = None,
    hf_hub_private: bool = False,
    hf_hub_revision: str = "main",
    hf_upload_mode: str = "latest",
) -> HFExportRequest:
    return HFExportRequest(
        schema_version=HF_EXPORT_REQUEST_SCHEMA_VERSION,
        status=HFExportStatus.PENDING,
        step=step,
        checkpoint_path=checkpoint_path,
        export_path=export_path,
        model_path=model_path,
        strategy=strategy,
        num_nodes=num_nodes,
        gpus_per_node=gpus_per_node,
        hf_hub_repo_id=hf_hub_repo_id,
        hf_hub_private=hf_hub_private,
        hf_hub_revision=hf_hub_revision,
        hf_upload_mode=hf_upload_mode,
    )


def write_hf_export_request(request: HFExportRequest) -> str:
    """Write one request atomically and return its path."""
    path = hf_export_request_path(request.checkpoint_path)
    payload = asdict(request)
    payload["status"] = request.status.value

    if io.is_cloud_path(path):
        with io.open_file(path, "w") as output:
            json.dump(payload, output, sort_keys=True)
        return path

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=destination.parent, delete=False) as output:
            temporary_path = output.name
            json.dump(payload, output, sort_keys=True)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.remove(temporary_path)
    return path


def read_hf_export_request(checkpoint_path: str) -> HFExportRequest | None:
    path = hf_export_request_path(checkpoint_path)
    if not io.exists(path):
        return None
    with io.open_file(path, "r") as source:
        payload = json.load(source)
    payload["status"] = HFExportStatus(payload["status"])
    request = HFExportRequest(**payload)
    if request.schema_version != HF_EXPORT_REQUEST_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported HF export request schema {request.schema_version} at {path}; "
            f"expected {HF_EXPORT_REQUEST_SCHEMA_VERSION}"
        )
    return request


def pending_hf_export_steps(checkpoint_base_path: str) -> set[int]:
    """Return checkpoint steps whose export request is absent from completion."""
    pending: set[int] = set()
    for directory in list_checkpoint_dirs(checkpoint_base_path):
        checkpoint_path = os.path.join(checkpoint_base_path, directory)
        try:
            request = read_hf_export_request(checkpoint_path)
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
            step = extract_step_from_path(checkpoint_path)
            if step >= 0:
                pending.add(step)
            logger.warning(f"Protecting checkpoint with unreadable HF export request at {checkpoint_path}: {error}")
            continue
        if request is not None and request.status is not HFExportStatus.COMPLETE:
            pending.add(request.step)
    return pending
