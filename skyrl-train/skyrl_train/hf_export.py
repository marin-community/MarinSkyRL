"""Durable requests for exporting immutable training checkpoints to Hugging Face format."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

from loguru import logger

from skyrl_train.utils.io import io
from skyrl_train.utils.trainer_utils import extract_step_from_path, list_checkpoint_dirs
from skyrl_train.hf_export_schema import (
    DEFAULT_HF_HUB_REVISION,
    DEFAULT_HF_UPLOAD_MODE,
    HF_EXPORT_REQUEST_FILENAME,
    HF_EXPORT_REQUEST_SCHEMA_VERSION,
    HFExportRequest,
    HFExportStatus,
    HFUploadMode,
)


def hf_export_request_path(checkpoint_path: str) -> str:
    return os.path.join(checkpoint_path, HF_EXPORT_REQUEST_FILENAME)


def new_hf_export_request(
    *,
    step: int,
    checkpoint_base_path: str,
    checkpoint_path: str,
    export_path: str,
    model_path: str,
    num_nodes: int,
    gpus_per_node: int,
    hf_hub_repo_id: str | None = None,
    hf_hub_private: bool = False,
    hf_hub_revision: str = DEFAULT_HF_HUB_REVISION,
    hf_upload_mode: HFUploadMode = DEFAULT_HF_UPLOAD_MODE,
) -> HFExportRequest:
    return HFExportRequest(
        schema_version=HF_EXPORT_REQUEST_SCHEMA_VERSION,
        status=HFExportStatus.PENDING,
        step=step,
        checkpoint_base_path=checkpoint_base_path,
        checkpoint_path=checkpoint_path,
        export_path=export_path,
        model_path=model_path,
        num_nodes=num_nodes,
        gpus_per_node=gpus_per_node,
        hf_hub_repo_id=hf_hub_repo_id,
        hf_hub_private=hf_hub_private,
        hf_hub_revision=hf_hub_revision,
        hf_upload_mode=hf_upload_mode,
    )


def write_hf_export_request(request: HFExportRequest) -> str:
    """Persist one request and return its path; local writes replace atomically."""
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
    payload["hf_upload_mode"] = HFUploadMode(payload["hf_upload_mode"])
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
        except (OSError, KeyError, TypeError, ValueError) as error:
            step = extract_step_from_path(checkpoint_path)
            if step >= 0:
                pending.add(step)
            logger.warning(f"Protecting checkpoint with unreadable HF export request at {checkpoint_path}: {error}")
            continue
        if request is not None and request.status is not HFExportStatus.COMPLETE:
            pending.add(request.step)
    return pending
