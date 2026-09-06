"""Torch-free completion receipts for Hugging Face policy exports."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath
from typing import Any

from fsspec.spec import AbstractFileSystem

from cloud.iris.artifacts import fs_and_path, relative_object_key
from marinskyrl.hf_model import validate_portable_hf_model_files
from marinskyrl.resource_locator import join_resource_path

EXPORT_RECEIPT_SCHEMA_VERSION = 1
HF_WEIGHT_FILENAME = "model.safetensors"
HF_WEIGHT_INDEX_FILENAME = "model.safetensors.index.json"


@dataclass(frozen=True)
class ExportReceipt:
    """Attempt metadata proving one exact export finished successfully."""

    request_fingerprint: str
    attempt_id: str
    export_path: str
    global_step: int
    schema_version: int = EXPORT_RECEIPT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("request_fingerprint", "attempt_id", "export_path"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Export receipt {name} must be a nonempty string")
        if type(self.global_step) is not int or self.global_step < 0:
            raise ValueError("Export receipt global_step must be a nonnegative integer")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ExportReceipt:
        if value.get("schema_version") != EXPORT_RECEIPT_SCHEMA_VERSION:
            raise ValueError("Unsupported export receipt schema_version")
        return cls(
            request_fingerprint=value["request_fingerprint"],
            attempt_id=value["attempt_id"],
            export_path=value["export_path"],
            global_step=value["global_step"],
        )


def _export_files(export_path: str) -> tuple[AbstractFileSystem, str, dict[str, int]]:
    filesystem, root = fs_and_path(export_path)
    try:
        inventory = filesystem.find(root, detail=True, withdirs=False)
    except FileNotFoundError as error:
        raise ValueError(f"Model export not found: {export_path}") from error
    files = {relative_object_key(root, path): int(details["size"]) for path, details in inventory.items()}
    validate_portable_hf_model_files(set(files), export_path)
    return filesystem, root, files


def validate_hf_export(export_path: str) -> None:
    """Validate the complete nonempty safetensors payload at ``export_path``."""
    filesystem, root, files = _export_files(export_path)
    if files.get("config.json", 0) <= 0:
        raise ValueError(f"Model export has an empty config.json: {export_path}")
    tokenizer_files = [name for name in files if name.startswith("tokenizer") or name.endswith(".model")]
    if not any(files[name] > 0 for name in tokenizer_files):
        raise ValueError(f"Model export has no nonempty tokenizer files: {export_path}")

    if HF_WEIGHT_INDEX_FILENAME not in files:
        if files.get(HF_WEIGHT_FILENAME, 0) > 0:
            return
        raise ValueError(f"Model export has no nonempty safetensors weights: {export_path}")
    if files[HF_WEIGHT_INDEX_FILENAME] <= 0:
        raise ValueError(f"Model export has an empty safetensors index: {export_path}")

    index_path = join_resource_path(root, HF_WEIGHT_INDEX_FILENAME)
    try:
        with filesystem.open(index_path, "r") as source:
            index = json.load(source)
    except (OSError, ValueError) as error:
        raise ValueError(f"Model export has an unreadable safetensors index: {export_path}") from error
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"Model export has an invalid safetensors weight map: {export_path}")
    shards = set(weight_map.values())
    if any(not isinstance(shard, str) or PurePosixPath(shard).name != shard for shard in shards):
        raise ValueError(f"Model export index contains invalid shard paths: {export_path}")
    missing = sorted(shard for shard in shards if files.get(shard, 0) <= 0)
    if missing:
        raise ValueError(f"Model export is missing {len(missing)} nonempty safetensors shard(s): {missing[:5]}")


def verify_export_receipt(
    receipt_uri: str,
    request_fingerprint: str,
    export_path: str,
    global_step: int,
) -> ExportReceipt:
    """Validate an attempt receipt and the exact HF artifact it identifies."""
    filesystem, path = fs_and_path(receipt_uri)
    try:
        with filesystem.open(path, "r") as source:
            value = json.load(source)
    except FileNotFoundError as error:
        raise ValueError(f"Export completion receipt not found: {receipt_uri}") from error
    if not isinstance(value, dict):
        raise TypeError(f"Export completion receipt must be a JSON object: {receipt_uri}")
    receipt = ExportReceipt.from_dict(value)
    expected = (request_fingerprint, export_path, global_step)
    actual = (receipt.request_fingerprint, receipt.export_path, receipt.global_step)
    if actual != expected:
        raise ValueError("Export completion receipt does not match this request")
    if not receipt.attempt_id:
        raise ValueError("Export completion receipt has an empty attempt_id")
    validate_hf_export(export_path)
    return receipt
