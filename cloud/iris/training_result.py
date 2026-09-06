"""Validate trainer completion without loading the training runtime."""

from __future__ import annotations

import hashlib
import json
import posixpath
from typing import Any

from hydra.core.override_parser.overrides_parser import OverridesParser

from cloud.iris.artifacts import fs_and_path, terminal_checkpoint_step
from cloud.iris.protocol import SkyRLLaunchRequest, SkyRLTrainingResult, training_receipt_uri, training_request_fingerprint
from marinskyrl.training_completion import CompletionMode, NativeCheckpoint, TrainingReceipt


def read_json(uri: str) -> dict[str, Any]:
    filesystem, path = fs_and_path(uri)
    with filesystem.open(path, "r") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object at {uri}")
    return value


def validate_native_checkpoint(checkpoint: NativeCheckpoint) -> None:
    """Check the exact receipt inventory, including the small trainer-state digest."""
    filesystem, root = fs_and_path(checkpoint.checkpoint_path)
    names = {entry.path for entry in checkpoint.files}
    if not {"trainer_state.pt", "data.pt"}.issubset(names) or not any(name.startswith("policy/") for name in names):
        raise ValueError("Checkpoint receipt has no complete training-state/policy inventory")
    for entry in checkpoint.files:
        if entry.path.startswith("/") or ".." in entry.path.split("/"):
            raise ValueError(f"Invalid checkpoint inventory path: {entry.path}")
        path = posixpath.join(root, entry.path)
        if not filesystem.isfile(path) or int(filesystem.info(path)["size"]) != entry.size:
            raise ValueError(f"Checkpoint file is missing or changed: {checkpoint.checkpoint_path}/{entry.path}")
    with filesystem.open(posixpath.join(root, "trainer_state.pt"), "rb") as source:
        digest = hashlib.sha256(source.read()).hexdigest()
    if digest != checkpoint.trainer_state_sha256:
        raise ValueError("Checkpoint trainer state differs from its completion receipt")


def read_training_result(request: SkyRLLaunchRequest, *, check_latest: bool = True, check_files: bool = True) -> SkyRLTrainingResult:
    receipt_uri = training_receipt_uri(request)
    receipt = TrainingReceipt.from_dict(read_json(receipt_uri))
    expected = (request.run_id, request.attempt_id, training_request_fingerprint(request), request.completion_mode)
    actual = (receipt.run_id, receipt.attempt_id, receipt.request_fingerprint, receipt.completion_mode)
    if actual != expected:
        raise ValueError("Training receipt does not match this request and attempt")
    filesystem, path = fs_and_path(request.output.resolved_config_uri)
    if not filesystem.isfile(path):
        raise ValueError(f"Training did not persist resolved config: {request.output.resolved_config_uri}")
    resolved = read_json(request.output.resolved_config_uri)
    if not isinstance(resolved.get("entrypoint"), str) or not isinstance(resolved.get("hydra_args"), list):
        raise ValueError("Resolved training config must contain entrypoint and hydra_args")
    recorded_completion = {}
    for override in OverridesParser.create().parse_overrides(resolved["hydra_args"]):
        if override.key_or_group.startswith("trainer.completion."):
            recorded_completion[override.key_or_group.removeprefix("trainer.completion.")] = override.value()
    expected_completion = {
        "mode": request.completion_mode.value,
        "run_id": request.run_id,
        "attempt_id": request.attempt_id,
        "request_fingerprint": training_request_fingerprint(request),
        "receipt_uri": receipt_uri,
    }
    if recorded_completion != expected_completion:
        raise ValueError("Resolved training config does not match this request and completion receipt")
    if request.completion_mode is CompletionMode.CHECKPOINT:
        checkpoint = receipt.checkpoint
        expected_path = posixpath.join(request.output.checkpoint_root.rstrip("/"), f"global_step_{receipt.global_step}")
        if checkpoint is None or checkpoint.global_step != receipt.global_step or checkpoint.checkpoint_path != expected_path:
            raise ValueError("Training receipt has no checkpoint at the completed optimizer step")
        if check_files:
            validate_native_checkpoint(checkpoint)
        if check_latest and terminal_checkpoint_step(request.output.checkpoint_root) != receipt.global_step:
            raise ValueError("Final checkpoint marker differs from the completed optimizer step")
    elif receipt.checkpoint is not None:
        raise ValueError("Metrics-only completion cannot contain a checkpoint")
    return SkyRLTrainingResult(
        global_step=receipt.global_step,
        receipt_uri=receipt_uri,
        resolved_config_uri=request.output.resolved_config_uri,
        checkpoint=receipt.checkpoint,
    )
