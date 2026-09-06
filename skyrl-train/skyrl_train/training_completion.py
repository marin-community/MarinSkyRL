"""Publish attempt-bound receipts after the trainer and tracker finish successfully."""

import hashlib
import io as bytes_io
import json
from pathlib import PurePosixPath
from typing import Protocol

import torch
from omegaconf import DictConfig

from marinskyrl.checkpoint_paths import LATEST_CHECKPOINT_FILE, POLICY_CHECKPOINT_SUBDIRECTORY
from marinskyrl.resource_locator import join_resource_path
from marinskyrl.training_completion import (
    CheckpointFile,
    CompletionMode,
    NativeCheckpoint,
    TrainingReceipt,
    completion_mode,
    validate_completion_config,
)
from skyrl_train.hf_export_schema import TRAINER_STATE_FILENAME
from skyrl_train.io import io


class CompletedTrainer(Protocol):
    global_step: int


def native_checkpoint(cfg: DictConfig, global_step: int) -> NativeCheckpoint:
    """Verify trainer-owned metadata and inventory without reading policy weights."""
    root = cfg.trainer.ckpt_path
    checkpoint_path = join_resource_path(root, f"global_step_{global_step}")
    latest = io.read_bytes(join_resource_path(root, LATEST_CHECKPOINT_FILE)).decode().strip()
    if latest != str(global_step):
        raise ValueError(f"Checkpoint marker {latest!r} does not match completed step {global_step}")
    state_bytes = io.read_bytes(join_resource_path(checkpoint_path, TRAINER_STATE_FILENAME))
    state = torch.load(bytes_io.BytesIO(state_bytes), map_location="cpu", weights_only=False)
    if state.get("global_step") != global_step:
        raise ValueError("Saved trainer state does not match completed optimizer step")
    prefix = checkpoint_path.split("://", 1)[-1].rstrip("/") + "/"
    inventory = io.find_files(checkpoint_path)
    files = tuple(CheckpointFile(path=path.removeprefix(prefix), size=size) for path, size in sorted(inventory.items()))
    by_path = {file.path: file.size for file in files}
    for required in ("data.pt", TRAINER_STATE_FILENAME):
        if by_path.get(required, 0) <= 0:
            raise ValueError(f"Native checkpoint is missing nonempty {required}")
    policy_prefix = POLICY_CHECKPOINT_SUBDIRECTORY + "/"
    if not any(
        file.path.startswith(policy_prefix)
        and PurePosixPath(file.path).suffix in {".pt", ".bin", ".distcp", ".safetensors"}
        and file.size > 0
        for file in files
    ):
        raise ValueError("Native checkpoint is missing policy payload")
    return NativeCheckpoint(
        checkpoint_path=checkpoint_path,
        global_step=global_step,
        trainer_state_sha256=hashlib.sha256(state_bytes).hexdigest(),
        files=files,
    )


def write_training_receipt(cfg: DictConfig, trainer: CompletedTrainer) -> TrainingReceipt | None:
    """Write the completion receipt last, after all lifecycle cleanup succeeded."""
    validate_completion_config(cfg)
    mode = completion_mode(cfg)
    if mode is None:
        return None
    completion = cfg.trainer.completion
    step = trainer.global_step
    if type(step) is not int or step < 0:
        raise ValueError("Completed optimizer step must be a nonnegative integer")
    checkpoint = native_checkpoint(cfg, step) if mode == CompletionMode.CHECKPOINT else None
    receipt = TrainingReceipt(
        run_id=completion.run_id,
        attempt_id=completion.attempt_id,
        request_fingerprint=completion.request_fingerprint,
        completion_mode=mode,
        global_step=step,
        checkpoint=checkpoint,
    )
    payload = receipt.to_dict()
    TrainingReceipt.from_dict(payload)
    io.write_bytes_atomic(completion.receipt_uri, json.dumps(payload, sort_keys=True).encode())
    return receipt
