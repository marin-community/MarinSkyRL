"""Torch-free training completion wire types and saving policy validation."""

import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Any


class CompletionMode(StrEnum):
    METRICS = "metrics"
    CHECKPOINT = "checkpoint"


@dataclass(frozen=True)
class CheckpointFile:
    path: str
    size: int


@dataclass(frozen=True)
class NativeCheckpoint:
    checkpoint_path: str
    global_step: int
    trainer_state_sha256: str
    files: tuple[CheckpointFile, ...]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["files"] = [asdict(file) for file in self.files]
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "NativeCheckpoint":
        if not isinstance(value, Mapping):
            raise TypeError("Native checkpoint must be an object")
        _validate_step(value.get("global_step"))
        _validate_digest(value.get("trainer_state_sha256"))
        if not isinstance(value.get("checkpoint_path"), str) or not value["checkpoint_path"].strip():
            raise ValueError("Native checkpoint path must be nonempty")
        if not isinstance(value.get("files"), list) or not value["files"]:
            raise ValueError("Native checkpoint files must be a nonempty list")
        names = set()
        for file in value["files"]:
            if not isinstance(file, Mapping):
                raise TypeError("Native checkpoint file must be an object")
            path = file.get("path")
            if (
                not isinstance(path, str)
                or not path
                or path.startswith("/")
                or "\\" in path
                or ":" in path
                or any(part in {"", ".", ".."} for part in path.split("/"))
                or path in names
            ):
                raise ValueError("Native checkpoint inventory paths must be unique relative file paths")
            if type(file.get("size")) is not int or file["size"] <= 0:
                raise ValueError("Native checkpoint file sizes must be positive integers")
            names.add(path)
        if not {"trainer_state.pt", "data.pt"}.issubset(names) or not any(
            path.startswith("policy/") and PurePosixPath(path).suffix in {".pt", ".bin", ".distcp", ".safetensors"}
            for path in names
        ):
            raise ValueError("Native checkpoint inventory requires trainer/data state and policy weights")
        return cls(
            checkpoint_path=value["checkpoint_path"],
            global_step=value["global_step"],
            trainer_state_sha256=value["trainer_state_sha256"],
            files=tuple(CheckpointFile(**file) for file in value["files"]),
        )


@dataclass(frozen=True)
class TrainingReceipt:
    run_id: str
    attempt_id: str
    request_fingerprint: str
    completion_mode: CompletionMode
    global_step: int
    checkpoint: NativeCheckpoint | None = None
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["completion_mode"] = self.completion_mode.value
        if self.checkpoint is None:
            del result["checkpoint"]
        else:
            result["checkpoint"] = self.checkpoint.to_dict()
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrainingReceipt":
        if type(value.get("schema_version")) is not int or value["schema_version"] != 1:
            raise ValueError("Unsupported training receipt schema_version")
        _validate_step(value.get("global_step"))
        _validate_digest(value.get("request_fingerprint"))
        for field in ("run_id", "attempt_id"):
            if not isinstance(value.get(field), str) or not value[field].strip():
                raise ValueError(f"Training receipt {field} must be nonempty")
        mode = CompletionMode(value["completion_mode"])
        if ("checkpoint" in value) != (mode == CompletionMode.CHECKPOINT):
            raise ValueError("Training receipt checkpoint does not match completion_mode")
        return cls(
            run_id=value["run_id"],
            attempt_id=value["attempt_id"],
            request_fingerprint=value["request_fingerprint"],
            completion_mode=mode,
            global_step=value["global_step"],
            checkpoint=NativeCheckpoint.from_dict(value["checkpoint"]) if "checkpoint" in value else None,
        )


def _validate_step(value: Any) -> None:
    if type(value) is not int or value < 0:
        raise ValueError("Completed optimizer step must be a nonnegative integer")


def _validate_digest(value: Any) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("Training completion digest must be a lowercase SHA-256 hex string")


def completion_mode(cfg: Mapping[str, Any]) -> CompletionMode | None:
    completion = cfg["trainer"].get("completion")
    return CompletionMode(completion["mode"]) if completion else None


def validate_completion_config(cfg: Mapping[str, Any]) -> None:
    """Reject training completion requests whose effective configuration saves forbidden artifacts."""
    mode = completion_mode(cfg)
    if mode is None:
        return
    trainer = cfg["trainer"]
    completion = trainer["completion"]
    for field in ("run_id", "attempt_id", "request_fingerprint", "receipt_uri"):
        if not isinstance(completion.get(field), str) or not completion[field].strip():
            raise ValueError(f"trainer.completion.{field} must be a nonempty string")
    # Require legacy intervals to be disabled too: a later callback composition
    # must not silently reactivate a prohibited side effect.
    if int(trainer.get("hf_save_interval", -1)) > 0:
        raise ValueError("Training completion forbids trainer.hf_save_interval > 0")
    if mode == CompletionMode.METRICS and int(trainer.get("ckpt_interval", -1)) > 0:
        raise ValueError("Metrics completion forbids trainer.ckpt_interval > 0")
    for callback in trainer.get("callbacks") or ():
        callback_type = callback.get("type")
        if callback_type == "hf_model_save" and int(callback.get("save_steps", -1)) > 0:
            raise ValueError("Training completion forbids HF model save callbacks")
        if mode == CompletionMode.METRICS and callback_type == "checkpoint" and int(callback.get("save_steps", 10)) > 0:
            raise ValueError("Metrics completion forbids checkpoint callbacks")
