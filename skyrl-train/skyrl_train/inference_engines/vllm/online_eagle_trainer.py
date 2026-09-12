"""Single-GPU online EAGLE-3 updates from sealed vLLM rollout captures."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
import time
from typing import Any, Iterator, Mapping
from uuid import uuid4

from safetensors.torch import load_file
import torch
from torch import nn

from marinskyrl.hf_model import sha256_file
from marinskyrl.resource_locator import is_cloud_uri, join_resource_path
from marinskyrl.speculative_decoding import SpeculatorTrainingConfig
from skyrl_train.io import io


_CANDIDATE_FORMAT = "marinskyrl-online-eagle-candidate"
_CAPTURE_FORMAT = "vllm-online-eagle-capture"
_FAILURE_FORMAT = "marinskyrl-online-eagle-failure"
_SERVED_FORMAT = "marinskyrl-served-speculator"
_TRAINER_STATE_FORMAT = "marinskyrl-online-eagle-trainer-state"
_TRAINER_STATE_VERSION = 2
_MANIFEST_FILENAME = "manifest.json"
_MERGED_CAPTURE_DIRECTORY = "merged"
ONLINE_EAGLE_SCRATCH_ROOT = Path("/tmp/marinskyrl-online-eagle")


def capture_rank_directory(capture_root: str | Path, worker_rank: int) -> Path:
    """Return the shared per-DP-rank capture directory."""
    return Path(capture_root) / f"rank-{worker_rank:05d}"


def remove_online_eagle_scratch(path: str | Path) -> None:
    """Remove one bounded child of the online-EAGLE scratch root."""
    allowed_root = ONLINE_EAGLE_SCRATCH_ROOT.resolve()
    requested = Path(path).resolve()
    if requested == allowed_root or not requested.is_relative_to(allowed_root):
        raise ValueError(f"Refusing to remove online EAGLE scratch outside a process tree: {path}")
    shutil.rmtree(requested, ignore_errors=True)


def cleanup_online_eagle_training_job(job: Mapping[str, Any], *, remove_candidate: bool) -> None:
    """Reclaim one joined or aborted trainer job on its inference node."""
    remove_online_eagle_scratch(Path(job["capture_dir"]).parent)
    if remove_candidate:
        remove_online_eagle_scratch(job["output_dir"])


def preserve_online_eagle_failure(job: "OnlineEagleTrainingJob", error: BaseException) -> str:
    """Hard-link the bounded inputs needed to diagnose one failed update."""
    process_root = Path(job.output_dir).parents[1]
    failure_root = process_root / "failures"
    failure_root.mkdir(parents=True, exist_ok=True)
    for prior in failure_root.iterdir():
        if prior.is_dir():
            shutil.rmtree(prior)
        else:
            prior.unlink()
    destination = failure_root / f"step-{job.step}"
    staging = destination.with_name(f".{destination.name}.tmp-{uuid4().hex}")
    staging.mkdir()
    try:
        shutil.copytree(job.capture_dir, staging / "capture", copy_function=os.link)
        shutil.copytree(job.draft_model_dir, staging / "incumbent", copy_function=os.link)
        failure = {
            "format": _FAILURE_FORMAT,
            "format_version": 1,
            "complete": True,
            "step": job.step,
            "error": f"{type(error).__name__}: {error}",
            "capture_inventory": _directory_inventory(staging / "capture"),
            "incumbent_inventory": _directory_inventory(staging / "incumbent"),
        }
        (staging / _MANIFEST_FILENAME).write_text(json.dumps(failure, indent=2, sort_keys=True))
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return str(destination)


def publish_online_eagle_failure_bundle(source: str, destination: str) -> dict[str, Any]:
    """Publish one bounded forensic bundle, committing its manifest last."""
    root = Path(source)
    manifest_path = root / _MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != _FAILURE_FORMAT or not manifest.get("complete", False):
        raise ValueError(f"Invalid online EAGLE failure bundle: {source}")
    destination_manifest = join_resource_path(destination, _MANIFEST_FILENAME)
    if io.exists(destination_manifest):
        existing = json.loads(io.read_bytes(destination_manifest))
        if existing != manifest:
            raise FileExistsError(f"A different online EAGLE failure bundle exists: {destination}")
        return {**manifest, "path": destination_manifest}
    if is_cloud_uri(destination):
        io.upload_directory(str(root / "capture"), join_resource_path(destination, "capture"))
        io.upload_directory(str(root / "incumbent"), join_resource_path(destination, "incumbent"))
        for name in ("job.json", "trainer.log"):
            path = root / name
            if path.is_file():
                io.upload_file(str(path), join_resource_path(destination, name))
        io.upload_file(str(manifest_path), destination_manifest)
    else:
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = target.with_name(f".{target.name}.tmp-{uuid4().hex}")
        shutil.copytree(root, staging)
        try:
            os.replace(staging, target)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    return {**manifest, "path": destination_manifest}


def child_cuda_visible_device(visible_devices: str | None, device_index: int | None) -> str | None:
    """Resolve the parent worker's logical device to one child-visible device."""
    if device_index is None:
        return None
    if not visible_devices:
        return str(device_index)
    devices = [device.strip() for device in visible_devices.split(",")]
    if any(not device for device in devices):
        raise RuntimeError(f"Invalid CUDA_VISIBLE_DEVICES mapping: {visible_devices!r}")
    if device_index < 0 or device_index >= len(devices):
        raise RuntimeError(f"CUDA device index {device_index} is outside CUDA_VISIBLE_DEVICES={','.join(devices)}")
    return devices[device_index]


@dataclass(frozen=True)
class OnlineEagleTrainingJob:
    """Validated subprocess contract for one bounded draft update."""

    step: int
    capture_dir: str
    draft_model_dir: str
    initial_draft_source_identity: str
    output_dir: str
    result_path: str
    failure_artifact_path: str
    num_speculative_tokens: int
    seed: int
    training: SpeculatorTrainingConfig

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "OnlineEagleTrainingJob":
        required = {
            "step",
            "capture_dir",
            "draft_model_dir",
            "initial_draft_source_identity",
            "output_dir",
            "result_path",
            "failure_artifact_path",
            "num_speculative_tokens",
            "seed",
            "training",
        }
        if set(value) != required:
            missing = required - set(value)
            unknown = set(value) - required
            details = []
            if missing:
                details.append(f"missing {', '.join(sorted(missing))}")
            if unknown:
                details.append(f"unknown {', '.join(sorted(unknown))}")
            raise ValueError(f"Invalid online EAGLE training job: {'; '.join(details)}")
        step = value["step"]
        seed = value["seed"]
        num_speculative_tokens = value["num_speculative_tokens"]
        if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
            raise ValueError("Online EAGLE training job step must be a positive integer")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("Online EAGLE training job seed must be an integer")
        if (
            isinstance(num_speculative_tokens, bool)
            or not isinstance(num_speculative_tokens, int)
            or num_speculative_tokens <= 0
        ):
            raise ValueError("Online EAGLE num_speculative_tokens must be a positive integer")
        paths = {field: value[field] for field in ("capture_dir", "draft_model_dir", "output_dir", "result_path")}
        invalid_paths = [
            field for field, path in paths.items() if not isinstance(path, str) or not Path(path).is_absolute()
        ]
        if invalid_paths:
            raise ValueError(f"Online EAGLE job paths must be absolute: {', '.join(invalid_paths)}")
        failure_artifact_path = value["failure_artifact_path"]
        if not isinstance(failure_artifact_path, str) or not failure_artifact_path:
            raise ValueError("Online EAGLE failure_artifact_path must be nonempty")
        if not is_cloud_uri(failure_artifact_path) and not Path(failure_artifact_path).is_absolute():
            raise ValueError("Online EAGLE failure_artifact_path must be an absolute path or cloud URI")
        initial_identity = value["initial_draft_source_identity"]
        if not isinstance(initial_identity, str) or not initial_identity:
            raise ValueError("Online EAGLE initial_draft_source_identity must be nonempty")
        return cls(
            step=step,
            capture_dir=paths["capture_dir"],
            draft_model_dir=paths["draft_model_dir"],
            initial_draft_source_identity=initial_identity,
            output_dir=paths["output_dir"],
            result_path=paths["result_path"],
            failure_artifact_path=failure_artifact_path,
            num_speculative_tokens=num_speculative_tokens,
            seed=seed,
            training=SpeculatorTrainingConfig.from_mapping(value["training"], context="online EAGLE training job"),
        )

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OnlineEagleCaptureConfig:
    """One bounded capture interval passed from the trainer to vLLM."""

    step: int
    max_tokens: int
    max_sequences_per_prompt_group: int
    target_revision: str
    draft_revision: str
    reserved_gpu_memory_gib: float

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OnlineEagleEvaluation:
    """Mean EAGLE loss and next-token agreement on one window set."""

    mean_loss: float
    agreement: float


@dataclass(frozen=True)
class OnlineEagleUpdateResult:
    """Typed result envelope shared by the trainer subprocess and coordinator."""

    active: bool
    accepted: bool
    step: int | None = None
    parent_draft_revision: str | None = None
    trained_against_target_revision: str | None = None
    train_sequences: int | None = None
    holdout_sequences: int | None = None
    train_loss: float | None = None
    incumbent_holdout_loss: float | None = None
    candidate_holdout_loss: float | None = None
    incumbent_holdout_agreement: float | None = None
    candidate_holdout_agreement: float | None = None
    holdout_loss_increase: float | None = None
    holdout_agreement_decrease: float | None = None
    max_validation_loss_increase: float | None = None
    max_validation_agreement_decrease: float | None = None
    duration_seconds: float | None = None
    candidate_dir: str | None = None
    draft_revision: str | None = None
    deferred: bool = False
    error: str | None = None
    log_path: str | None = None
    failure_dir: str | None = None
    failure_artifact_path: str | None = None
    failure_preservation_error: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "OnlineEagleUpdateResult":
        fields = cls.__dataclass_fields__
        return cls(**{name: value[name] for name in fields if name in value})

    def to_mapping(self) -> dict[str, Any]:
        return {name: value for name, value in asdict(self).items() if value is not None}


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid4().hex}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    os.replace(temporary, path)


def _directory_inventory(directory: Path) -> dict[str, dict[str, Any]]:
    return {
        str(path.relative_to(directory)): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(directory.rglob("*"))
        if path.is_file() and ".cache" not in path.parts
    }


def publish_speculator_checkpoint(
    source_dir: str,
    destination: str,
    *,
    draft_revision: str,
    served_target_revision: str,
) -> dict[str, Any]:
    """Publish the exact served draft, writing its completion manifest last."""
    source = Path(source_dir)
    if not source.is_dir():
        raise FileNotFoundError(f"Served online EAGLE checkpoint does not exist: {source}")
    inventory = _directory_inventory(source)
    lineage: dict[str, Any] = {"initial_source_identity": draft_revision}
    candidate_manifest_path = source / _MANIFEST_FILENAME
    if candidate_manifest_path.is_file():
        candidate_manifest = json.loads(candidate_manifest_path.read_text())
        if candidate_manifest.get("format") == _CANDIDATE_FORMAT and candidate_manifest.get("complete", False):
            lineage = {
                field: candidate_manifest[field]
                for field in (
                    "initial_source_identity",
                    "parent_draft_revision",
                    "trained_against_target_revision",
                    "capture_manifest_sha256",
                    "training",
                    "metrics",
                )
                if field in candidate_manifest
            }
    manifest = {
        "format": _SERVED_FORMAT,
        "format_version": 1,
        "complete": True,
        "draft_revision": draft_revision,
        "served_target_revision": served_target_revision,
        "lineage": lineage,
        "inventory": inventory,
    }
    manifest_path = join_resource_path(destination, _MANIFEST_FILENAME)
    if io.exists(manifest_path):
        existing = json.loads(io.read_bytes(manifest_path))
        if existing != manifest:
            raise FileExistsError(f"A different served speculator checkpoint already exists: {destination}")
        return {**manifest, "path": manifest_path}

    def build_local_layout(root: Path) -> None:
        shutil.copytree(
            source,
            root / "weights",
            ignore=shutil.ignore_patterns(".cache"),
        )
        (root / _MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2, sort_keys=True))

    if is_cloud_uri(destination):
        with tempfile.TemporaryDirectory(prefix="marinskyrl-speculator-publish-") as temporary:
            layout = Path(temporary)
            build_local_layout(layout)
            io.upload_directory(str(layout / "weights"), join_resource_path(destination, "weights"))
            io.upload_file(str(layout / _MANIFEST_FILENAME), manifest_path)
    else:
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = target.with_name(f".{target.name}.tmp-{uuid4().hex}")
        staging.mkdir()
        try:
            build_local_layout(staging)
            if target.exists():
                raise FileExistsError(f"Speculator checkpoint already exists: {target}")
            os.replace(staging, target)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    return {**manifest, "path": manifest_path}


def _validate_served_speculator_root(root: Path, source: str) -> dict[str, Any]:
    manifest_path = root / _MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != _SERVED_FORMAT or not manifest.get("complete", False):
        raise ValueError(f"Incomplete served speculator checkpoint: {source}")
    if _directory_inventory(root / "weights") != manifest["inventory"]:
        raise ValueError(f"Served speculator checkpoint inventory mismatch: {source}")
    return manifest


def export_served_speculator_checkpoint(source: str, destination: str) -> dict[str, Any]:
    """Copy a validated served draft to its policy export, committing its manifest last."""
    with io.local_read_dir(source) as local_source:
        root = Path(local_source)
        manifest = _validate_served_speculator_root(root, source)
        destination_manifest = join_resource_path(destination, _MANIFEST_FILENAME)
        if io.exists(destination_manifest):
            with io.local_read_dir(destination) as local_destination:
                existing = _validate_served_speculator_root(Path(local_destination), destination)
            if existing != manifest:
                raise FileExistsError(f"A different served speculator export already exists: {destination}")
            return {**manifest, "path": destination_manifest}

        if is_cloud_uri(destination):
            io.upload_directory(str(root / "weights"), join_resource_path(destination, "weights"))
            io.upload_file(str(root / _MANIFEST_FILENAME), destination_manifest)
        else:
            target = Path(destination)
            target.parent.mkdir(parents=True, exist_ok=True)
            staging = target.with_name(f".{target.name}.tmp-{uuid4().hex}")
            shutil.copytree(root, staging)
            try:
                if target.exists():
                    raise FileExistsError(f"Incomplete served speculator export already exists: {target}")
                os.replace(staging, target)
            except BaseException:
                shutil.rmtree(staging, ignore_errors=True)
                raise
    return {**manifest, "path": destination_manifest}


def _validate_restored_trainer_state(path: Path) -> None:
    if not path.is_file():
        return
    state = torch.load(path, map_location="cpu", weights_only=False)
    if (
        state.get("format") != _TRAINER_STATE_FORMAT
        or state.get("format_version") != _TRAINER_STATE_VERSION
        or state.get("master_parameter_dtype") != str(torch.float32)
        or "master_parameters" not in state
    ):
        raise ValueError(f"Incompatible online EAGLE trainer state: {path}")


def restore_speculator_checkpoint(source: str, destination: str) -> dict[str, Any]:
    """Stage and validate a complete served-draft checkpoint for in-place reload."""
    with io.local_read_dir(source) as local_source:
        root = Path(local_source)
        manifest = _validate_served_speculator_root(root, source)
        weights = root / "weights"
        _validate_restored_trainer_state(weights / "trainer_state.pt")
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = target.with_name(f".{target.name}.tmp-{uuid4().hex}")
        shutil.copytree(weights, staging)
        try:
            candidate_manifest_path = staging / _MANIFEST_FILENAME
            candidate_manifest = (
                json.loads(candidate_manifest_path.read_text()) if candidate_manifest_path.exists() else {}
            )
            if candidate_manifest.get("format") != _CANDIDATE_FORMAT:
                weights_path = staging / "model.safetensors"
                tensors = load_file(weights_path)
                candidate_manifest_path.write_text(
                    json.dumps(
                        {
                            "format": _CANDIDATE_FORMAT,
                            "format_version": 1,
                            "complete": True,
                            "draft_revision": manifest["draft_revision"],
                            "weights_path": weights_path.name,
                            "weights_sha256": sha256_file(weights_path),
                            "tensor_inventory": {
                                name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                                for name, value in tensors.items()
                            },
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
            if target.exists():
                raise FileExistsError(f"Restored speculator destination already exists: {target}")
            os.replace(staging, target)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    return {**manifest, "restored_path": destination}


def partition_capture_windows(
    windows: list[dict[str, Any]],
    *,
    step: int,
    holdout_fraction: float,
    min_train_sequences: int,
    min_holdout_sequences: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return a deterministic partition that keeps every prompt group in one split."""
    required = min_train_sequences + min_holdout_sequences
    if len(windows) < required:
        raise ValueError(f"Online EAGLE update needs at least {required} captured windows, got {len(windows)}")
    ordered_windows = sorted(windows, key=lambda window: str(window["request_id"]))
    groups: dict[str, list[dict[str, Any]]] = {}
    for window in ordered_windows:
        group_id = str(window.get("group_id", window["request_id"]))
        groups.setdefault(group_id, []).append(window)
    ordered_groups = sorted(
        groups.items(),
        key=lambda item: hashlib.sha256(f"{step}:{item[0]}".encode()).digest(),
    )
    target_holdout_count = max(min_holdout_sequences, math.ceil(len(windows) * holdout_fraction))
    holdout_groups: set[str] = set()
    holdout_count = 0
    for group_id, group_windows in ordered_groups:
        if holdout_count >= target_holdout_count:
            break
        if len(windows) - holdout_count - len(group_windows) < min_train_sequences:
            continue
        holdout_groups.add(group_id)
        holdout_count += len(group_windows)
    if holdout_count < min_holdout_sequences:
        raise ValueError(
            "Online EAGLE capture cannot form group-disjoint train and holdout splits with "
            f"at least {min_train_sequences} and {min_holdout_sequences} sequences"
        )
    holdout = [
        window for window in ordered_windows if str(window.get("group_id", window["request_id"])) in holdout_groups
    ]
    train = [
        window for window in ordered_windows if str(window.get("group_id", window["request_id"])) not in holdout_groups
    ]
    return train, holdout


def _load_and_validate_capture(capture_dir: Path, *, verify_digests: bool = True) -> dict[str, Any]:
    manifest_path = capture_dir / _MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != _CAPTURE_FORMAT or not manifest.get("active", False):
        raise ValueError(f"Invalid online EAGLE capture manifest: {manifest_path}")
    for window in manifest["windows"]:
        path = capture_dir / window["path"]
        if not path.is_file():
            raise ValueError(f"Captured EAGLE window is missing: {path}")
        if verify_digests and sha256_file(path) != window["sha256"]:
            raise ValueError(f"Captured EAGLE window digest mismatch: {path}")
    target = manifest["target"]
    for key in ("weights", "config"):
        path = capture_dir / target[f"{key}_path"]
        if not path.is_file():
            raise ValueError(f"Captured target {key} is missing: {path}")
        if verify_digests and sha256_file(path) != target[f"{key}_sha256"]:
            raise ValueError(f"Captured target {key} digest mismatch: {path}")
    return manifest


def _hardlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.link(source, destination)


def merge_online_eagle_captures(
    capture_root: str,
    *,
    expected_workers: int,
    expected_step: int,
    max_tokens: int,
    max_sequences_per_prompt_group: int,
) -> dict[str, Any]:
    """Merge colocated DP-rank captures into one bounded, immutable trainer input."""
    root = Path(capture_root)
    if expected_workers <= 0:
        raise ValueError("Online EAGLE capture worker count must be positive")
    if max_tokens <= 0 or max_sequences_per_prompt_group <= 0:
        raise ValueError("Online EAGLE merged capture bounds must be positive")
    destination = root / _MERGED_CAPTURE_DIRECTORY
    if destination.exists():
        raise FileExistsError(f"Merged online EAGLE capture already exists: {destination}")

    manifests: list[tuple[Path, dict[str, Any]]] = []
    for worker_rank in range(expected_workers):
        directory = capture_rank_directory(root, worker_rank)
        manifest = _load_and_validate_capture(directory, verify_digests=False)
        if manifest.get("worker_rank") != worker_rank:
            raise ValueError(
                "Online EAGLE capture worker-rank mismatch: "
                f"expected {worker_rank}, got {manifest.get('worker_rank')!r}"
            )
        manifests.append((directory, manifest))

    identity_fields = (
        "format",
        "format_version",
        "step",
        "target_revision",
        "draft_revision",
        "aux_layer_ids",
        "head_input_semantics",
    )
    baseline = manifests[0][1]
    if baseline.get("step") != expected_step:
        raise ValueError(f"Online EAGLE capture step mismatch: expected {expected_step}, got {baseline.get('step')!r}")
    baseline_identity = {field: baseline.get(field) for field in identity_fields}
    baseline_target = baseline["target"]
    target_identity = {
        "weights_sha256": baseline_target["weights_sha256"],
        "config_sha256": baseline_target["config_sha256"],
        "inventory": baseline_target["inventory"],
    }
    candidates: list[tuple[Path, dict[str, Any]]] = []
    request_ids: set[str] = set()
    for directory, manifest in manifests:
        identity = {field: manifest.get(field) for field in identity_fields}
        if identity != baseline_identity:
            raise ValueError("Online EAGLE DP captures do not share one target/draft identity")
        target = manifest["target"]
        if {
            "weights_sha256": target["weights_sha256"],
            "config_sha256": target["config_sha256"],
            "inventory": target["inventory"],
        } != target_identity:
            raise ValueError("Online EAGLE DP captures do not share one target snapshot")
        for window in manifest["windows"]:
            request_id = str(window["request_id"])
            if request_id in request_ids:
                raise ValueError(f"Duplicate online EAGLE request across DP captures: {request_id}")
            request_ids.add(request_id)
            candidates.append((directory, window))

    step = int(baseline["step"])
    candidates.sort(
        key=lambda item: hashlib.sha256(
            f"{step}:{item[1].get('group_id', item[1]['request_id'])}:{item[1]['request_id']}".encode()
        ).digest()
    )
    selected: list[tuple[Path, dict[str, Any]]] = []
    group_counts: dict[str, int] = {}
    selected_tokens = 0
    for directory, window in candidates:
        group_id = str(window.get("group_id", window["request_id"]))
        tokens = window.get("tokens")
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0:
            raise ValueError(f"Invalid online EAGLE captured-window token count: {tokens!r}")
        if group_counts.get(group_id, 0) >= max_sequences_per_prompt_group:
            continue
        if selected_tokens + tokens > max_tokens:
            continue
        selected.append((directory, window))
        selected_tokens += tokens
        group_counts[group_id] = group_counts.get(group_id, 0) + 1

    staging = root / f".{_MERGED_CAPTURE_DIRECTORY}.tmp-{uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        windows: list[dict[str, Any]] = []
        for index, (directory, source_window) in enumerate(selected):
            filename = f"window-{index:06d}.safetensors"
            _hardlink(directory / source_window["path"], staging / filename)
            windows.append({**source_window, "path": filename})

        baseline_directory = manifests[0][0]
        merged_target = dict(baseline_target)
        for kind in ("weights", "config"):
            source_name = baseline_target[f"{kind}_path"]
            destination_name = "target.safetensors" if kind == "weights" else "target-config.json"
            _hardlink(baseline_directory / source_name, staging / destination_name)
            merged_target[f"{kind}_path"] = destination_name

        manifest = {
            **baseline_identity,
            "active": True,
            "worker_rank": 0,
            "worker_ranks": list(range(expected_workers)),
            "windows": windows,
            "captured_rows": selected_tokens,
            "source_captured_rows": sum(int(manifest.get("captured_rows", 0)) for _, manifest in manifests),
            "source_windows": len(candidates),
            "dropped_requests": sum(int(manifest.get("dropped_requests", 0)) for _, manifest in manifests),
            "dropped_windows": sum(int(manifest.get("dropped_windows", 0)) for _, manifest in manifests),
            "unselected_windows": len(candidates) - len(selected),
            "target": merged_target,
        }
        (staging / _MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2, sort_keys=True))
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    for directory, _ in manifests:
        shutil.rmtree(directory, ignore_errors=True)
    return {**manifest, "path": str(destination / _MANIFEST_FILENAME)}


def _prepare_model(draft_model_dir: Path, capture_dir: Path, device: torch.device):
    from speculators.models.eagle3 import (  # noqa: PLC0415
        Eagle3DraftModel,
        Eagle3SpeculatorConfig,
    )
    from transformers import PreTrainedModel  # noqa: PLC0415

    config = Eagle3SpeculatorConfig.from_pretrained(
        draft_model_dir,
        local_files_only=True,
    )
    # Speculators' FlexAttention mask is block-padded, but online captures retain
    # their exact sequence length. EAGLE's time-shift extension can therefore
    # produce real Q/KV dimensions smaller than the padded BlockMask contract.
    # This bounded one-layer trainer favors exact mask semantics over that kernel.
    config.transformer_layer_config._attn_implementation = "eager"  # noqa: SLF001
    # The public embedding-free Snowball checkpoint intentionally has no verifier
    # path. Bypass Speculators' generic post-load verifier hook because the exact
    # target-owned tensors come from this sealed rollout, not a second HF model.
    model = PreTrainedModel.from_pretrained.__func__(
        Eagle3DraftModel,
        draft_model_dir,
        config=config,
        local_files_only=True,
    )
    target = load_file(capture_dir / "target.safetensors")
    embedding = target["model.embed_tokens.weight"]
    head = target["lm_head.weight"]
    if model.t2d is None or not torch.any(model.t2d):
        raise ValueError("The EAGLE checkpoint has no target-to-draft vocabulary map")
    draft_head = head[model.t2d.to(dtype=torch.bool)]
    with torch.no_grad():
        model.embed_tokens.weight.copy_(embedding)
        model.lm_head.weight.copy_(draft_head)
        model.verifier_lm_head.weight.copy_(draft_head)
    model.verifier_norm = nn.Identity()
    model.verifier_gate_down = None
    model.verifier_gate_up = None
    for name, parameter in model.named_parameters():
        target_owned = name == "embed_tokens.weight" or name == "lm_head.weight" or name.startswith("verifier_")
        parameter.requires_grad_(not target_owned)
    return model.to(device=device, dtype=_serving_dtype(device))


def _serving_dtype(device: torch.device) -> torch.dtype:
    return torch.bfloat16 if device.type == "cuda" else torch.float32


def _forward_context(device: torch.device) -> Any:
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _require_finite_tensor(tensor: torch.Tensor, *, label: str) -> None:
    if (tensor.is_floating_point() or tensor.is_complex()) and not bool(torch.isfinite(tensor).all()):
        raise FloatingPointError(f"Online EAGLE found a non-finite tensor: {label}")


def _require_finite_trainable_state(model: nn.Module, optimizer: torch.optim.Optimizer | None = None) -> None:
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            _require_finite_tensor(parameter, label=f"parameter {name}")
    if optimizer is not None:
        for parameter_index, state in enumerate(optimizer.state.values()):
            for name, value in state.items():
                if isinstance(value, torch.Tensor):
                    _require_finite_tensor(value, label=f"optimizer parameter {parameter_index} state {name}")


def _convert_trainable_parameters(model: nn.Module, dtype: torch.dtype) -> None:
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.to(dtype=dtype)
    _require_finite_trainable_state(model)


def _capture_trainable_master_state(model: nn.Module) -> dict[str, torch.Tensor]:
    state = {
        name: parameter.detach().to(device="cpu", dtype=torch.float32).contiguous()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if not state:
        raise ValueError("Online EAGLE model has no trainable master parameters")
    return state


def _restore_trainable_master_state(
    model: nn.Module,
    state: Mapping[str, torch.Tensor],
    *,
    serving_dtype: torch.dtype,
) -> None:
    parameters = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    if set(state) != set(parameters):
        raise ValueError("Online EAGLE FP32 master tensor inventory does not match the draft model")
    with torch.no_grad():
        for name, parameter in parameters.items():
            master = state[name]
            if master.dtype != torch.float32 or master.shape != parameter.shape:
                raise ValueError(f"Invalid online EAGLE FP32 master tensor: {name}")
            _require_finite_tensor(master, label=f"FP32 master tensor {name}")
            master = master.to(device=parameter.device)
            if not torch.equal(master.to(dtype=serving_dtype), parameter.detach().to(dtype=serving_dtype)):
                raise ValueError(f"Online EAGLE FP32 master does not round to the served tensor: {name}")
            parameter.copy_(master)
    _require_finite_trainable_state(model)


def _load_batch(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    window = load_file(path)
    input_ids = window["input_ids"]
    if input_ids.shape[0] < 2:
        raise ValueError(f"Online EAGLE window is too short: {path}")
    batch = {
        "input_ids": input_ids[1:].unsqueeze(0),
        "hidden_states": window["hidden_states"][:-1].unsqueeze(0),
        "verifier_last_hidden_states": window["head_input_hidden_states"][1:].unsqueeze(0),
        "loss_mask": window["loss_mask"][1:].unsqueeze(0),
        "position_ids": window["position_ids"][1:].unsqueeze(0),
    }
    batch["document_ids"] = torch.zeros_like(batch["input_ids"])
    for name, value in batch.items():
        _require_finite_tensor(value, label=f"capture {path} tensor {name}")
    return {name: value.to(device) for name, value in batch.items()}


def _pack_windows(windows: list[dict[str, Any]], max_tokens: int) -> list[list[dict[str, Any]]]:
    """Group request-local forwards into bounded gradient-accumulation steps."""
    batches: list[list[dict[str, Any]]] = []
    batch: list[dict[str, Any]] = []
    batch_tokens = 0
    for window in windows:
        window_tokens = max(1, int(window["tokens"]) - 1)
        if batch and batch_tokens + window_tokens > max_tokens:
            batches.append(batch)
            batch = []
            batch_tokens = 0
        batch.append(window)
        batch_tokens += window_tokens
    if batch:
        batches.append(batch)
    return batches


def _load_window_group(
    windows: list[dict[str, Any]],
    capture_dir: Path,
    device: torch.device,
) -> Iterator[tuple[dict[str, Any], dict[str, torch.Tensor]]]:
    """Load independent histories without crossing teacher-forcing boundaries."""
    for window in windows:
        yield window, _load_batch(capture_dir / window["path"], device)


def _evaluate(
    model: nn.Module,
    capture_dir: Path,
    windows: list[dict[str, Any]],
    num_speculative_tokens: int,
    max_tokens_per_micro_batch: int,
    loss_config,
) -> OnlineEagleEvaluation:
    """Evaluate mean loss and next-token agreement on the supplied windows."""
    weighted_loss = 0.0
    loss_tokens = 0.0
    correct = 0.0
    total = 0.0
    model.eval()
    with torch.no_grad():
        for window_group in _pack_windows(windows, max_tokens_per_micro_batch):
            device = next(model.parameters()).device
            for window, batch in _load_window_group(window_group, capture_dir, device):
                path = capture_dir / window["path"]
                with _forward_context(device):
                    _, loss, metrics = model(
                        **batch,
                        ttt_steps=num_speculative_tokens,
                        loss_config=loss_config,
                    )
                _require_finite_tensor(loss, label=f"evaluation loss for {path}")
                supervised_tokens = float(batch["loss_mask"].sum())
                weighted_loss += float(loss) * supervised_tokens
                loss_tokens += supervised_tokens
                for index in range(num_speculative_tokens):
                    correct_value = metrics[f"full_acc_{index}_sum"]
                    total_value = metrics[f"full_acc_{index}_total"]
                    _require_finite_tensor(correct_value, label=f"evaluation agreement numerator for {path}")
                    _require_finite_tensor(total_value, label=f"evaluation agreement denominator for {path}")
                    correct += float(correct_value)
                    total += float(total_value)
    if loss_tokens == 0:
        raise ValueError("Online EAGLE holdout contains no supervised tokens")
    return OnlineEagleEvaluation(
        mean_loss=weighted_loss / loss_tokens,
        agreement=correct / total if total else 0.0,
    )


def _candidate_state(model: nn.Module, *, serving_dtype: torch.dtype) -> dict[str, torch.Tensor]:
    state = {}
    for name, value in model.state_dict().items():
        target_owned = "embed_tokens" in name or name.startswith("verifier_")
        if not target_owned:
            if value.is_floating_point() and value.dtype != serving_dtype:
                raise ValueError(
                    f"Online EAGLE candidate tensor {name} has dtype {value.dtype}, expected {serving_dtype}"
                )
            _require_finite_tensor(value, label=f"candidate tensor {name}")
            state[name] = value.detach().cpu().contiguous()
    return state


def candidate_is_acceptable(
    *,
    incumbent_loss: float,
    candidate_loss: float,
    incumbent_agreement: float,
    candidate_agreement: float,
    max_loss_increase: float,
    max_agreement_decrease: float,
) -> bool:
    """Apply the declared same-holdout loss and agreement tolerances."""
    values = (incumbent_loss, candidate_loss, incumbent_agreement, candidate_agreement)
    return (
        all(math.isfinite(value) for value in values)
        and candidate_loss <= incumbent_loss + max_loss_increase
        and candidate_agreement >= incumbent_agreement - max_agreement_decrease
    )


def _restore_rng_states(saved_state: Mapping[str, Any], device: torch.device) -> None:
    """Restore trainer RNG state after a device-mapped checkpoint load."""
    torch.set_rng_state(saved_state["torch_rng_state"].cpu())
    cuda_rng_state = saved_state.get("cuda_rng_state")
    if device.type == "cuda" and cuda_rng_state is not None:
        # CUDA generator state is represented by a CPU ByteTensor. ``torch.load``
        # maps it to ``device`` with the optimizer tensors, so move it back first.
        torch.cuda.set_rng_state(cuda_rng_state.cpu(), device=device)
    python_rng_state = saved_state.get("python_rng_state")
    if python_rng_state is not None:
        random.setstate(python_rng_state)


def _save_candidate(
    model,
    optimizer: torch.optim.Optimizer,
    output_dir: Path,
    lineage: dict[str, Any],
    *,
    serving_dtype: torch.dtype,
    master_parameters: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.with_name(f".{output_dir.name}.tmp-{uuid4().hex}")
    staging.mkdir()
    try:
        model.save_pretrained(
            staging,
            state_dict=_candidate_state(model, serving_dtype=serving_dtype),
            safe_serialization=True,
            max_shard_size="5GB",
        )
        weights_path = staging / "model.safetensors"
        if not weights_path.exists():
            raise RuntimeError("Online EAGLE candidate unexpectedly produced sharded weights")
        state = load_file(weights_path)
        trainer_state_path = staging / "trainer_state.pt"
        torch.save(
            {
                "format": _TRAINER_STATE_FORMAT,
                "format_version": _TRAINER_STATE_VERSION,
                "master_parameter_dtype": str(torch.float32),
                "served_parameter_dtype": str(serving_dtype),
                "master_parameters": dict(master_parameters),
                "optimizer": optimizer.state_dict(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                "python_rng_state": random.getstate(),
            },
            trainer_state_path,
        )
        weights_sha256 = sha256_file(weights_path)
        manifest = {
            "format": _CANDIDATE_FORMAT,
            "format_version": 1,
            "complete": True,
            **lineage,
            "weights_path": weights_path.name,
            "weights_sha256": weights_sha256,
            "trainer_state_path": trainer_state_path.name,
            "trainer_state_sha256": sha256_file(trainer_state_path),
            "tensor_inventory": {
                name: {"shape": list(value.shape), "dtype": str(value.dtype)} for name, value in state.items()
            },
        }
        (staging / _MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2, sort_keys=True))
        if output_dir.exists():
            raise FileExistsError(f"Online EAGLE candidate already exists: {output_dir}")
        os.replace(staging, output_dir)
        return {**manifest, "path": str(output_dir / _MANIFEST_FILENAME)}
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def run_training_job(job: OnlineEagleTrainingJob) -> OnlineEagleUpdateResult:
    """Train and gate one draft candidate against a sealed rollout holdout."""
    started_at = time.perf_counter()
    capture_dir = Path(job.capture_dir)
    draft_model_dir = Path(job.draft_model_dir)
    output_dir = Path(job.output_dir)
    manifest = _load_and_validate_capture(capture_dir)
    training = job.training
    train_windows, holdout_windows = partition_capture_windows(
        manifest["windows"],
        step=job.step,
        holdout_fraction=training.holdout_fraction,
        min_train_sequences=training.min_train_sequences,
        min_holdout_sequences=training.min_holdout_sequences,
    )
    seed = job.seed + job.step
    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _prepare_model(draft_model_dir, capture_dir, device)
    from speculators.losses import resolve_loss_config  # noqa: PLC0415

    loss_config = resolve_loss_config("kl_div", "fused" if device.type == "cuda" else "eager")
    serving_dtype = _serving_dtype(device)
    incumbent = _evaluate(
        model,
        capture_dir,
        holdout_windows,
        job.num_speculative_tokens,
        training.max_tokens_per_micro_batch,
        loss_config,
    )
    _convert_trainable_parameters(model, torch.float32)
    prior_trainer_state = draft_model_dir / "trainer_state.pt"
    saved_state = None
    if prior_trainer_state.exists():
        saved_state = torch.load(prior_trainer_state, map_location=device, weights_only=False)
        if (
            saved_state.get("format") != _TRAINER_STATE_FORMAT
            or saved_state.get("format_version") != _TRAINER_STATE_VERSION
            or saved_state.get("master_parameter_dtype") != str(torch.float32)
            or saved_state.get("served_parameter_dtype") != str(serving_dtype)
        ):
            raise ValueError(f"Incompatible online EAGLE trainer state: {prior_trainer_state}")
        master_parameters = saved_state.pop("master_parameters", None)
        if not isinstance(master_parameters, Mapping):
            raise ValueError(f"Online EAGLE trainer state has no FP32 masters: {prior_trainer_state}")
        _restore_trainable_master_state(model, master_parameters, serving_dtype=serving_dtype)
        del master_parameters
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=training.learning_rate)
    if saved_state is not None:
        optimizer.load_state_dict(saved_state["optimizer"])
        _restore_rng_states(saved_state, device)
    _require_finite_trainable_state(model, optimizer)

    weighted_train_loss = 0.0
    train_loss_tokens = 0.0
    model.train()
    for _epoch in range(training.epochs_per_update):
        epoch_windows = list(train_windows)
        random.shuffle(epoch_windows)
        for group_index, window_group in enumerate(_pack_windows(epoch_windows, training.max_tokens_per_micro_batch)):
            group_supervised_tokens = sum(int(window["supervised_tokens"]) for window in window_group)
            if group_supervised_tokens <= 0:
                raise ValueError("Online EAGLE training microbatch contains no supervised tokens")
            optimizer.zero_grad(set_to_none=True)
            for window, batch in _load_window_group(window_group, capture_dir, device):
                path = capture_dir / window["path"]
                supervised_tokens = int(batch["loss_mask"].sum())
                if supervised_tokens != int(window["supervised_tokens"]):
                    raise ValueError(
                        "Online EAGLE supervised-token count does not match its manifest: "
                        f"{path} has {supervised_tokens}, manifest says {window['supervised_tokens']}"
                    )
                with _forward_context(device):
                    _, loss, _metrics = model(
                        **batch,
                        ttt_steps=job.num_speculative_tokens,
                        loss_config=loss_config,
                    )
                _require_finite_tensor(loss, label=f"training loss for {path}")
                # Speculators' loss_function divides each window's masked loss
                # sum by its supervised-token count. Weight the per-window mean
                # here to reproduce one token-weighted optimizer microbatch.
                (loss * (supervised_tokens / group_supervised_tokens)).backward()
                weighted_train_loss += float(loss.detach()) * supervised_tokens
                train_loss_tokens += supervised_tokens
                print(
                    json.dumps(
                        {
                            "kind": "online_eagle_microbatch",
                            "step": job.step,
                            "epoch": _epoch,
                            "group": group_index,
                            "request_id": window["request_id"],
                            "path": str(path),
                            "supervised_tokens": supervised_tokens,
                            "loss": float(loss.detach()),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainable,
                max_norm=1.0,
                error_if_nonfinite=True,
            )
            _require_finite_tensor(gradient_norm, label=f"gradient norm for group {group_index}")
            optimizer.step()
            _require_finite_trainable_state(model, optimizer)
            print(
                json.dumps(
                    {
                        "kind": "online_eagle_optimizer_step",
                        "step": job.step,
                        "epoch": _epoch,
                        "group": group_index,
                        "supervised_tokens": group_supervised_tokens,
                        "gradient_norm": float(gradient_norm),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if train_loss_tokens == 0:
        raise ValueError("Online EAGLE training partition contains no supervised tokens")
    master_parameters = _capture_trainable_master_state(model)
    _convert_trainable_parameters(model, serving_dtype)
    candidate = _evaluate(
        model,
        capture_dir,
        holdout_windows,
        job.num_speculative_tokens,
        training.max_tokens_per_micro_batch,
        loss_config,
    )
    max_loss_increase = training.max_validation_loss_increase
    max_agreement_decrease = training.max_validation_agreement_decrease
    accepted = candidate_is_acceptable(
        incumbent_loss=incumbent.mean_loss,
        candidate_loss=candidate.mean_loss,
        incumbent_agreement=incumbent.agreement,
        candidate_agreement=candidate.agreement,
        max_loss_increase=max_loss_increase,
        max_agreement_decrease=max_agreement_decrease,
    )
    result = OnlineEagleUpdateResult(
        active=True,
        accepted=accepted,
        step=job.step,
        parent_draft_revision=manifest["draft_revision"],
        trained_against_target_revision=manifest["target_revision"],
        train_sequences=len(train_windows),
        holdout_sequences=len(holdout_windows),
        train_loss=weighted_train_loss / train_loss_tokens,
        incumbent_holdout_loss=incumbent.mean_loss,
        candidate_holdout_loss=candidate.mean_loss,
        incumbent_holdout_agreement=incumbent.agreement,
        candidate_holdout_agreement=candidate.agreement,
        holdout_loss_increase=candidate.mean_loss - incumbent.mean_loss,
        holdout_agreement_decrease=incumbent.agreement - candidate.agreement,
        max_validation_loss_increase=max_loss_increase,
        max_validation_agreement_decrease=max_agreement_decrease,
        duration_seconds=time.perf_counter() - started_at,
    )
    if accepted:
        draft_revision = f"draft-step-{job.step}"
        _save_candidate(
            model,
            optimizer,
            output_dir,
            {
                "draft_revision": draft_revision,
                "initial_source_identity": job.initial_draft_source_identity,
                "parent_draft_revision": manifest["draft_revision"],
                "trained_against_target_revision": manifest["target_revision"],
                "capture_manifest_sha256": sha256_file(capture_dir / _MANIFEST_FILENAME),
                "training": asdict(training),
                "metrics": result.to_mapping(),
            },
            serving_dtype=serving_dtype,
            master_parameters=master_parameters,
        )
        result = replace(result, candidate_dir=str(output_dir), draft_revision=draft_revision)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("job_config")
    args = parser.parse_args()
    job_path = Path(args.job_config)
    job = OnlineEagleTrainingJob.from_mapping(json.loads(job_path.read_text()))
    result_path = Path(job.result_path)
    try:
        result = run_training_job(job)
    except BaseException as error:
        failure_dir = None
        preservation_error = None
        try:
            failure_dir = preserve_online_eagle_failure(job, error)
        except BaseException as failure_error:
            preservation_error = f"{type(failure_error).__name__}: {failure_error}"
        _atomic_json(
            result_path,
            {
                "active": True,
                "accepted": False,
                "error": f"{type(error).__name__}: {error}",
                "failure_dir": failure_dir,
                "failure_artifact_path": job.failure_artifact_path,
                "failure_preservation_error": preservation_error,
            },
        )
        raise
    _atomic_json(result_path, result.to_mapping())
    return 0


if __name__ == "__main__":
    sys.exit(main())
