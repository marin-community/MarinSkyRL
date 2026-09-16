"""Publish complete native FSDP2 LoRA checkpoints for independent evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import posixpath

from cloud.iris.artifacts import fs_and_path
from marinskyrl.checkpoint_paths import GLOBAL_STEP_PREFIX, LATEST_CHECKPOINT_FILE
from marinskyrl.resource_locator import join_resource_path

COMMIT_FILENAME = "commit.json"


@dataclass(frozen=True)
class CheckpointFile:
    path: str
    size: int


@dataclass(frozen=True)
class VerifiedCheckpoint:
    step: int
    adapter_uri: str
    commit_sha256: str


def required_checkpoint_files(policy_ranks: int) -> set[str]:
    if policy_ranks <= 0:
        raise ValueError("policy_ranks must be positive")
    required = {
        "data.pt",
        "trainer_state.pt",
        "policy/huggingface/config.json",
        "policy/lora_adapter/adapter_model.safetensors",
        "policy/lora_adapter/adapter_config.json",
    }
    for rank in range(policy_ranks):
        for kind in ("model", "optim", "extra_state"):
            required.add(f"policy/{kind}_world_size_{policy_ranks}_rank_{rank}.pt")
    return required


def checkpoint_inventory(step_root: Path, policy_ranks: int) -> tuple[CheckpointFile, ...]:
    """Validate the loadable FSDP2/LoRA checkpoint shape and list its files."""
    required = required_checkpoint_files(policy_ranks)
    files = {
        path.relative_to(step_root).as_posix(): path
        for path in step_root.rglob("*")
        if path.is_file() and path.name != COMMIT_FILENAME
    }
    missing = required - files.keys()
    if missing:
        raise ValueError(f"Incomplete native checkpoint {step_root}: missing {sorted(missing)}")
    empty = [relative for relative, path in files.items() if path.stat().st_size == 0]
    if empty:
        raise ValueError(f"Incomplete native checkpoint {step_root}: empty {sorted(empty)}")
    return tuple(CheckpointFile(path=relative, size=path.stat().st_size) for relative, path in sorted(files.items()))


def verify_remote_checkpoint(checkpoint_uri: str) -> VerifiedCheckpoint:
    """Reject incomplete or uncommitted remote checkpoints before adapter evaluation."""
    filesystem, target = fs_and_path(checkpoint_uri)
    step_name = posixpath.basename(target.rstrip("/"))
    step_text = step_name.removeprefix(GLOBAL_STEP_PREFIX)
    if not step_name.startswith(GLOBAL_STEP_PREFIX) or not step_text.isdigit():
        raise ValueError(f"Expected a global_step_N checkpoint URI, got {checkpoint_uri}")
    step = int(step_text)
    remote_files = filesystem.find(target, detail=True, withdirs=False)
    remote_sizes = {path: int(details["size"]) for path, details in remote_files.items()}
    commit_path = posixpath.join(target, COMMIT_FILENAME)
    if commit_path not in remote_sizes:
        raise ValueError(f"Remote checkpoint has no commit record: {checkpoint_uri}")
    commit_payload = filesystem.cat_file(commit_path)
    if len(commit_payload) != remote_sizes[commit_path]:
        raise ValueError(f"Remote checkpoint commit record size mismatch: {checkpoint_uri}")
    commit = json.loads(commit_payload)
    if commit.get("schema_version") != 1 or commit.get("step") != step:
        raise ValueError(f"Remote checkpoint commit record does not match step {step}: {checkpoint_uri}")
    policy_ranks = commit.get("policy_ranks")
    if not isinstance(policy_ranks, int) or policy_ranks <= 0:
        raise ValueError(f"Remote checkpoint commit record has invalid policy_ranks: {checkpoint_uri}")
    files = commit.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError(f"Remote checkpoint commit record has no file inventory: {checkpoint_uri}")
    inventory = {}
    for entry in files:
        if not isinstance(entry, dict):
            raise ValueError(f"Remote checkpoint commit record has an invalid file entry: {checkpoint_uri}")
        relative = entry.get("path")
        size = entry.get("size")
        if (
            not isinstance(relative, str)
            or not relative
            or relative.startswith("/")
            or ".." in relative.split("/")
            or not isinstance(size, int)
            or size <= 0
            or relative in inventory
        ):
            raise ValueError(f"Remote checkpoint commit record has an invalid file entry: {checkpoint_uri}")
        inventory[relative] = size
    missing_required = required_checkpoint_files(policy_ranks) - inventory.keys()
    if missing_required:
        raise ValueError(f"Remote checkpoint commit record is missing required files: {sorted(missing_required)}")
    missing_remote = [
        relative
        for relative, size in inventory.items()
        if remote_sizes.get(posixpath.join(target, relative)) != size
    ]
    if missing_remote:
        raise ValueError(f"Remote checkpoint is missing or has changed files: {sorted(missing_remote)}")
    return VerifiedCheckpoint(
        step=step,
        adapter_uri=join_resource_path(checkpoint_uri, "policy/lora_adapter"),
        commit_sha256=hashlib.sha256(commit_payload).hexdigest(),
    )


def publish_committed_checkpoints(checkpoint_root: Path, output_uri: str, policy_ranks: int) -> tuple[int, ...]:
    """Publish committed local steps, verify remote sizes, and advance the remote pointer last."""
    marker = checkpoint_root / LATEST_CHECKPOINT_FILE
    if not marker.is_file():
        return ()
    marker_payload = marker.read_bytes()
    marker_value = marker_payload.decode().strip()
    if not marker_value.isdigit():
        raise ValueError(f"Invalid native checkpoint pointer {marker_value!r}")
    committed_step = int(marker_value)
    filesystem, target = fs_and_path(output_uri)
    remote_files = filesystem.find(target, detail=True, withdirs=False)
    remote_sizes = {path: int(details["size"]) for path, details in remote_files.items()}
    remote_marker = posixpath.join(target, LATEST_CHECKPOINT_FILE)
    if remote_marker in remote_sizes:
        remote_value = filesystem.cat_file(remote_marker).decode().strip()
        if not remote_value.isdigit():
            raise ValueError(f"Invalid remote native checkpoint pointer {remote_value!r}")
        if int(remote_value) > committed_step:
            raise ValueError(f"Remote checkpoint step {remote_value} is newer than local step {committed_step}")

    step_roots = sorted(
        (
            (int(suffix), path)
            for path in checkpoint_root.glob(f"{GLOBAL_STEP_PREFIX}*")
            if path.is_dir()
            if (suffix := path.name.removeprefix(GLOBAL_STEP_PREFIX)).isdigit()
            if int(suffix) <= committed_step
        ),
        key=lambda entry: entry[0],
    )
    published = []
    for step, step_root in step_roots:
        inventory = checkpoint_inventory(step_root, policy_ranks)
        remote_step = posixpath.join(target, step_root.name)
        manifest = json.dumps(
            {
                "schema_version": 1,
                "step": step,
                "policy_ranks": policy_ranks,
                "files": [vars(entry) for entry in inventory],
            },
            sort_keys=True,
        ).encode()
        commit_path = posixpath.join(remote_step, COMMIT_FILENAME)
        already_committed = commit_path in remote_sizes and filesystem.cat_file(commit_path) == manifest
        payload_complete = all(
            remote_sizes.get(posixpath.join(remote_step, entry.path)) == entry.size for entry in inventory
        )
        if already_committed and payload_complete:
            continue
        for entry in inventory:
            local_path = step_root / entry.path
            remote_path = posixpath.join(remote_step, entry.path)
            if remote_sizes.get(remote_path) != entry.size:
                filesystem.makedirs(posixpath.dirname(remote_path), exist_ok=True)
                filesystem.put_file(str(local_path), remote_path)
            observed_size = int(filesystem.info(remote_path)["size"])
            if observed_size != entry.size:
                raise IOError(
                    f"Checkpoint upload size mismatch for {remote_path}: expected {entry.size}, found {observed_size}"
                )
        filesystem.pipe_file(commit_path, manifest)
        if int(filesystem.info(commit_path)["size"]) != len(manifest):
            raise IOError(f"Checkpoint commit was not fully published: {commit_path}")
        published.append(step)

    selected_commit = posixpath.join(target, f"{GLOBAL_STEP_PREFIX}{committed_step}", COMMIT_FILENAME)
    if selected_commit not in remote_sizes and committed_step not in published:
        raise ValueError(f"Native checkpoint step {committed_step} has no committed artifact at {output_uri}")
    filesystem.pipe_file(remote_marker, marker_payload)
    return tuple(published)
