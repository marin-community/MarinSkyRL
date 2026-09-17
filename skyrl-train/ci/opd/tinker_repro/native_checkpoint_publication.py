"""Publish and restore committed native FSDP2 LoRA checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import posixpath
import shutil

from cloud.iris.artifacts import FileEntry, copy_tree, fs_and_path
from marinskyrl.checkpoint_paths import GLOBAL_STEP_PREFIX, LATEST_CHECKPOINT_FILE
from marinskyrl.resource_locator import join_resource_path
from skyrl_train.hf_export import protected_hf_export_steps

COMMIT_FILENAME = "commit.json"
PRUNING_PREFIX = ".pruning-"


@dataclass(frozen=True)
class VerifiedCheckpoint:
    step: int
    adapter_uri: str
    commit_sha256: str
    policy_ranks: int


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


def checkpoint_inventory(step_root: Path, policy_ranks: int) -> tuple[FileEntry, ...]:
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
    return tuple(FileEntry(path=relative, size=path.stat().st_size) for relative, path in sorted(files.items()))


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
        relative for relative, size in inventory.items() if remote_sizes.get(posixpath.join(target, relative)) != size
    ]
    if missing_remote:
        raise ValueError(f"Remote checkpoint is missing or has changed files: {sorted(missing_remote)}")
    return VerifiedCheckpoint(
        step=step,
        adapter_uri=join_resource_path(checkpoint_uri, "policy/lora_adapter"),
        commit_sha256=hashlib.sha256(commit_payload).hexdigest(),
        policy_ranks=policy_ranks,
    )


def restore_latest_committed_checkpoint(
    checkpoint_uri: str, checkpoint_root: Path, policy_ranks: int
) -> VerifiedCheckpoint:
    """Stage the selected remote checkpoint and its pointer for trainer resume."""
    marker_uri = join_resource_path(checkpoint_uri, LATEST_CHECKPOINT_FILE)
    filesystem, marker_path = fs_and_path(marker_uri)
    marker_payload = filesystem.cat_file(marker_path)
    marker_value = marker_payload.decode().strip()
    if not marker_value.isdigit() or int(marker_value) <= 0:
        raise ValueError(f"Invalid native checkpoint pointer {marker_value!r}")
    step = int(marker_value)
    step_uri = join_resource_path(checkpoint_uri, f"{GLOBAL_STEP_PREFIX}{step}")
    verified = verify_remote_checkpoint(step_uri)
    if verified.policy_ranks != policy_ranks:
        raise ValueError(f"Native checkpoint has {verified.policy_ranks} policy ranks; expected {policy_ranks}")

    checkpoint_root.mkdir(parents=True, exist_ok=False)
    step_root = checkpoint_root / f"{GLOBAL_STEP_PREFIX}{step}"
    copy_tree(step_uri, step_root)
    commit_payload = (step_root / COMMIT_FILENAME).read_bytes()
    if hashlib.sha256(commit_payload).hexdigest() != verified.commit_sha256:
        raise ValueError("Staged native checkpoint commit differs from the verified remote commit")
    commit = json.loads(commit_payload)
    staged_files = {entry.path: entry.size for entry in checkpoint_inventory(step_root, policy_ranks)}
    committed_files = {entry["path"]: entry["size"] for entry in commit["files"]}
    if staged_files != committed_files:
        raise ValueError("Staged native checkpoint files differ from the remote commit inventory")
    if filesystem.cat_file(marker_path) != marker_payload:
        raise ValueError("Native checkpoint pointer changed during staging")
    (checkpoint_root / LATEST_CHECKPOINT_FILE).write_bytes(marker_payload)
    return verified


def prune_verified_local_checkpoints(
    checkpoint_root: Path, output_uri: str, step_roots: list[tuple[int, Path]], retain_local_checkpoints: int
) -> None:
    """Keep recent and pending-export steps after verifying remote copies."""
    if retain_local_checkpoints < 1:
        raise ValueError("retain_local_checkpoints must be positive")
    # Rename before removing so an interrupted local deletion cannot leave a
    # partial global_step_N directory that blocks the next publication pass.
    for staged in checkpoint_root.glob(f"{PRUNING_PREFIX}{GLOBAL_STEP_PREFIX}*"):
        suffix = staged.name.removeprefix(f"{PRUNING_PREFIX}{GLOBAL_STEP_PREFIX}")
        if staged.is_dir() and suffix.isdigit():
            verify_remote_checkpoint(join_resource_path(output_uri, f"{GLOBAL_STEP_PREFIX}{suffix}"))
            shutil.rmtree(staged)
    protected_steps = protected_hf_export_steps(str(checkpoint_root))
    for step, step_root in step_roots[:-retain_local_checkpoints]:
        if step in protected_steps:
            continue
        verify_remote_checkpoint(join_resource_path(output_uri, step_root.name))
        staged = step_root.with_name(f"{PRUNING_PREFIX}{step_root.name}")
        step_root.rename(staged)
        shutil.rmtree(staged)


def publish_committed_checkpoints(
    checkpoint_root: Path, output_uri: str, policy_ranks: int, retain_local_checkpoints: int
) -> tuple[int, ...]:
    """Publish committed steps and advance the pointer before local retention."""
    if retain_local_checkpoints < 1:
        raise ValueError("retain_local_checkpoints must be positive")
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

    prune_verified_local_checkpoints(checkpoint_root, output_uri, step_roots, retain_local_checkpoints)
    return tuple(published)
