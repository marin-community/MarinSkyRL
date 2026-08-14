"""Storage policy for direct and typed Iris RL launches."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping
from urllib.parse import urlparse

from marinskyrl.resource_locator import join_resource_path

ALLOWED_STORAGE_TTL_DAYS = (1, 2, 3, 4, 5, 6, 7, 14, 30)
DEFAULT_STORAGE_TTL_DAYS = 14
ALLOWED_RESUME_CHECKPOINT_COUNTS = (1, 2, 3, 4, 5)
DEFAULT_RESUME_CHECKPOINT_COUNT = 2

_DURABLE_USER_SEGMENT = "users"
_FORBIDDEN_RUN_SEGMENT = "iris"
_TEMPORARY_SEGMENT = "tmp"
_TTL_SEGMENT_PATTERN = re.compile(r"ttl=\d+d")


@dataclass(frozen=True)
class RLStoragePolicy:
    """Inputs that determine storage for one RL launch."""

    job_name: str
    storage_user: str
    marin_prefix: str
    durable_output_root: str | None
    temporary_output_root: str | None
    storage_ttl_days: int
    resume_checkpoint_count: int
    rendezvous_dir: str | None
    trials_dir: str
    resolved_config_uri: str | None
    overrides: tuple[str, ...]
    config: Mapping[str, object]
    checkpoint_export: bool


@dataclass(frozen=True)
class RLStoragePaths:
    """Resolved storage paths for one RL launch."""

    storage_user: str
    checkpoint_root: str
    export_root: str
    trace_root: str
    trajectory_root: str
    rendezvous_root: str
    resolved_config_uri: str | None
    resume_checkpoint_count: int


def hydra_override_value(overrides: tuple[str, ...] | list[str], key: str) -> str | None:
    """Return the final value assigned to one Hydra key."""
    for override in reversed(overrides):
        assigned_key, separator, value = override.partition("=")
        if separator and assigned_key.strip().lstrip("+~") == key:
            return value.strip().strip("'\"")
    return None


def _config_value(config: Mapping[str, object], *keys: str) -> object | None:
    value: object = config
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _config_path(config: Mapping[str, object], *keys: str) -> str | None:
    value = _config_value(config, *keys)
    if value is None:
        return None
    if not isinstance(value, str):
        raise SystemExit(f"RL storage path {'.'.join(keys)} must be a string, found {value!r}")
    return value


def _cloud_path_segments(path: str) -> tuple[str, ...]:
    parsed = urlparse(path)
    if parsed.scheme not in {"s3", "gs", "gcs"} or not parsed.netloc:
        raise SystemExit(f"RL storage policy requires an s3:// or gs:// path, found {path!r}")
    return tuple(part for part in parsed.path.split("/") if part)


def _validate_temporary_path(label: str, path: str) -> None:
    segments = _cloud_path_segments(path)
    ttl_index = next((index for index, part in enumerate(segments) if part == _TEMPORARY_SEGMENT), None)
    if (
        _FORBIDDEN_RUN_SEGMENT in segments
        or ttl_index is None
        or ttl_index + 1 >= len(segments)
        or not _TTL_SEGMENT_PATTERN.fullmatch(segments[ttl_index + 1])
    ):
        raise SystemExit(
            f"RL storage policy requires {label} under a lifecycle-managed tmp/ttl=Nd prefix, found {path!r}"
        )


def _validate_durable_path(label: str, path: str) -> None:
    segments = _cloud_path_segments(path)
    if _FORBIDDEN_RUN_SEGMENT in segments or _DURABLE_USER_SEGMENT not in segments:
        raise SystemExit(
            f"RL storage policy requires {label} under a user-owned durable users/<username> prefix, found {path!r}"
        )


def _configured_checkpoint_count(policy: RLStoragePolicy) -> int:
    configured = hydra_override_value(policy.overrides, "trainer.max_ckpts_to_keep")
    if configured is None:
        configured = _config_value(policy.config, "trainer", "max_ckpts_to_keep")
    if configured is None:
        return policy.resume_checkpoint_count
    if isinstance(configured, bool):
        raise SystemExit(f"trainer.max_ckpts_to_keep must be an integer, found {configured!r}")
    try:
        return int(configured)
    except (TypeError, ValueError) as error:
        raise SystemExit(f"trainer.max_ckpts_to_keep must be an integer, found {configured!r}") from error


def resolve_storage_paths(policy: RLStoragePolicy) -> RLStoragePaths:
    """Resolve and validate one launch's temporary and durable storage."""
    parsed_prefix = urlparse(policy.marin_prefix)
    bucket_root = f"{parsed_prefix.scheme}://{parsed_prefix.netloc}"
    durable_root = policy.durable_output_root or join_resource_path(
        policy.marin_prefix, _DURABLE_USER_SEGMENT, policy.storage_user, "skyrl", policy.job_name
    )
    temporary_root = policy.temporary_output_root or join_resource_path(
        bucket_root,
        _TEMPORARY_SEGMENT,
        f"ttl={policy.storage_ttl_days}d",
        "skyrl",
        _DURABLE_USER_SEGMENT,
        policy.storage_user,
        policy.job_name,
    )

    checkpoint_root = join_resource_path(temporary_root, "checkpoints")
    export_root = join_resource_path(durable_root, "exports")
    trace_root = join_resource_path(temporary_root, "trace_jobs")
    trajectory_root = join_resource_path(temporary_root, "trajectories")
    resume_checkpoint_count = policy.resume_checkpoint_count

    if not policy.checkpoint_export:
        checkpoint_root = (
            hydra_override_value(policy.overrides, "trainer.ckpt_path")
            or _config_path(policy.config, "trainer", "ckpt_path")
            or checkpoint_root
        )
        export_root = (
            hydra_override_value(policy.overrides, "trainer.export_path")
            or _config_path(policy.config, "trainer", "export_path")
            or export_root
        )
        configured_trials = hydra_override_value(policy.overrides, "terminal_bench_config.trials_dir")
        if policy.trials_dir != "auto":
            configured_trials = policy.trials_dir
        trace_root = (
            configured_trials
            or _config_path(policy.config, "terminal_bench", "trials_dir")
            or _config_path(policy.config, "terminal_bench_config", "trials_dir")
            or trace_root
        )
        trajectory_root = (
            hydra_override_value(policy.overrides, "generator.trajectory_retention.output_path")
            or _config_path(policy.config, "generator", "trajectory_retention", "output_path")
            or trajectory_root
        )
        resume_checkpoint_count = _configured_checkpoint_count(policy)

    rendezvous_root = policy.rendezvous_dir or join_resource_path(temporary_root, "rendezvous")
    resolved_config_uri = policy.resolved_config_uri
    if resolved_config_uri is None and not policy.checkpoint_export:
        resolved_config_uri = join_resource_path(durable_root, "resolved-skyrl.json")

    if resume_checkpoint_count not in ALLOWED_RESUME_CHECKPOINT_COUNTS:
        raise SystemExit(
            "RL storage policy requires trainer.max_ckpts_to_keep to be between one and five; "
            f"found {resume_checkpoint_count}"
        )
    _validate_temporary_path("resume checkpoints", checkpoint_root)
    _validate_temporary_path("raw traces", trace_root)
    _validate_temporary_path("retained trajectories", trajectory_root)
    _validate_temporary_path("rendezvous and Ray session data", rendezvous_root)
    _validate_durable_path("canonical exports", export_root)
    if resolved_config_uri is not None:
        _validate_durable_path("resolved launch configuration", resolved_config_uri)

    return RLStoragePaths(
        storage_user=policy.storage_user,
        checkpoint_root=checkpoint_root,
        export_root=export_root,
        trace_root=trace_root,
        trajectory_root=trajectory_root,
        rendezvous_root=rendezvous_root,
        resolved_config_uri=resolved_config_uri,
        resume_checkpoint_count=resume_checkpoint_count,
    )
