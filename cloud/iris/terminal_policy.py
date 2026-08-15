"""Terminal policy export orchestration for Iris RL launches."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from typing import Mapping
from urllib.parse import urlparse

from cloud.iris.artifacts import fs_and_path, terminal_checkpoint_step
from cloud.iris.model_paths import model_source_cli_args
from cloud.iris.paths import PROJECT_ROOT
from cloud.iris.storage_policy import hydra_override_value
from marinskyrl.checkpoint_paths import GLOBAL_STEP_PREFIX, HF_EXPORT_REQUEST_FILENAME
from marinskyrl.resource_locator import join_resource_path


@dataclass(frozen=True)
class TerminalPolicyExport:
    """Inputs needed to export one terminal policy checkpoint."""

    checkpoint_root: str
    export_root: str
    config_path: str
    model_path: str
    model_source_uri: str | None
    model_source_identity: str | None
    policy_num_nodes: int
    policy_num_gpus_per_node: int
    cluster: str
    priority: str
    job_name: str
    cluster_config: str | None = None
    target_cluster: str | None = None
    parent_cluster_config: str | None = None
    cpu: float | None = None
    memory: str | None = None
    disk: str | None = None
    storage_user: str | None = None


def storage_user_from_resource_path(path: str) -> str | None:
    """Return the user segment from a canonical storage-policy path."""
    parts = tuple(part for part in urlparse(path).path.split("/") if part)
    for index, part in enumerate(parts[:-1]):
        if part == "users":
            return parts[index + 1]
    return None


def policy_export_geometry(
    config: Mapping[str, object],
    overrides: tuple[str, ...] | list[str],
    *,
    default_num_nodes: int,
    default_gpus_per_node: int,
) -> tuple[int, int]:
    """Return the policy worker geometry required to restore a checkpoint."""
    trainer = config.get("trainer")
    placement = trainer.get("placement") if isinstance(trainer, dict) else None
    configured_nodes = placement.get("policy_num_nodes") if isinstance(placement, dict) else None
    configured_gpus = placement.get("policy_num_gpus_per_node") if isinstance(placement, dict) else None
    nodes = hydra_override_value(overrides, "trainer.placement.policy_num_nodes") or configured_nodes
    gpus = hydra_override_value(overrides, "trainer.placement.policy_num_gpus_per_node") or configured_gpus
    nodes_text = str(nodes)
    gpus_text = str(gpus)
    return (
        int(nodes_text) if nodes_text.isdigit() and int(nodes_text) > 0 else default_num_nodes,
        int(gpus_text) if gpus_text.isdigit() and int(gpus_text) > 0 else default_gpus_per_node,
    )


def submit_terminal_policy_export(spec: TerminalPolicyExport) -> None:
    """Submit and verify conversion of the latest committed checkpoint."""
    global_step = terminal_checkpoint_step(spec.checkpoint_root)
    checkpoint_path = join_resource_path(spec.checkpoint_root, f"{GLOBAL_STEP_PREFIX}{global_step}")
    command = [
        sys.executable,
        "-m",
        "cloud.iris.export_hf_checkpoint",
        "--rl_config",
        spec.config_path,
        "--cluster",
        spec.cluster,
        "--priority",
        spec.priority,
        "--job-name",
        f"{spec.job_name}-export-{global_step}",
    ]
    if spec.cluster_config:
        command.extend(["--cluster-config", spec.cluster_config])
    if spec.target_cluster:
        command.extend(["--target-cluster", spec.target_cluster])
    if spec.parent_cluster_config:
        command.extend(["--parent-cluster-config", spec.parent_cluster_config])
    if spec.cpu is not None:
        command.extend(["--cpu", str(spec.cpu)])
    if spec.memory:
        command.extend(["--memory", spec.memory])
    if spec.disk:
        command.extend(["--disk", spec.disk])
    if spec.storage_user:
        command.extend(["--storage-user", spec.storage_user])
    export_request_uri = join_resource_path(checkpoint_path, HF_EXPORT_REQUEST_FILENAME)
    export_filesystem, export_request_path = fs_and_path(export_request_uri)
    if export_filesystem.exists(export_request_path):
        command.extend(["--request", checkpoint_path])
    else:
        command.extend(
            [
                "--ckpt_path",
                spec.checkpoint_root,
                "--step",
                str(global_step),
                "--model_path",
                spec.model_path,
                "--num-nodes",
                str(spec.policy_num_nodes),
                "--gpus-per-node",
                str(spec.policy_num_gpus_per_node),
                "--export_path",
                spec.export_root,
            ]
        )
        command.extend(model_source_cli_args(spec.model_source_uri, spec.model_source_identity))
    exit_code = subprocess.call(command, cwd=str(PROJECT_ROOT))
    if exit_code != 0:
        raise subprocess.CalledProcessError(exit_code, command)
