"""Shared checkpoint and export path contracts."""

import re

from marinskyrl.resource_locator import join_resource_path

GLOBAL_STEP_PREFIX = "global_step_"
LATEST_CHECKPOINT_FILE = "latest_ckpt_global_step.txt"
HF_EXPORT_REQUEST_FILENAME = "hf_export_request.json"
POLICY_CHECKPOINT_SUBDIRECTORY = "policy"


def extract_step_from_path(path: str) -> int:
    """Find a global step in a checkpoint path, including attempt payloads."""
    for part in reversed(path.rstrip("/").split("/")):
        match = re.fullmatch(rf"{re.escape(GLOBAL_STEP_PREFIX)}(\d+)", part)
        if match:
            return int(match.group(1))
    return -1


def policy_export_path(export_root: str, global_step: int) -> str:
    """Return the durable policy export path for one trainer step."""
    return join_resource_path(export_root, f"{GLOBAL_STEP_PREFIX}{global_step}", POLICY_CHECKPOINT_SUBDIRECTORY)
