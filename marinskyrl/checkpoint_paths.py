"""Shared checkpoint and export path contracts."""

from marinskyrl.resource_locator import join_resource_path

GLOBAL_STEP_PREFIX = "global_step_"
LATEST_CHECKPOINT_FILE = "latest_ckpt_global_step.txt"
HF_EXPORT_REQUEST_FILENAME = "hf_export_request.json"
POLICY_CHECKPOINT_SUBDIRECTORY = "policy"
SPECULATOR_CHECKPOINT_SUBDIRECTORY = "speculator"


def policy_export_path(export_root: str, global_step: int) -> str:
    """Return the durable policy export path for one trainer step."""
    return join_resource_path(export_root, f"{GLOBAL_STEP_PREFIX}{global_step}", POLICY_CHECKPOINT_SUBDIRECTORY)


def speculator_export_path(export_root: str, global_step: int) -> str:
    """Return the durable speculator export paired with one policy step."""
    return join_resource_path(export_root, f"{GLOBAL_STEP_PREFIX}{global_step}", SPECULATOR_CHECKPOINT_SUBDIRECTORY)
