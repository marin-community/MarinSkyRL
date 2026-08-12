"""Shared checkpoint and export path contracts."""

from marinskyrl.resource_locator import join_resource_path

GLOBAL_STEP_PREFIX = "global_step_"
POLICY_CHECKPOINT_SUBDIRECTORY = "policy"


def policy_export_path(export_root: str, global_step: int) -> str:
    """Return the durable policy export path for one trainer step."""
    return join_resource_path(export_root, f"{GLOBAL_STEP_PREFIX}{global_step}", POLICY_CHECKPOINT_SUBDIRECTORY)
