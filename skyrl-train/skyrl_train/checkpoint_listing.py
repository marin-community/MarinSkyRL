"""Torch-free helpers for enumerating checkpoint directories; safe to import from launcher environments."""

import os

from loguru import logger
from marinskyrl.checkpoint_paths import GLOBAL_STEP_PREFIX, extract_step_from_path as extract_step_from_path

from skyrl_train.checkpoint_generation import resolve_checkpoint_payload
from skyrl_train.io import io


def list_checkpoint_dirs(checkpoint_base_path: str) -> list[str]:
    """
    List all checkpoint directories in the base path.

    Args:
        checkpoint_base_path: Base path where checkpoints are stored

    Returns:
        list[str]: List of checkpoint directory names
    """
    if not io.exists(checkpoint_base_path):
        return []

    try:
        all_items = io.list_dir(checkpoint_base_path)

        # Filter for directories that match the global_step_* pattern
        checkpoint_dirs = []
        for item in all_items:
            # Get just the basename for pattern matching
            basename = os.path.basename(item)
            if basename.startswith(GLOBAL_STEP_PREFIX) and io.isdir(os.path.join(checkpoint_base_path, basename)):
                checkpoint_dirs.append(basename)

        return sorted(checkpoint_dirs)
    except Exception as e:
        logger.warning(f"Failed to list checkpoint directories from {checkpoint_base_path}: {e}")
        return []


def list_committed_checkpoint_dirs(checkpoint_base_path: str) -> list[str]:
    """Exclude incomplete or invalid generations from retention candidates.

    A bad commit record must not make us delete an older valid checkpoint.
    Direct resume still resolves the advertised pointer and raises on corruption.
    """
    committed = []
    for directory in list_checkpoint_dirs(checkpoint_base_path):
        try:
            resolve_checkpoint_payload(os.path.join(checkpoint_base_path, directory))
        except (FileNotFoundError, OSError, ValueError, KeyError, TypeError) as error:
            logger.warning(f"Ignoring uncommitted checkpoint directory {directory}: {error}")
        else:
            committed.append(directory)
    return committed
