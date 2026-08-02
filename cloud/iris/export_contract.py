"""Shared path and file contracts for staged and terminal model exports."""

from __future__ import annotations

import posixpath

CHECKPOINT_MARKER_FILENAME = "latest_ckpt_global_step.txt"
SOURCE_MANIFEST_FILENAME = ".marinskyrl-source.json"


def relative_object_key(root: str, path: str) -> str:
    """Return ``path`` below ``root`` or reject a non-descendant object key."""
    normalized_root = posixpath.normpath(root)
    normalized_path = posixpath.normpath(path)
    relative = posixpath.relpath(normalized_path, normalized_root)
    if relative == ".." or relative.startswith("../") or posixpath.isabs(relative):
        raise ValueError(f"Object key {path!r} is not below source root {root!r}")
    return relative


def validate_hf_export(names: set[str], source: str) -> None:
    """Validate the minimum portable Hugging Face model export contract."""
    if "config.json" not in names:
        raise ValueError(f"Model export is missing config.json: {source}")
    if not any(name.endswith((".safetensors", ".bin")) for name in names):
        raise ValueError(f"Model export has no weight shards: {source}")
    if not any(name.startswith("tokenizer") or name.endswith(".model") for name in names):
        raise ValueError(f"Model export has no tokenizer files: {source}")
