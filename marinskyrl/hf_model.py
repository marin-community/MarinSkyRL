"""Shared Hugging Face model export contracts."""

import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    """Return a file's SHA-256 digest without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_portable_hf_model_files(names: set[str], source: str) -> None:
    """Validate the minimum portable Hugging Face model export contract."""
    if "config.json" not in names:
        raise ValueError(f"Model export is missing config.json: {source}")
    if not any(name.endswith((".safetensors", ".bin")) for name in names):
        raise ValueError(f"Model export has no weight shards: {source}")
    if not any(name.startswith("tokenizer") or name.endswith(".model") for name in names):
        raise ValueError(f"Model export has no tokenizer files: {source}")


def validate_speculator_model_files(names: set[str], source: str) -> None:
    """Validate an EAGLE draft export, whose target owns the tokenizer and embedding."""
    if "config.json" not in names:
        raise ValueError(f"Speculator export is missing config.json: {source}")
    if not any(name.endswith((".safetensors", ".bin")) for name in names):
        raise ValueError(f"Speculator export has no weight shards: {source}")
