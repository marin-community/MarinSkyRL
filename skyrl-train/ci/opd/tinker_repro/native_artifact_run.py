"""Transactional subprocess execution for native reproduction artifacts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
from typing import TypeVar

from skyrl_train.io.io import upload_directory

Manifest = TypeVar("Manifest")


def run_artifact_command(
    *,
    command: tuple[str, ...],
    initial_manifest: Manifest,
    manifest_path: Path,
    output_root: Path,
    output_uri: str,
    complete_manifest: Callable[[Manifest, int], Manifest],
    failed_manifest: Callable[[Manifest, str], Manifest],
    environment: dict[str, str] | None = None,
) -> int:
    """Run a command while durably recording its initial and terminal manifest."""
    manifest = initial_manifest
    try:
        manifest_path.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n")
        upload_directory(str(output_root), output_uri)
        result = subprocess.run(command, check=False, env=environment)
        manifest = complete_manifest(manifest, result.returncode)
        return result.returncode
    except Exception as error:
        manifest = failed_manifest(manifest, str(error))
        raise
    finally:
        manifest_path.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n")
        upload_directory(str(output_root), output_uri)
