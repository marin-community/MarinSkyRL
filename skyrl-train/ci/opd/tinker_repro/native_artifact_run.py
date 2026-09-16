"""Transactional subprocess execution for native reproduction artifacts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
import json
import logging
from pathlib import Path
import subprocess
from threading import Event, Thread
from typing import TypeVar

from marinskyrl.resource_locator import join_resource_path
from skyrl_train.io.io import upload_directory, upload_file

Manifest = TypeVar("Manifest")
logger = logging.getLogger(__name__)
CHECKPOINT_PUBLICATION_INTERVAL = 30


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
    publish_checkpoints: Callable[[], tuple[int, ...]] | None = None,
) -> int:
    """Run a command while durably recording its initial and terminal manifest."""
    manifest = initial_manifest
    publisher_stop = Event()
    publisher: Thread | None = None

    def publish_periodically() -> None:
        assert publish_checkpoints is not None
        while not publisher_stop.wait(CHECKPOINT_PUBLICATION_INTERVAL):
            try:
                publish_checkpoints()
            except Exception as error:
                logger.warning("Native checkpoint publication failed; will retry: %s", error, exc_info=True)

    try:
        manifest_path.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n")
        if publish_checkpoints is None:
            upload_directory(str(output_root), output_uri)
        else:
            upload_file(str(manifest_path), join_resource_path(output_uri, manifest_path.name))
        if publish_checkpoints is not None:
            publisher = Thread(target=publish_periodically, name="native-checkpoint-publication", daemon=True)
            publisher.start()
        result = subprocess.run(command, check=False, env=environment)
        manifest = complete_manifest(manifest, result.returncode)
        return result.returncode
    except Exception as error:
        manifest = failed_manifest(manifest, str(error))
        raise
    finally:
        publisher_stop.set()
        if publisher is not None:
            publisher.join()
        publication_error = None
        if publish_checkpoints is not None:
            try:
                publish_checkpoints()
            except Exception as error:
                publication_error = error
                manifest = failed_manifest(manifest, f"Checkpoint publication failed: {error}")
        manifest_path.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n")
        if publish_checkpoints is None:
            upload_directory(str(output_root), output_uri)
        else:
            upload_file(str(manifest_path), join_resource_path(output_uri, manifest_path.name))
            export_dir = output_root / "exports"
            if export_dir.is_dir():
                upload_directory(str(export_dir), join_resource_path(output_uri, "exports"))
        if publication_error is not None:
            raise publication_error
