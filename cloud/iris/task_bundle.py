"""Build the minimal workspace synchronized into a MarinSkyRL task image."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path


def build_task_bundle() -> Path:
    """Copy packaged Iris controller code into an isolated Iris workspace."""
    workspace = Path(tempfile.mkdtemp(prefix="marinskyrl-task-bundle-"))
    package_dir = Path(__file__).resolve().parent
    shutil.copytree(
        package_dir,
        workspace / "cloud" / "iris",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "tests"),
    )
    return workspace
