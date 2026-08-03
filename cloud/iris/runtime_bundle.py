"""Build the launcher-revision runtime bundle synchronized into Iris tasks."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

RUNTIME_MODULES = (
    "__init__.py",
    "artifacts.py",
    "hf_datasets.py",
    "ingress_utils.py",
    "literal_proxy_utils.py",
    "model_paths.py",
    "paths.py",
    "ray_storage.py",
    "rl_config_translation.py",
    "rl_data.py",
    "task_runtime.py",
    "training_driver.py",
)


def build_runtime_bundle() -> Path:
    """Copy the per-task runtime without host submission or training source."""
    workspace = Path(tempfile.mkdtemp(prefix="marinskyrl-runtime-bundle-"))
    package_dir = Path(__file__).resolve().parent
    runtime_package = workspace / "cloud" / "iris"
    runtime_package.mkdir(parents=True)
    for module in RUNTIME_MODULES:
        shutil.copy2(package_dir / module, runtime_package / module)
    shutil.copytree(package_dir.parents[1] / "chat_templates", workspace / "chat_templates")
    return workspace
