"""Regression tests for the runtime bundle file manifest."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from cloud.iris.runtime_bundle import (  # noqa: E402
    BUNDLE_FILE_MANIFEST,
    LauncherSource,
    LauncherSourceKind,
    build_runtime_bundle,
    read_manifest_paths,
)


def test_every_runtime_module_is_in_the_runtime_bundle() -> None:
    """Ship both Python packages imported by the task runtime."""
    manifest = set(read_manifest_paths(_REPOSITORY_ROOT))
    on_disk = {
        path.relative_to(_REPOSITORY_ROOT).as_posix()
        for package in ("cloud/iris", "marinskyrl")
        for path in (_REPOSITORY_ROOT / package).glob("*.py")
    }
    missing = on_disk - manifest
    assert not missing, (
        f"runtime modules not shipped to Iris tasks: {sorted(missing)}. "
        f"Add them to {BUNDLE_FILE_MANIFEST} or add an explicit exclusion in this test."
    )


def test_controller_bundle_imports_the_task_factory_before_the_checkout(tmp_path, monkeypatch) -> None:
    # Only source-location metadata is supplied at the installation boundary;
    # copying, file identity, package shadowing, and subprocess imports are real.
    source = LauncherSource(root=_REPOSITORY_ROOT, commit="a" * 40, kind=LauncherSourceKind.INSTALLED)
    monkeypatch.setattr("cloud.iris.runtime_bundle.resolve_launcher_source", lambda: source)
    bundle = build_runtime_bundle(source.commit)
    environment = dict(os.environ)
    inherited = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = f"{bundle}:{_REPOSITORY_ROOT}:{inherited}"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json; from pathlib import Path; from omegaconf import OmegaConf; "
            "import marinskyrl.task_machine_factories as module; "
            "factories = module.task_machine_factories(OmegaConf.create({})); "
            "print(json.dumps({'module': str(Path(module.__file__).resolve()), 'backends': sorted(factories)}))",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    observed = json.loads(result.stdout)
    assert observed["module"] == str((bundle / "marinskyrl/task_machine_factories.py").resolve())
    assert "shellsim" in observed["backends"]
    assert "local" not in observed["backends"]
