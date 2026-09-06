"""Regression tests for the runtime bundle file manifest."""

from __future__ import annotations

import sys
import shutil
import subprocess
from pathlib import Path

import pytest

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from cloud.iris.runtime_bundle import BUNDLE_FILE_MANIFEST, read_manifest_paths  # noqa: E402
from cloud.iris import runtime_bundle  # noqa: E402


def test_every_marinskyrl_module_is_in_the_runtime_bundle() -> None:
    """Every .py file under marinskyrl/ must be listed in the bundle manifest.

    PR #357 added marinskyrl/hf_model.py and marinskyrl/checkpoint_paths.py without
    updating the manifest, causing ModuleNotFoundError at runtime.  This test forces
    a deliberate choice when adding a new module: list it in the manifest or add an
    explicit exclusion in the test.
    """
    manifest = set(read_manifest_paths(_REPOSITORY_ROOT))
    on_disk = {p.relative_to(_REPOSITORY_ROOT).as_posix() for p in (_REPOSITORY_ROOT / "marinskyrl").rglob("*.py")}
    missing = on_disk - manifest
    assert not missing, (
        f"modules under marinskyrl/ not shipped to Iris tasks: {sorted(missing)}. "
        f"Add them to {BUNDLE_FILE_MANIFEST} or add an explicit exclusion in this test."
    )


def test_installed_wheel_and_staged_bundle_import_inference_placement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise both installed imports and the /app package that shadows them.

    The Iris workspace contains marinskyrl/__init__.py from the launcher bundle.
    A module missing there cannot fall back to the complete installed wheel.
    Neither subprocess can import modules from the source checkout.
    """
    wheel_dir = tmp_path / "wheels"
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(wheel_dir)],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    (wheel,) = wheel_dir.glob("*.whl")
    environment = tmp_path / "runtime"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(environment)], check=True)
    python = environment / "bin" / "python"
    subprocess.run(
        ["uv", "pip", "install", "--python", str(python), "--no-deps", str(wheel)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    program = """
from pathlib import Path
import marinskyrl.inference_placement as placement
placement.validate_node_local_config({"generator": {"inference_engine_node_local": False}})
print(Path(placement.__file__).resolve())
"""
    installed = subprocess.run(
        [str(python), "-I", "-c", program], cwd=tmp_path, check=True, capture_output=True, text=True
    )
    installed_module = Path(installed.stdout.strip())
    assert installed_module.is_relative_to(environment)
    source = runtime_bundle.LauncherSource(
        root=installed_module.parents[1],
        commit="installed-wheel-smoke",
        kind=runtime_bundle.LauncherSourceKind.INSTALLED,
    )
    monkeypatch.setattr(runtime_bundle, "resolve_launcher_source", lambda: source)
    bundle = Path(shutil.move(runtime_bundle.build_runtime_bundle(source.commit), tmp_path / "app"))
    assert runtime_bundle.validate_bundled_runtime(bundle) == source.commit
    staged = subprocess.run(
        [str(python), "-E", "-s", "-c", program], cwd=bundle, check=True, capture_output=True, text=True
    )
    assert Path(staged.stdout.strip()) == bundle / "marinskyrl" / "inference_placement.py"
