"""Regression tests for the runtime bundle file manifest."""

from __future__ import annotations

import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from cloud.iris.runtime_bundle import BUNDLE_FILE_MANIFEST, read_manifest_paths  # noqa: E402


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
