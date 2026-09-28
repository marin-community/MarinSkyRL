from __future__ import annotations

from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).parents[2]
SIF_TEST_MARKER = "Jupiter-only SIF test"


def sif_test_paths() -> tuple[Path, ...]:
    tests_root = REPOSITORY_ROOT / "skyrl-train" / "tests"
    return tuple(
        path
        for path in sorted(tests_root.rglob("*"))
        if path.suffix in {".py", ".sbatch"} and "apptainer exec" in path.read_text()
    )


@pytest.mark.parametrize("path", sif_test_paths(), ids=lambda path: str(path.relative_to(REPOSITORY_ROOT)))
def test_sif_tests_are_marked_as_jupiter_only(path: Path) -> None:
    assert SIF_TEST_MARKER in path.read_text()
