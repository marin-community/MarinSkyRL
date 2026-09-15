import importlib.util
from pathlib import Path

import pytest


INSTALLER_PATH = Path(__file__).parents[3] / "docker" / "install_open_mopd_runtime.py"
SPEC = importlib.util.spec_from_file_location("install_open_mopd_runtime", INSTALLER_PATH)
assert SPEC is not None and SPEC.loader is not None
INSTALLER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INSTALLER)


@pytest.mark.parametrize(
    ("expected", "actual", "matches"),
    [
        ("2.8.0", "2.8.0+cu128", True),
        ("2.8.0+cu128", "2.8.0+cu128", True),
        ("2.8.0+cu128", "2.8.0+cu126", False),
        ("2.8.0", "2.8.1+cu128", False),
    ],
)
def test_versions_match_local_build_contract(expected: str, actual: str, matches: bool) -> None:
    assert INSTALLER.versions_match(expected, actual) is matches
