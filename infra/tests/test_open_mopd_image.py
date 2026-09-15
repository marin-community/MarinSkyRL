"""Contracts for the Open-MOPD authors-runtime image builder."""

import importlib.util
import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).parents[2]
BUILD_SCRIPT = ROOT / "docker" / "build_open_mopd_kaniko.sh"
INSTALLER_PATH = ROOT / "docker" / "install_open_mopd_runtime.py"
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


def test_open_mopd_runtime_manifest_rejects_a_missing_image_override(tmp_path: Path) -> None:
    config = tmp_path / "fidelity.json"
    config.write_text('{"environment":{"packages":{"s3fs":"2025.9.0"}}}')

    with pytest.raises(ValueError, match="runtime manifest is missing packages"):
        INSTALLER.expected_packages(config)


def test_open_mopd_builder_rejects_non_iris_execution(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    apt_marker = tmp_path / "apt-ran"
    (fake_bin / "apt-get").write_text(f"#!/usr/bin/env bash\ntouch {apt_marker}\n")
    (fake_bin / "apt-get").chmod(0o755)
    (fake_bin / "uname").write_text("#!/usr/bin/env bash\necho x86_64\n")
    (fake_bin / "uname").chmod(0o755)

    result = subprocess.run(
        ["bash", str(BUILD_SCRIPT)],
        env={
            "GITSHA": "a" * 40,
            "REGISTRY_USER": "test-user",
            "REGISTRY_TOKEN": "test-token",
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
        },
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 2
    assert not apt_marker.exists()


def test_open_mopd_builder_does_not_trace_credentials_on_guard_failure(tmp_path: Path) -> None:
    credential = "registry-secret-sentinel"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "uname").write_text("#!/usr/bin/env bash\necho x86_64\n")
    (fake_bin / "uname").chmod(0o755)
    result = subprocess.run(
        ["bash", str(BUILD_SCRIPT)],
        env={
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "SHELLOPTS": "braceexpand:hashall:interactive-comments:xtrace",
            "GITSHA": "a" * 40,
            "REGISTRY_USER": "test-user",
            "REGISTRY_TOKEN": credential,
        },
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 2
    assert credential not in result.stderr
