from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest


REPOSITORY_ROOT = Path(__file__).parents[2]
RESHARD_LAUNCHER = REPOSITORY_ROOT / "skyrl-train" / "scripts" / "launch_reshard_iris.sh"
SIF_TEST_MARKER = "Jupiter-only SIF test"


def test_reshard_launcher_uses_frozen_runtime_in_standard_task(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    arguments_path = tmp_path / "iris-arguments"
    fake_uv = fake_bin / "uv"
    fake_uv.write_text('#!/bin/sh\nprintf \'%s\\0\' "$@" > "$IRIS_ARGUMENTS_PATH"\n')
    fake_uv.chmod(0o755)
    environment = os.environ | {
        "AWS_ACCESS_KEY_ID": "access-key",
        "AWS_SECRET_ACCESS_KEY": "secret-key",
        "HF_TOKEN": "hf-token",
        "IRIS_ARGUMENTS_PATH": str(arguments_path),
        "MARIN_ROOT": str(tmp_path / "marin"),
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
    }
    (tmp_path / "marin").mkdir()

    result = subprocess.run(
        ["bash", str(RESHARD_LAUNCHER)],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    arguments = arguments_path.read_bytes().split(b"\0")[:-1]
    assert b"--task-image" not in arguments
    assert b"--no-sync" in arguments
    command = arguments[arguments.index(b"--") + 3].decode()
    assert "cloud/iris/bootstrap_runtime.sh" in command
    assert '"$IRIS_VENV/bin/python"' in command
    assert "skyrl-train/scripts/reshard_fsdp2_to_hf.py" in command
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    assert source_commit in command
    subprocess.run(["bash", "-n", "-c", command], check=True)


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
