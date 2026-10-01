"""Standalone archival dependency exports must not need missing local projects."""

from pathlib import Path
import os
import shutil
import subprocess

from packaging.requirements import Requirement
import pytest


@pytest.mark.parametrize("extra", [None, "verl", "tinker"])
def test_standalone_agent_frozen_export_resolves_without_sibling_projects(tmp_path: Path, extra: str | None) -> None:
    source = Path(__file__).parents[2] / "skyrl-agent"
    for filename in ("pyproject.toml", "uv.lock"):
        shutil.copyfile(source / filename, tmp_path / filename)

    env = {**os.environ, "UV_CACHE_DIR": str(tmp_path / "cache")}
    subprocess.run(["uv", "lock", "--check", "--offline"], cwd=tmp_path, env=env, capture_output=True, check=True)
    command = ["uv", "export", "--frozen", "--offline", "--no-dev", "--no-hashes", "--no-annotate", "--no-emit-project"]
    if extra is not None:
        command.extend(("--extra", extra))
    result = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True, check=True)
    requirements = [Requirement(line) for line in result.stdout.splitlines() if line and not line.startswith("#")]
    names = {requirement.name for requirement in requirements}

    assert "openhands-ai" in names
    assert "skyrl-train" not in names
    assert all(requirement.url is None or not requirement.url.startswith("file:") for requirement in requirements)
    if extra is not None:
        assert extra in names
