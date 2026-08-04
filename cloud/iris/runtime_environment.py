"""Resolve and install the immutable MarinSkyRL task environment."""

from __future__ import annotations

import importlib.metadata
import json
import shlex
import subprocess
from enum import StrEnum
from pathlib import Path
from urllib.parse import unquote, urlparse


MARINSKYRL_REPOSITORY = "https://github.com/marin-community/MarinSkyRL.git"
MARINSKYRL_TASK_ROOT = "/app/marinskyrl"
MARINSKYRL_RUNTIME_ENV = f"{MARINSKYRL_TASK_ROOT}/.iris-runtime-env"
MARINSKYRL_BOOTSTRAP_SCRIPT = "cloud/iris/bootstrap_runtime.sh"


class RuntimeProfile(StrEnum):
    """Locked dependency set installed for one training strategy."""

    FSDP = "fsdp"
    MEGATRON = "megatron"


def installed_commit() -> str:
    """Return the exact VCS revision of the installed launcher distribution."""
    try:
        direct_url = importlib.metadata.distribution("marinskyrl").read_text("direct_url.json")
    except importlib.metadata.PackageNotFoundError:
        direct_url = None
    if direct_url:
        direct_url_value = json.loads(direct_url)
        commit = direct_url_value.get("vcs_info", {}).get("commit_id")
        if commit:
            return str(commit)
        parsed_url = urlparse(direct_url_value.get("url", ""))
        if parsed_url.scheme == "file":
            checkout = Path(unquote(parsed_url.path))
            if (checkout / ".git").exists() and (checkout / "pyproject.toml").exists():
                return _checkout_commit(checkout)

    repository_root = Path(__file__).resolve().parents[2]
    if not (repository_root / ".git").exists() or not (repository_root / "pyproject.toml").exists():
        raise RuntimeError("Installed marinskyrl wheel has no VCS commit identity in direct_url.json")
    return _checkout_commit(repository_root)


def _checkout_commit(checkout: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def task_setup_script(commit: str, profile: RuntimeProfile) -> str:
    """Build the Iris setup script for a frozen SkyRL checkout and dependency profile."""
    checkout = MARINSKYRL_TASK_ROOT
    runtime_env = MARINSKYRL_RUNTIME_ENV
    bootstrap_script = f"{checkout}/{MARINSKYRL_BOOTSTRAP_SCRIPT}"
    return f"""set -euo pipefail
checkout={shlex.quote(checkout)}
git init -q "$checkout"
git -C "$checkout" remote add origin {shlex.quote(MARINSKYRL_REPOSITORY)}
git -C "$checkout" fetch --depth=1 origin {shlex.quote(commit)}
git -C "$checkout" checkout -q --detach FETCH_HEAD
test "$(git -C "$checkout" rev-parse HEAD)" = {shlex.quote(commit)}
bash {shlex.quote(bootstrap_script)} \
  "$checkout" \
  "$IRIS_VENV" \
  {shlex.quote(runtime_env)} \
  {shlex.quote(profile.value)} \
  production
source {shlex.quote(runtime_env)}
"""
