"""Resolve and install the immutable MarinSkyRL task environment."""

from __future__ import annotations

import shlex
from enum import StrEnum


MARINSKYRL_REPOSITORY = "https://github.com/marin-community/MarinSkyRL.git"
MARINSKYRL_TASK_ROOT = "/app/marinskyrl"
MARINSKYRL_ACTIVATION_FILE = f"{MARINSKYRL_TASK_ROOT}/.iris-runtime-env"
MARINSKYRL_BOOTSTRAP_SCRIPT = "cloud/iris/bootstrap_runtime.sh"


class RuntimeProfile(StrEnum):
    """Locked dependency set installed for one training strategy."""

    FSDP = "fsdp"
    MEGATRON = "megatron"


def task_setup_script(commit: str, profile: RuntimeProfile) -> str:
    """Build the Iris setup script for a frozen SkyRL checkout and dependency profile."""
    checkout = MARINSKYRL_TASK_ROOT
    activation_file = MARINSKYRL_ACTIVATION_FILE
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
  {shlex.quote(activation_file)} \
  {shlex.quote(profile.value)} \
  production
source {shlex.quote(activation_file)}
"""
