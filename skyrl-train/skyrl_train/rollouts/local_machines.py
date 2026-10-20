"""Shellbox local machines: bubblewrap sandboxes on the rollout worker's own host.

``environment.task_sessions.local`` enables them. ``{}`` takes Shellbox's defaults; ``package_cache``
names the directory where ``PackageSpec`` machines' uv-managed CPython environments are built. The
factory probes bubblewrap when it is constructed and raises ``SandboxUnavailable`` when the host
cannot build a sandbox (on CoreWeave this needs the Iris ``container_profile_privileged`` profile).
The worker never falls back to running commands without a sandbox.
"""

import shutil
import sys
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from loguru import logger
from shellbox.backends.local.python_environment import resolve_python_environment
from shellbox.machine import MachineFactory

LOCAL_SETTINGS = frozenset({"package_cache"})
WARM_THREAD = "warm-local-python"


@runtime_checkable
class PythonEnvironmentFactory(Protocol):
    """A factory that resolves ``PackageSpec`` machines into a cached Python environment, as the local one does."""

    package_cache: Path
    python_version: str


def warm_local_python(factory: MachineFactory) -> threading.Thread | None:
    """Build the bare CPython environment of ``PackageSpec(pypi_packages=())`` before the first machine asks.

    The local factory resolves a ``PackageSpec`` environment once per cache and reuses it; the first wave of
    machines in a fresh process queues on that build. Warming overlaps it with engine startup. A failure is
    logged, and the first machine then builds the environment itself and reports its own error.
    Ported from Taskforge's ``sandbox.factories.warm_local_python``.
    """
    if not isinstance(factory, PythonEnvironmentFactory):
        return None
    cache, version = factory.package_cache, factory.python_version

    def warm() -> None:
        started = time.monotonic()
        try:
            resolve_python_environment(cache, (), version)
        except Exception:
            logger.opt(exception=True).warning("Warming the local Python {} environment in {} failed", version, cache)
            return
        logger.info(
            "Warmed the local Python {} environment in {} in {:.1f}s", version, cache, time.monotonic() - started
        )

    thread = threading.Thread(target=warm, name=WARM_THREAD, daemon=True)
    thread.start()
    return thread


def local_machine_factory(settings: Mapping[str, Any]) -> MachineFactory:
    """Probe the host's bubblewrap sandbox and return Shellbox's local factory, warming its CPython."""
    unknown = set(settings) - LOCAL_SETTINGS
    if unknown:
        raise ValueError(f"Unknown environment.task_sessions.local settings: {sorted(unknown)}")
    if sys.platform != "linux":
        raise ValueError("Local task execution requires a Linux host with working bubblewrap sandboxes")
    if shutil.which("uv") is None:
        # The local factory builds each PackageSpec machine's CPython with uv on the worker's PATH.
        raise ValueError("Local task execution requires uv on the rollout worker's PATH")
    # The local backend's bubblewrap package installs only on Linux.
    from shellbox.backends.local.machine import LocalMachineFactory  # noqa: PLC0415

    package_cache = settings.get("package_cache")
    factory = LocalMachineFactory(package_cache=None if package_cache is None else Path(package_cache))
    logger.info("Local machines sandbox with {}; Python environments in {}", factory.bwrap, factory.package_cache)
    warm_local_python(factory)
    return factory
