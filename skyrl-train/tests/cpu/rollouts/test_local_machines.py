"""Shellbox local machines in the rollout worker."""

import shutil
import sys
import threading
from pathlib import Path

import pytest
from shellbox.machine import Backend, Command, MachineSpec, NetworkPolicy, PackageSpec

from skyrl_train.config.utils import get_default_config
from skyrl_train.rollouts import local_machines, task_worker
from skyrl_train.rollouts.task_worker import TaskRolloutWorkerSpec
from skyrl_train.rollouts.workers import WorkerShard
from tests.cpu.rollouts.engine_fakes import Tokenizer


def test_local_machines_refuse_hosts_without_linux_sandboxes(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    with pytest.raises(ValueError, match="Linux host with working bubblewrap"):
        local_machines.local_machine_factory({})


def test_local_machines_refuse_unknown_settings():
    with pytest.raises(ValueError, match="Unknown environment.task_sessions.local settings"):
        local_machines.local_machine_factory({"memory_mb": 1024})


@pytest.mark.asyncio
@pytest.mark.parametrize("local", [None, {}, {"package_cache": "/cache/python"}])
async def test_worker_registers_local_machines_only_when_configured(monkeypatch, local):
    received = []

    class Factory:
        backend = Backend.LOCAL

    def factory(settings):
        received.append(settings)
        return Factory()

    monkeypatch.setattr(task_worker, "local_machine_factory", factory)
    config = get_default_config()
    config.environment.task_sessions.local = local
    worker = TaskRolloutWorkerSpec.from_config(config, []).build(Tokenizer(), WorkerShard(0, 1))
    try:
        assert (Backend.LOCAL in worker.factories) is (local is not None)
        assert received == ([] if local is None else [local])
    finally:
        await worker.shutdown()


def test_warming_builds_the_bare_cpython_environment_in_the_factory_cache(monkeypatch, tmp_path: Path):
    built = []
    monkeypatch.setattr(local_machines, "resolve_python_environment", lambda *args: built.append(args))

    class Factory:
        package_cache = tmp_path
        python_version = "3.12.12"

    thread = local_machines.warm_local_python(Factory())

    assert isinstance(thread, threading.Thread)
    thread.join(timeout=10)
    assert built == [(tmp_path, (), "3.12.12")]
    assert local_machines.warm_local_python(object()) is None


def _bubblewrap_works() -> bool:
    if sys.platform != "linux" or shutil.which("uv") is None:
        return False
    from shellbox.backends.local.machine import LocalMachineFactory, SandboxUnavailable  # noqa: PLC0415

    try:
        LocalMachineFactory()
    except SandboxUnavailable:
        return False
    return True


@pytest.mark.asyncio
@pytest.mark.skipif(not _bubblewrap_works(), reason="needs Linux with working bubblewrap sandboxes and uv")
async def test_local_machine_runs_the_grader_base_python(tmp_path: Path):
    factory = local_machines.local_machine_factory({"package_cache": str(tmp_path)})
    machine = await factory.create(
        MachineSpec(
            source=PackageSpec(pypi_packages=()), workdir="/app", network=NetworkPolicy.DENY, startup_timeout=600
        )
    )
    try:
        result = await machine.run(
            Command(("python3", "-c", "import sys, json; print(json.dumps(sys.version_info[:2]))"), timeout=60)
        )
    finally:
        await machine.close()
    assert result.exit_code == 0
    assert result.stdout.decode().strip() == "[3, 12]"
