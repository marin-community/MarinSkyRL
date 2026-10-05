"""Check the image-backed task factory's execution limits."""

import asyncio
import hashlib
import json
import tarfile
from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf
from shellbox.backends.qemu.bundle import guest_code_id
from shellbox.image import RegistryImage
from shellbox.machine import MachineSpec, NetworkPolicy, UnsupportedMachineSpec

from skyrl_train.rollouts.task_machines import task_machine_factory


def qemu_config(root: Path) -> DictConfig:
    config = OmegaConf.load(Path(__file__).parents[2] / "skyrl_train/config/ppo_base_config.yaml").trajectory_runner
    config.machine.backend = "qemu"
    config.machine.qemu = {
        "acceleration": "tcg",
        "bundle_cache": str(root / "bundles"),
        "assets": {
            "qemu": str(root / "qemu"),
            "kernel": str(root / "kernel"),
            "busybox": str(root / "busybox"),
            "firmware": str(root / "firmware"),
            "libraries": str(root / "libraries"),
            "umoci": str(root / "umoci"),
            "disk_size_mb": 256,
            "runtime_id": "test-runtime",
        },
    }
    return config


def test_qemu_tasks_reject_network_access_before_image_preparation(tmp_path):
    factory = task_machine_factory(qemu_config(tmp_path))
    spec = MachineSpec(source=RegistryImage("localhost/unavailable:test"), network=NetworkPolicy.ALLOW)
    with pytest.raises(UnsupportedMachineSpec, match="networking is unsupported"):
        asyncio.run(factory.create(spec))


def test_bundle_factory_uses_verified_prepared_registry_image(tmp_path):
    source = tmp_path / "source" / "test-runtime"
    source.mkdir(parents=True)
    kernel = source / "kernel"
    kernel.write_bytes(b"test-kernel")
    reference = "localhost/prepared@sha256:" + "a" * 64
    image = source / "image.json"
    image.write_text(
        json.dumps(
            {"image_reference": reference, "manifest_digest": "sha256:" + "a" * 64, "guest_code_id": guest_code_id()}
        )
    )
    archive = tmp_path / "runtime.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        output.add(source, arcname=source.name)
    parent = tmp_path / "installed"
    installed = parent / source.name
    manifest = {
        "directory_name": source.name,
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "files": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (kernel, image)},
        "host_packages": {},
        "tools": {},
        "source_image": reference,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    config = qemu_config(tmp_path)
    config.machine.qemu.assets = None
    config.machine.runtime_bundle = {
        "manifest_uri": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "archive_uri": str(archive),
        "archive_sha256": manifest["archive_sha256"],
        "installation_parent": str(parent),
    }
    factory = task_machine_factory(config)
    assert factory.prepared_registry_bundles == {manifest["source_image"]: installed}
    assert (installed / "kernel").read_bytes() == b"test-kernel"
