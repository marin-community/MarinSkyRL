"""Select the Shellbox machine backend for image-backed tasks."""

from enum import StrEnum
from pathlib import Path

from omegaconf import DictConfig
from rigging.runtime_bundle import RuntimeBundle, install_runtime_bundle
from shellbox.backends.docker.machine import DockerMachineFactory
from shellbox.backends.qemu.image import QemuAssets
from shellbox.backends.qemu.machine import Acceleration, QemuMachineFactory
from shellbox.machine import MachineFactory


class TaskMachineBackend(StrEnum):
    DOCKER = "docker"
    QEMU = "qemu"


def task_machine_factory(config: DictConfig) -> MachineFactory:
    """Create the selected image-backed task factory."""
    match TaskMachineBackend(config.machine.backend):
        case TaskMachineBackend.DOCKER:
            return DockerMachineFactory(
                skopeo=Path(config.skopeo).expanduser(), image_cache=Path(config.image_cache).expanduser()
            )
        case TaskMachineBackend.QEMU:
            qemu_config = config.machine.qemu
            if qemu_config is None:
                raise ValueError("The QEMU task backend requires trajectory_runner.machine.qemu")
            bundle = config.machine.get("runtime_bundle")
            if bundle is not None:
                if qemu_config.assets is not None:
                    raise ValueError("QEMU assets must come from either the runtime bundle or explicit paths")
                runtime_bundle = RuntimeBundle(**dict(bundle))
                manifest = install_runtime_bundle(runtime_bundle)
                return QemuMachineFactory(
                    acceleration=Acceleration(qemu_config.acceleration),
                    prepared_registry_bundles={
                        manifest["source_image"]: Path(runtime_bundle.installation_parent) / manifest["directory_name"],
                    },
                )
            assets_config = None if qemu_config.assets is None else dict(qemu_config.assets)
            if assets_config is None:
                raise ValueError("The QEMU task backend requires trajectory_runner.machine.qemu.assets")
            assets = QemuAssets(
                qemu=Path(assets_config["qemu"]).expanduser(),
                kernel=Path(assets_config["kernel"]).expanduser(),
                busybox=Path(assets_config["busybox"]).expanduser(),
                firmware=Path(assets_config["firmware"]).expanduser(),
                libraries=Path(assets_config["libraries"]).expanduser(),
                umoci=Path(assets_config["umoci"]).expanduser(),
                disk_size_mb=assets_config["disk_size_mb"],
                runtime_id=assets_config["runtime_id"],
            )
            return QemuMachineFactory(
                acceleration=Acceleration(qemu_config.acceleration),
                assets=assets,
                bundle_cache=Path(qemu_config.bundle_cache).expanduser(),
                image_cache=Path(config.image_cache).expanduser(),
                skopeo=Path(config.skopeo).expanduser(),
            )
