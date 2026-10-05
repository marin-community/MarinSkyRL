"""Run a configured image-backed task machine without a trainer."""

import argparse
import asyncio
from pathlib import Path

from omegaconf import OmegaConf
from shellbox.image import RegistryImage
from shellbox.machine import Command, MachineSpec, NetworkPolicy

from skyrl_train.rollouts.task_machines import task_machine_factory


async def check(config: Path, image: str) -> None:
    base = OmegaConf.load(Path(__file__).parents[1] / "skyrl_train/config/ppo_base_config.yaml")
    factory = task_machine_factory(OmegaConf.merge(base, OmegaConf.load(config)).trajectory_runner)
    machine = await factory.create(
        MachineSpec(source=RegistryImage(image), network=NetworkPolicy.DENY, workdir="/tmp", memory_mb=512)
    )
    try:
        written = await machine.run(Command(("/bin/sh", "-c", "printf shellbox-task > answer.txt"), timeout=30))
        if written.exit_code != 0:
            raise RuntimeError(f"Task machine write failed: {written}")
        read = await machine.run(Command(("/bin/sh", "-c", "cat answer.txt"), timeout=30))
        if read.exit_code != 0 or read.stdout != b"shellbox-task":
            raise RuntimeError(f"Task machine read failed: {read}")
    finally:
        await machine.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("image", help="Registry image pinned by digest.")
    options = parser.parse_args()
    asyncio.run(check(options.config, options.image))
