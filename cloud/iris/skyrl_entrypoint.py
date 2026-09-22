"""Run a registered SkyRL entrypoint from a composed launch configuration."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Callable, cast

from omegaconf import DictConfig

from cloud.iris.launch_config import load_launch_config
from cloud.iris.rl_config_translation import rl_entrypoint_spec


def run_config(config_path: Path) -> None:
    """Load one launch config and call its registered SkyRL function."""
    config = load_launch_config(config_path)
    spec = rl_entrypoint_spec(str(config.runtime.entrypoint))
    run = cast(Callable[[DictConfig], None], getattr(importlib.import_module(spec.module), spec.callable))
    run(config.skyrl)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    run_config(args.config)


if __name__ == "__main__":
    main()
