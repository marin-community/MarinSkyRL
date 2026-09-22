"""Evaluation-only entrypoint for TerminalBench tasks."""

import asyncio

import hydra
import ray
from omegaconf import DictConfig

from skyrl_train.config.trajectory_runner_capabilities import EntrypointOperation, TrajectoryRunnerMode
from skyrl_train.entrypoints.main_base import (
    config_dir,
    run_ray_driver,
)
from skyrl_train.entrypoints.main_generate import run_evaluation_only
from skyrl_train.entrypoints.terminal_bench import TerminalBenchExp


class TerminalBenchGenerateExp(TerminalBenchExp):
    def get_train_dataset(self):
        """Avoid loading training data for an evaluation-only run."""
        return None

    def run(self):
        return asyncio.run(run_evaluation_only(self))


@ray.remote(num_cpus=1, max_retries=0)
def skyrl_entrypoint(cfg: DictConfig):
    # make sure that the training loop is not run on the head node.
    exp = TerminalBenchGenerateExp(cfg)
    exp.run()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    run_ray_driver(
        cfg,
        skyrl_entrypoint,
        TrajectoryRunnerMode.HARBOR,
        operation=EntrypointOperation.GENERATE,
        failure_message="Generation failed",
    )


if __name__ == "__main__":
    main()
