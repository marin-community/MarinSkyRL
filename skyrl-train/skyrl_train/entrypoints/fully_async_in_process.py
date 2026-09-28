"""Fully asynchronous training with the in-process SkyRL Gym trajectory runner.

Keep the base entrypoint's runner selection, including the Nemotron router, and
change only the trainer schedule. The separate ``fully_async`` entrypoint requires
conversation multi-turn mode and a custom chat template.
"""

import hydra
import ray
from omegaconf import DictConfig

from skyrl_train.config.trajectory_runner_capabilities import TrajectoryRunnerMode
from skyrl_train.entrypoints.main_base import BasePPOExp, config_dir, run_ray_driver


class FullyAsyncInProcessExp(BasePPOExp):
    def uses_fully_async_trainer(self) -> bool:
        return True


@ray.remote(num_cpus=1, max_retries=0)
def skyrl_entrypoint(cfg: DictConfig):
    # The training loop must not run on the head node.
    FullyAsyncInProcessExp(cfg).run()


def run(cfg: DictConfig) -> None:
    run_ray_driver(cfg, skyrl_entrypoint, TrajectoryRunnerMode.SKYRL_GYM)


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
