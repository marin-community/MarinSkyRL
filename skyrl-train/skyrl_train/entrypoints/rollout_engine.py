"""Opt-in training entrypoint for native TaskCompendium rollouts."""

import hydra
from omegaconf import DictConfig
import ray

from skyrl_train.config.trajectory_runner_capabilities import TrajectoryRunnerMode
from skyrl_train.entrypoints.main_base import BasePPOExp, config_dir, run_ray_driver
from skyrl_train.rollouts.workers import RolloutWorkerPool, RolloutWorkerResources
from skyrl_train.trajectory_runners.rollout_engine import RolloutEngineRunnerSpec, validate_rollout_engine_config


class RolloutEnginePPOExp(BasePPOExp):
    def get_trajectory_runner(self, cfg, tokenizer, inference_engine_client):
        del tokenizer
        return RolloutWorkerPool(
            RolloutEngineRunnerSpec.from_config(cfg, inference_engine_client.engines),
            RolloutWorkerResources.from_config(cfg),
        )


@ray.remote(num_cpus=1, max_retries=0)
def skyrl_entrypoint(cfg: DictConfig) -> None:
    RolloutEnginePPOExp(cfg).run()


def run(cfg: DictConfig) -> None:
    validate_rollout_engine_config(cfg)
    run_ray_driver(cfg, skyrl_entrypoint, TrajectoryRunnerMode.ROLLOUT_ENGINE)


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
