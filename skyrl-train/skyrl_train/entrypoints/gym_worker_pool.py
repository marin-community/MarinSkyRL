"""Training entrypoint that runs SkyRL-Gym in a pool of rollout worker processes."""

from dataclasses import dataclass

import hydra
import ray
from omegaconf import DictConfig, OmegaConf
from transformers import PreTrainedTokenizerBase

from skyrl_train.config.trajectory_runner_capabilities import TrajectoryRunnerMode
from skyrl_train.entrypoints.main_base import BasePPOExp, build_gym_trajectory_runner, config_dir, run_ray_driver
from skyrl_train.inference_engines.base import InferenceEngineInterface
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.rollouts.workers import RolloutWorkerPool
from skyrl_train.trajectory_runners.base import TrajectoryRunner


@dataclass(frozen=True)
class GymRunnerSpec:
    """Builds the SkyRL-Gym runner inside a rollout worker, with its own client for the shared engines."""

    config: DictConfig
    engines: list[InferenceEngineInterface]

    def build(self, tokenizer: PreTrainedTokenizerBase) -> TrajectoryRunner:
        # Only the trainer's client serves the HTTP endpoint.
        client_config = OmegaConf.merge(self.config, {"generator": {"enable_http_endpoint": False}})
        client = InferenceEngineClient(self.engines, tokenizer, client_config)
        return build_gym_trajectory_runner(self.config, tokenizer, client)


class GymWorkerPoolExp(BasePPOExp):
    """Generate rollouts in worker processes, so environment execution runs outside the trainer process.

    Environments are built from their registered names inside each worker, so an environment registered only in
    the driver process is unavailable here; use the standard entrypoint for those.
    """

    def get_trajectory_runner(self, cfg, tokenizer, inference_engine_client):
        """Returns the worker pool, which also serves evaluation requests."""
        process_pool = cfg.trajectory_runner.process_pool
        spec = GymRunnerSpec(
            config=OmegaConf.create(OmegaConf.to_container(cfg, resolve=True)),
            engines=list(inference_engine_client.engines),
        )
        return RolloutWorkerPool(
            spec,
            num_workers=process_pool.num_coordinators,
            cpus_per_worker=process_pool.cpus_per_coordinator,
        )


@ray.remote(num_cpus=1, max_retries=0)
def skyrl_entrypoint(cfg: DictConfig):
    # make sure that the training loop is not run on the head node.
    exp = GymWorkerPoolExp(cfg)
    exp.run()


def run(cfg: DictConfig) -> None:
    run_ray_driver(cfg, skyrl_entrypoint, TrajectoryRunnerMode.SKYRL_GYM)


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
