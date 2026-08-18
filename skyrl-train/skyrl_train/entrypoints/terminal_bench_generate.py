"""
Main entrypoint for generating rollouts on terminal bench tasks.
"""

import ray
import asyncio
import hydra
from omegaconf import DictConfig

from skyrl_train.entrypoints.main_base import (
    config_dir,
    run_ray_driver,
)
from skyrl_train.entrypoints.terminal_bench import TerminalBenchExp
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_train.trajectory_runners.trajectory_processing import prepare_trajectory_request


class TerminalBenchGenerateExp(TerminalBenchExp):
    def _setup_trajectory_runner(self):
        inference_engine_client = self.create_inference_engine_client()
        asyncio.run(inference_engine_client.wake_up())
        return self.get_trajectory_runner(self.cfg, self.tokenizer, inference_engine_client)

    def run(self):
        trajectory_runner = self._setup_trajectory_runner()

        input_batch, _ = prepare_trajectory_request(
            list(self.train_dataset),
            self.cfg.generator.n_samples_per_prompt,
            get_sampling_params_for_backend(self.cfg.generator.backend, self.cfg.generator.sampling_params),
            self.cfg.environment.env_class,
            "eval",
            0,
        )

        asyncio.run(trajectory_runner.run(input_batch))


@ray.remote(num_cpus=1, max_retries=0)
def skyrl_entrypoint(cfg: DictConfig):
    # make sure that the training loop is not run on the head node.
    exp = TerminalBenchGenerateExp(cfg)
    exp.run()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    run_ray_driver(cfg, skyrl_entrypoint, failure_message="Generation failed")


if __name__ == "__main__":
    main()
