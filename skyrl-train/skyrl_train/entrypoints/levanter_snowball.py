"""Synchronous MSRL entrypoint with a Levanter Snowball policy learner."""

from __future__ import annotations

import hydra
import ray
from omegaconf import DictConfig
from skyrl_train.config.trajectory_runner_capabilities import TrajectoryRunnerMode
from skyrl_train.entrypoints.main_base import BasePPOExp, config_dir, run_ray_driver
from skyrl_train.learners.levanter_config import LevanterSnowballRuntimeConfig


class LevanterSnowballExp(BasePPOExp):
    def get_trainer(
        self,
        cfg,
        tracker,
        tokenizer,
        train_dataset,
        eval_dataset,
        inference_engine_client,
        trajectory_runner,
        colocate_pg,
    ):
        self.learner.connect_inference_engine(inference_engine_client)
        return super().get_trainer(
            cfg,
            tracker,
            tokenizer,
            train_dataset,
            eval_dataset,
            inference_engine_client,
            trajectory_runner,
            colocate_pg,
        )


@ray.remote(num_cpus=1, max_retries=0)
def skyrl_entrypoint(cfg: DictConfig) -> None:
    # Importing this module initializes JAX. This task already owns the learner
    # GPUs, so JAX sees only its Ray-assigned devices.
    from skyrl_train.learners.levanter_snowball import LevanterSnowballLearner

    runtime = LevanterSnowballRuntimeConfig.from_msrl(cfg)
    learner = LevanterSnowballLearner(runtime)
    LevanterSnowballExp(cfg, learner=learner).run()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    runtime = LevanterSnowballRuntimeConfig.from_msrl(cfg)
    learner_entrypoint = skyrl_entrypoint.options(num_gpus=runtime.training_gpus)
    run_ray_driver(cfg, learner_entrypoint, TrajectoryRunnerMode.SKYRL_GYM)


if __name__ == "__main__":
    main()
