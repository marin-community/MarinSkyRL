"""Synchronous MSRL entrypoint with a Levanter Snowball policy learner."""

from __future__ import annotations

import hydra
import ray
from omegaconf import DictConfig
from skyrl_train.config.trajectory_runner_capabilities import TrajectoryRunnerMode
from skyrl_train.entrypoints.main_base import BasePPOExp, config_dir, run_ray_driver
from skyrl_train.learners.levanter_config import LevanterSnowballRuntimeConfig


def create_levanter_snowball_learner(cfg: DictConfig, runtime: LevanterSnowballRuntimeConfig):
    """Create the local or distributed learner without importing JAX on the driver."""
    if runtime.training_nodes == 1:
        # Importing the concrete learner initializes JAX. This task owns the
        # single learner node, so JAX sees only its Ray-assigned devices.
        from skyrl_train.learners.levanter_snowball import LevanterSnowballLearner  # noqa: PLC0415

        return LevanterSnowballLearner(runtime)

    # The entrypoint owns no GPUs in this path. The facade reserves whole
    # nodes and imports JAX only inside one actor on each learner node.
    from skyrl_train.learners.distributed_levanter import DistributedLevanterSnowballLearner  # noqa: PLC0415

    return DistributedLevanterSnowballLearner(
        runtime,
        placement_timeout_seconds=int(cfg.trainer.distributed.placement_group_timeout_seconds),
    )


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
    runtime = LevanterSnowballRuntimeConfig.from_msrl(cfg)
    learner = create_levanter_snowball_learner(cfg, runtime)
    try:
        LevanterSnowballExp(cfg, learner=learner).run()
    finally:
        learner.close()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    runtime = LevanterSnowballRuntimeConfig.from_msrl(cfg)
    entrypoint_gpus = runtime.training_gpus_per_node if runtime.training_nodes == 1 else 0
    learner_entrypoint = skyrl_entrypoint.options(num_gpus=entrypoint_gpus)
    run_ray_driver(cfg, learner_entrypoint, TrajectoryRunnerMode.SKYRL_GYM)


if __name__ == "__main__":
    main()
