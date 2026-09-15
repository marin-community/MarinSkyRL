"""Fully asynchronous MSRL entrypoint with a Levanter Snowball learner."""

from __future__ import annotations

import asyncio

import hydra
import ray
from omegaconf import DictConfig

from skyrl_train.config.trajectory_runner_capabilities import TrajectoryRunnerMode
from skyrl_train.entrypoints.levanter_snowball import create_levanter_snowball_learner
from skyrl_train.entrypoints.main_base import BasePPOExp, config_dir, run_ray_driver
from skyrl_train.learners.levanter_config import LevanterSnowballRuntimeConfig


def validate_fully_async_levanter_config(cfg: DictConfig) -> None:
    """Reject unsupported async scheduling choices before Ray reserves GPUs."""
    fully_async = cfg.trainer.fully_async
    workers = int(fully_async.num_parallel_generation_workers)
    buffered = fully_async.max_buffered_groups
    configured_publication_steps = fully_async.get("policy_publication_steps")
    unsupported = []
    if cfg.environment.env_class != "gsm8k":
        unsupported.append("environment.env_class=gsm8k")
    if cfg.generator.batched:
        unsupported.append("generator.batched=false")
    if cfg.generator.enable_http_endpoint:
        unsupported.append("generator.enable_http_endpoint=false for direct token evidence")
    if cfg.generator.use_conversation_multi_turn:
        unsupported.append("generator.use_conversation_multi_turn=false")
    if int(fully_async.max_staleness_steps) < 0:
        unsupported.append("a nonnegative max staleness")
    if workers < int(cfg.trainer.policy_mini_batch_size):
        unsupported.append("generation workers at least equal to policy_mini_batch_size")
    if buffered is None or int(buffered) <= 0 or int(buffered) > workers:
        unsupported.append("an explicit positive max_buffered_groups no greater than the worker count")
    if cfg.trainer.train_batch_size != cfg.trainer.policy_mini_batch_size:
        unsupported.append("train_batch_size equal to policy_mini_batch_size")
    if configured_publication_steps is not None:
        publication_steps = [int(step) for step in configured_publication_steps]
        final_step = int(cfg.trainer.max_steps) if cfg.trainer.max_steps is not None else -1
        if (
            publication_steps != sorted(set(publication_steps))
            or not publication_steps
            or publication_steps[0] != 1
            or publication_steps[-1] != final_step
        ):
            unsupported.append("policy_publication_steps containing sorted unique first and final steps")
    if unsupported:
        raise ValueError("the fully async Levanter Snowball entrypoint requires " + ", ".join(unsupported))


class FullyAsyncLevanterSnowballExp(BasePPOExp):
    """Run the async scheduler with exact direct-engine token evidence."""

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
        # The launcher-only environment intentionally omits Torch. Keep the
        # trainer import inside the GPU-owning entrypoint task so callers can
        # compose and validate this configuration without training extras.
        from skyrl_train.fully_async_trainer import FullyAsyncRayPPOTrainer  # noqa: PLC0415

        self.learner.connect_inference_engine(inference_engine_client)
        return FullyAsyncRayPPOTrainer(
            cfg=cfg,
            tracker=tracker,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            inference_engine_client=inference_engine_client,
            trajectory_runner=trajectory_runner,
            colocate_pg=colocate_pg,
            learner=self.learner,
        )

    def run(self) -> None:
        asyncio.run(self._setup_trainer().train())


@ray.remote(num_cpus=1, max_retries=0)
def skyrl_entrypoint(cfg: DictConfig) -> None:
    validate_fully_async_levanter_config(cfg)
    runtime = LevanterSnowballRuntimeConfig.from_msrl(cfg)
    learner = create_levanter_snowball_learner(cfg, runtime)
    try:
        FullyAsyncLevanterSnowballExp(cfg, learner=learner).run()
    finally:
        learner.close()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    validate_fully_async_levanter_config(cfg)
    runtime = LevanterSnowballRuntimeConfig.from_msrl(cfg)
    entrypoint_gpus = runtime.training_gpus_per_node if runtime.training_nodes == 1 else 0
    learner_entrypoint = skyrl_entrypoint.options(num_gpus=entrypoint_gpus)
    # This entrypoint retains the base runner's direct-engine token path. The
    # generic fully-async entrypoint uses a text-only HTTP client instead.
    run_ray_driver(cfg, learner_entrypoint, TrajectoryRunnerMode.SKYRL_GYM)


if __name__ == "__main__":
    main()
