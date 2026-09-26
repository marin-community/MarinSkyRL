"""
Main entrypoint for async training.
"""

from dataclasses import dataclass

import hydra
import ray
from omegaconf import DictConfig, OmegaConf
from transformers import PreTrainedTokenizerBase

from skyrl_train.config.trajectory_runner_capabilities import TrajectoryRunnerMode
from skyrl_train.entrypoints.main_base import BasePPOExp, config_dir, run_ray_driver
from skyrl_train.fully_async_trainer import FullyAsyncRayPPOTrainer
from skyrl_train.inference_engines.base import InferenceEngineInterface
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.rollouts.context import TrainingContext
from skyrl_train.rollouts.workers import RolloutWorkerPool
from skyrl_train.trajectory_runners.model_clients import DirectModelClient
from skyrl_train.trajectory_runners.skyrl_gym import SkyRLGymTrajectoryRunner


def build_gym_runner(
    cfg: DictConfig, tokenizer: PreTrainedTokenizerBase, inference_engine_client: InferenceEngineClient
) -> SkyRLGymTrajectoryRunner:
    """Build the SkyRL-Gym runner that fully asynchronous training requires."""
    if not cfg.generator.use_conversation_multi_turn:
        raise ValueError("the fully asynchronous Gym entrypoint requires multi-turn conversations")
    runner = SkyRLGymTrajectoryRunner(
        trajectory_runner_cfg=cfg.generator,
        skyrl_gym_cfg=cfg.environment.skyrl_gym,
        inference_engine_client=inference_engine_client,
        tokenizer=tokenizer,
        model_client=DirectModelClient(inference_engine_client),
    )
    if runner.custom_chat_template is None:
        raise ValueError("the fully asynchronous Gym entrypoint requires a custom chat template")
    return runner


@dataclass(frozen=True)
class GymRunnerSpec:
    """Builds the SkyRL-Gym runner inside a rollout worker, with its own client for the shared engines."""

    config: DictConfig
    engines: list[InferenceEngineInterface]

    def build(self, tokenizer: PreTrainedTokenizerBase) -> SkyRLGymTrajectoryRunner:
        # Only the trainer's client serves the HTTP endpoint.
        client_config = OmegaConf.merge(self.config, {"generator": {"enable_http_endpoint": False}})
        return build_gym_runner(self.config, tokenizer, InferenceEngineClient(self.engines, tokenizer, client_config))


class AsyncPPOExp(BasePPOExp):
    def uses_fully_async_trainer(self) -> bool:
        return True

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
        return FullyAsyncRayPPOTrainer(
            context=TrainingContext.from_config(cfg, train_dataset, trajectory_runner),
            cfg=cfg,
            tracker=tracker,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            inference_engine_client=inference_engine_client,
            trajectory_runner=trajectory_runner,
            colocate_pg=colocate_pg,
        )

    def get_trajectory_runner(self, cfg, tokenizer, inference_engine_client):
        """Run SkyRL-Gym in a pool of rollout worker processes.

        Returns:
            RolloutWorkerPool: The pool, which also serves evaluation requests.
        """
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
    exp = AsyncPPOExp(cfg)
    exp.run()


def run(cfg: DictConfig) -> None:
    run_ray_driver(cfg, skyrl_entrypoint, TrajectoryRunnerMode.FULLY_ASYNC_SKYRL_GYM)


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
