"""
Main entrypoint for async training.
"""

import hydra
from omegaconf import DictConfig
from skyrl_train.entrypoints.main_base import BasePPOExp, config_dir, run_ray_driver
from skyrl_train.fully_async_trainer import FullyAsyncRayPPOTrainer
import asyncio
from skyrl_train.trajectory_runners.model_clients import OpenAIHTTPModelClient
from skyrl_train.trajectory_runners.skyrl_gym import SkyRLGymTrajectoryRunner
from skyrl_train.config.trajectory_runner_capabilities import TrajectoryRunnerMode
import ray


def trajectory_runner_mode(cfg: DictConfig) -> TrajectoryRunnerMode:
    """Select the async runner without weakening behavior-evidence checks."""
    http_enabled = bool(cfg.generator.enable_http_endpoint)
    multi_turn = bool(cfg.generator.use_conversation_multi_turn)
    if http_enabled != multi_turn:
        raise ValueError(
            "fully asynchronous HTTP rollouts require generator.enable_http_endpoint=true and "
            "generator.use_conversation_multi_turn=true; direct-engine rollouts require both false"
        )
    if http_enabled:
        return TrajectoryRunnerMode.FULLY_ASYNC_SKYRL_GYM
    return TrajectoryRunnerMode.SKYRL_GYM


class AsyncPPOExp(BasePPOExp):
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

    def get_trajectory_runner(self, cfg, tokenizer, inference_engine_client):
        """Initialize the direct-engine or HTTP-backed SkyRL-Gym runner.

        Returns:
            TrajectoryRunner: The runner.
        """
        mode = trajectory_runner_mode(cfg)
        if mode is TrajectoryRunnerMode.SKYRL_GYM:
            return super().get_trajectory_runner(cfg, tokenizer, inference_engine_client)

        model_client = OpenAIHTTPModelClient(
            base_url=f"http://{cfg.generator.http_endpoint_host}:{cfg.generator.http_endpoint_port}",
            model_name=cfg.trainer.policy.model.path,
            tokenizer=tokenizer,
        )
        runner = SkyRLGymTrajectoryRunner(
            trajectory_runner_cfg=cfg.generator,
            skyrl_gym_cfg=cfg.environment.skyrl_gym,
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
            model_client=model_client,
        )
        if runner.custom_chat_template is None:
            raise ValueError("the fully asynchronous HTTP entrypoint requires a custom chat template")
        return runner

    def run(self):
        trainer = self._setup_trainer()
        # Start the async training loop
        asyncio.run(trainer.train())


@ray.remote(num_cpus=1, max_retries=0)
def skyrl_entrypoint(cfg: DictConfig):
    # make sure that the training loop is not run on the head node.
    exp = AsyncPPOExp(cfg)
    exp.run()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    run_ray_driver(cfg, skyrl_entrypoint, trajectory_runner_mode(cfg))


if __name__ == "__main__":
    main()
