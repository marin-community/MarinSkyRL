"""
Main entrypoint for async training.
"""

import signal
import sys
import hydra
from loguru import logger
from omegaconf import DictConfig
from skyrl_train.entrypoints.main_base import BasePPOExp, config_dir, validate_cfg
from skyrl_train.fully_async_trainer import FullyAsyncRayPPOTrainer
import asyncio
from skyrl_train.utils import initialize_ray
import ray


class AsyncPPOExp(BasePPOExp):
    def get_trainer(
        self,
        cfg,
        tracker,
        tokenizer,
        train_dataset,
        eval_dataset,
        inference_engine_client,
        generator,
        colocate_pg,
    ):
        return FullyAsyncRayPPOTrainer(
            cfg=cfg,
            tracker=tracker,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            inference_engine_client=inference_engine_client,
            generator=generator,
            colocate_pg=colocate_pg,
        )

    def get_generator(self, cfg, tokenizer, inference_engine_client):
        """Initializes the generator.

        Returns:
            GeneratorInterface: The generator.
        """
        from .skyrl_gym_http_generator import SkyRLGymHTTPGenerator

        return SkyRLGymHTTPGenerator(
            generator_cfg=cfg.generator,
            skyrl_gym_cfg=cfg.environment.skyrl_gym,
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
            model_name=cfg.trainer.policy.model.path,
        )

    def run(self):
        trainer = self._setup_trainer()
        # Start the async training loop
        asyncio.run(trainer.train())


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: DictConfig):
    # make sure that the training loop is not run on the head node.
    exp = AsyncPPOExp(cfg)
    exp.run()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    # validate the arguments
    validate_cfg(cfg)

    initialize_ray(cfg)

    # Register SIGTERM handler so that cluster preemption / job scheduler
    # timeouts trigger a clean Ray shutdown instead of leaving orphaned actors.
    def _sigterm_handler(signum, frame):
        logger.warning("Received SIGTERM on head node, shutting down Ray...")
        ray.shutdown()
        sys.exit(1)

    signal.signal(signal.SIGTERM, _sigterm_handler)

    try:
        ray.get(skyrl_entrypoint.remote(cfg))
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        logger.info("Shutting down Ray on head node...")
        ray.shutdown()


if __name__ == "__main__":
    main()
