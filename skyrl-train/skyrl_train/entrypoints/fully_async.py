"""Fully asynchronous training with the direct SkyRL-Gym trajectory runner."""

import hydra
from omegaconf import DictConfig
import ray

from skyrl_train.entrypoints.main_base import BasePPOExp, config_dir, run_ray_driver
from skyrl_train.fully_async_trainer import FullyAsyncRayPPOTrainer


class AsyncPPOExp(BasePPOExp):
    def __init__(self, cfg: DictConfig):
        # Reject before BasePPOExp creates tokenizer, datasets, or placement groups.
        if cfg.trainer.offload_optimizer_during_rollouts:
            raise ValueError("Fully async training requires trainer.offload_optimizer_during_rollouts=false")
        super().__init__(cfg)

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
        )


@ray.remote(num_cpus=1, max_retries=0)
def skyrl_entrypoint(cfg: DictConfig):
    # make sure that the training loop is not run on the head node.
    exp = AsyncPPOExp(cfg)
    exp.run()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    run_ray_driver(cfg, skyrl_entrypoint)


if __name__ == "__main__":
    main()
