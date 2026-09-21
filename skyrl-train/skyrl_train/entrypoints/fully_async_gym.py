"""Fully asynchronous training with the in-process SkyRL Gym trajectory runner.

``skyrl_train.entrypoints.fully_async`` pairs ``FullyAsyncRayPPOTrainer`` with an
HTTP-backed runner. That runner's plain chat path re-tokenizes generated text and
returns neither behavior logprobs nor the student's top-k candidates, so it cannot
feed ``student_selected_topk`` teacher evidence or truncated importance sampling.
This entrypoint keeps the in-process runner ``main_base`` uses, which reports exact
sampled token IDs, and changes only the trainer schedule.
"""

import hydra
import ray
from omegaconf import DictConfig

from skyrl_train.config.trajectory_runner_capabilities import TrajectoryRunnerMode
from skyrl_train.entrypoints.main_base import BasePPOExp, config_dir, run_ray_driver


class FullyAsyncGymExp(BasePPOExp):
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
        from skyrl_train.fully_async_trainer import FullyAsyncRayPPOTrainer  # noqa: PLC0415

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
    # The training loop must not run on the head node.
    FullyAsyncGymExp(cfg).run()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    run_ray_driver(cfg, skyrl_entrypoint, TrajectoryRunnerMode.SKYRL_GYM)


if __name__ == "__main__":
    main()
