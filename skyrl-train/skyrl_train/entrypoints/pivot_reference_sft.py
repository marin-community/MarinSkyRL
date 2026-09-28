"""Train on released SWE reference actions through the standard Megatron lifecycle."""

import hydra
import ray
from omegaconf import DictConfig

from skyrl_train.config.trajectory_runner_capabilities import TrajectoryRunnerMode
from skyrl_train.entrypoints.main_base import config_dir, run_ray_driver
from skyrl_train.entrypoints.pivot_token_budget import PivotTokenExp
from skyrl_train.trajectory_runners.pivot_reference import PivotReferenceRunner


class PivotSFTExp(PivotTokenExp):
    def get_trajectory_runner(self, cfg, tokenizer, inference_engine_client):
        if cfg.trainer.algorithm.policy_loss_type != "sft" or cfg.trainer.algorithm.use_kl_loss:
            raise ValueError("Pivot SFT requires supervised loss without a KL penalty")
        evaluation_runner = super().get_trajectory_runner(cfg, tokenizer, inference_engine_client)
        return PivotReferenceRunner(evaluation_runner, tokenizer, cfg.generator)


@ray.remote(num_cpus=1, max_retries=0)
def sft_entrypoint(cfg: DictConfig):
    PivotSFTExp(cfg).run()


def run(cfg: DictConfig) -> None:
    run_ray_driver(cfg, sft_entrypoint, TrajectoryRunnerMode.SKYRL_GYM)


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
