"""
Main entrypoint for generating rollouts on terminal bench tasks.
"""

import signal
import sys
import ray
import asyncio
import hydra
from loguru import logger
from omegaconf import DictConfig

from skyrl_train.utils import validate_cfg
from skyrl_train.utils.utils import initialize_ray
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.entrypoints.main_base import (
    create_ray_wrapped_inference_engines_from_config,
    create_remote_inference_engines_from_config,
    BasePPOExp,
    config_dir,
)
from skyrl_train.trajectory_runners.base import TrajectoryRequestBatch
from skyrl_train.utils.algorithm_registry import rollout_logprobs_enabled


class TerminalBenchGenerateExp(BasePPOExp):
    def get_trajectory_runner(self, cfg, tokenizer, inference_engine_client):
        """
        Initializes the HarborTrajectoryRunner.
        """
        from skyrl_train.trajectory_runners.harbor.runner import HarborTrajectoryRunner

        return HarborTrajectoryRunner(
            generator_cfg=cfg.generator,
            terminal_bench_cfg=cfg.terminal_bench_config,  # Pass terminal_bench config to the generator
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
            moe_router_replay=bool(cfg.trainer.policy.fsdp_config.get("moe_router_replay", False)),
            rollout_logprobs_required=rollout_logprobs_enabled(cfg.trainer.algorithm),
            tito_full=cfg.trainer.algorithm.get("tito_full", None),
        )

    def _setup_trajectory_runner(self):
        logger.info(self.get_cfg_as_str(self.cfg))

        tokenizer = self.tokenizer
        if self.cfg.generator.run_engines_locally:
            inference_engines = create_ray_wrapped_inference_engines_from_config(self.cfg, self.colocate_pg, tokenizer)
        else:
            inference_engines = create_remote_inference_engines_from_config(self.cfg, tokenizer)

        inference_engine_client = InferenceEngineClient(inference_engines, tokenizer, self.cfg)
        asyncio.run(inference_engine_client.wake_up())

        return self.get_trajectory_runner(self.cfg, tokenizer, inference_engine_client)

    def get_train_dataset(self):
        """Initializes the training dataset.

        Returns:
            TerminalBenchTaskDataset: The training dataset.
        """
        from skyrl_train.trajectory_runners.harbor.dataset import TerminalBenchTaskDataset

        prompts_dataset = TerminalBenchTaskDataset(
            data_files=self.cfg.data.train_data,
        )
        assert len(prompts_dataset) >= self.cfg.trainer.train_batch_size, (
            f"dataset should be atleast as large as `train_batch_size` {self.cfg.trainer.train_batch_size}, got size {len(prompts_dataset)}"
        )
        return prompts_dataset

    def run(self):
        trajectory_runner = self._setup_trajectory_runner()

        # Build input from the training dataset
        input_batch = TrajectoryRequestBatch(
            prompts=[item["prompt"] for item in self.train_dataset],
            env_classes=None,
            env_extras=None,
            sampling_params=None,
        )

        # Start generation
        asyncio.run(trajectory_runner.run(input_batch))


@ray.remote(num_cpus=1, max_retries=0)
def skyrl_entrypoint(cfg: DictConfig):
    # make sure that the training loop is not run on the head node.
    exp = TerminalBenchGenerateExp(cfg)
    exp.run()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    # validate the arguments
    validate_cfg(cfg)

    initialize_ray(cfg)

    def _sigterm_handler(signum, frame):
        logger.warning("Received SIGTERM on head node, shutting down Ray...")
        ray.shutdown()
        sys.exit(1)

    signal.signal(signal.SIGTERM, _sigterm_handler)

    try:
        ray.get(skyrl_entrypoint.remote(cfg))
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise
    finally:
        logger.info("Shutting down Ray on head node...")
        ray.shutdown()


if __name__ == "__main__":
    main()
