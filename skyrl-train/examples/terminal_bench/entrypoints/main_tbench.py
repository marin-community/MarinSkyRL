"""
Main entrypoint for training on terminal bench tasks.
"""

import os
import signal
import sys
import ray
import hydra
from loguru import logger
from omegaconf import DictConfig
from skyrl_train.entrypoints.main_base import BasePPOExp, config_dir
from skyrl_train.utils import validate_cfg
from skyrl_train.utils.utils import initialize_ray
from examples.terminal_bench.terminal_bench_generator import TerminalBenchGenerator
from examples.terminal_bench.dataset import TerminalBenchTaskDataset
from examples.terminal_bench.fd_monitor import start_fd_monitor
from skyrl_train.fully_async_trainer import FullyAsyncRayPPOTrainer
from skyrl_train.trainer import RayPPOTrainer

class TerminalBenchExp(BasePPOExp):
    def get_generator(self, cfg, tokenizer, inference_engine_client):
        """
        Initializes the TerminalBenchGenerator.
        """
        return TerminalBenchGenerator(
            generator_cfg=cfg.generator,
            terminal_bench_cfg=cfg.terminal_bench_config,  # Pass terminal_bench config to the generator
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
        )

    def get_train_dataset(self):
        """Initializes the training dataset.

        Returns:
            TerminalBenchTaskDataset: The training dataset.
        """
        prompts_dataset = TerminalBenchTaskDataset(
            data_files=self.cfg.data.train_data,
        )
        # make sure the dataset is large enough to train on
        assert (
            len(prompts_dataset) >= self.cfg.trainer.train_batch_size
        ), f"dataset should be atleast as large as `train_batch_size` {self.cfg.trainer.train_batch_size}, got size {len(prompts_dataset)}"
        return prompts_dataset

    def get_eval_dataset(self):
        """Initializes the evaluation dataset.

        Returns:
            TerminalBenchTaskDataset: The evaluation dataset.
        """
        if self.cfg.trainer.eval_interval > 0 and self.cfg.data.val_data:
            prompts_dataset = TerminalBenchTaskDataset(
                data_files=self.cfg.data.val_data,
            )
            return prompts_dataset
        return None

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
        # Check if async training is configured via placement.colocate_all=false
        # Async training requires non-colocated placement (separate GPU sets for policy/ref/inference)
        use_async = (
            hasattr(cfg.trainer, "placement")
            and cfg.trainer.placement is not None
            and getattr(cfg.trainer.placement, "colocate_all", True) is False
        )

        trainer_cls = FullyAsyncRayPPOTrainer if use_async else RayPPOTrainer
        return trainer_cls(
            cfg=cfg,
            tracker=tracker,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            inference_engine_client=inference_engine_client,
            generator=generator,
            colocate_pg=colocate_pg,
        )


@ray.remote(num_cpus=1, max_retries=0)
def skyrl_entrypoint(cfg: DictConfig):
    # make sure that the training loop is not run on the head node.
    # Start the file-descriptor monitor on the driver process. This is the
    # process whose logs show "(skyrl_entrypoint pid=...)" and which FD-aborts
    # (uv__epoll_ctl_prep SIGABRT) on long a3 RL chains. Self-contained daemon
    # thread; only runs here (the driver), not in the per-rank Ray workers.
    start_fd_monitor()
    exp = TerminalBenchExp(cfg)
    exp.run()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    # validate the arguments
    validate_cfg(cfg)

    # Set FP8 fuse_weights env vars from config (must happen before Ray init
    # so all workers inherit them).
    if getattr(cfg.generator, "fuse_weights", False):
        os.environ["SKYRL_FUSE_WEIGHTS"] = "1"
        os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
        logger.info("FP8 fuse_weights enabled: set SKYRL_FUSE_WEIGHTS=1, VLLM_ALLOW_INSECURE_SERIALIZATION=1")

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
