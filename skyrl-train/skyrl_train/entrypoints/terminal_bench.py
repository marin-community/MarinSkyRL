"""
Main entrypoint for training on terminal bench tasks.
"""

import ray
import hydra
from omegaconf import DictConfig
from skyrl_train.entrypoints.main_base import BasePPOExp, config_dir, run_ray_driver
from skyrl_train.utils.fd_monitor import start_fd_monitor
from skyrl_train.trajectory_runners.harbor.dataset import TerminalBenchTaskDataset
from skyrl_train.fully_async_trainer import FullyAsyncRayPPOTrainer
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.utils.algorithm_registry import rollout_logprobs_enabled


class TerminalBenchExp(BasePPOExp):
    def get_trajectory_runner(self, cfg, tokenizer, inference_engine_client):
        # Harbor is an optional agent-harness dependency and is absent from the CPU launcher environment.
        from skyrl_train.trajectory_runners.harbor.runner import HarborTrajectoryRunner  # noqa: PLC0415

        return HarborTrajectoryRunner(
            trajectory_runner_cfg=cfg.generator,
            terminal_bench_cfg=cfg.terminal_bench_config,
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
            moe_router_replay=bool(cfg.trainer.policy.fsdp_config.get("moe_router_replay", False)),
            rollout_logprobs_required=rollout_logprobs_enabled(cfg.trainer.algorithm),
            tito_full=cfg.trainer.algorithm.get("tito_full", None),
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
        assert len(prompts_dataset) >= self.cfg.trainer.train_batch_size, (
            f"dataset should be atleast as large as `train_batch_size` {self.cfg.trainer.train_batch_size}, got size {len(prompts_dataset)}"
        )
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
        trajectory_runner,
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
            trajectory_runner=trajectory_runner,
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
    run_ray_driver(cfg, skyrl_entrypoint)


if __name__ == "__main__":
    main()
