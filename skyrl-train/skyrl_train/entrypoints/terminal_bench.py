"""
Main entrypoint for training on terminal bench tasks.
"""

import ray
import hydra
from pathlib import Path
from omegaconf import DictConfig
from skyrl_train.entrypoints.main_base import BasePPOExp, config_dir, run_ray_driver
from skyrl_train.rollouts.workers import RolloutWorkerPool, RolloutWorkerResources
from skyrl_train.rollouts.task_worker import TaskRolloutWorkerSpec
from skyrl_train.dataset.harbor import HarborTaskDataset
from skyrl_train.rollouts.harbor_tasks import HarborTaskSettings


class TerminalBenchExp(BasePPOExp):
    def get_trajectory_runner(self, cfg, tokenizer, inference_engine_client):
        del tokenizer
        return RolloutWorkerPool(
            TaskRolloutWorkerSpec.from_config(
                cfg, inference_engine_client.engines, harbor_config=cfg.terminal_bench_config
            ),
            RolloutWorkerResources.from_config(cfg),
        )

    def get_train_dataset(self):
        """Initializes the training dataset.

        Returns:
            HarborTaskDataset: The training dataset.
        """
        prompts_dataset = HarborTaskDataset(
            data_files=self.cfg.data.train_data,
            tokenizer=self.tokenizer,
            max_prompt_length=self.cfg.trainer.max_prompt_length,
            cache_dir=Path(self.cfg.data.task_cache_dir),
            verifier_override=HarborTaskSettings.from_config(self.cfg.terminal_bench_config).verifier_override(),
        )
        # make sure the dataset is large enough to train on
        assert len(prompts_dataset) >= self.cfg.trainer.train_batch_size, (
            f"dataset should be atleast as large as `train_batch_size` {self.cfg.trainer.train_batch_size}, got size {len(prompts_dataset)}"
        )
        return prompts_dataset

    def get_eval_dataset(self):
        """Initializes the evaluation dataset.

        Returns:
            HarborTaskDataset: The evaluation dataset.
        """
        if self.cfg.trainer.eval_interval > 0 and self.cfg.data.val_data:
            prompts_dataset = HarborTaskDataset(
                data_files=self.cfg.data.val_data,
                tokenizer=self.tokenizer,
                max_prompt_length=self.cfg.trainer.max_prompt_length,
                cache_dir=Path(self.cfg.data.task_cache_dir),
                verifier_override=HarborTaskSettings.from_config(self.cfg.terminal_bench_config).verifier_override(),
            )
            return prompts_dataset
        return None


@ray.remote(num_cpus=1, max_retries=0)
def skyrl_entrypoint(cfg: DictConfig):
    from skyrl_train.utils.fd_monitor import start_fd_monitor  # noqa: PLC0415

    # make sure that the training loop is not run on the head node.
    # Start the file-descriptor monitor on the driver process. This is the
    # process whose logs show "(skyrl_entrypoint pid=...)" and which FD-aborts
    # (uv__epoll_ctl_prep SIGABRT) on long a3 RL chains. Self-contained daemon
    # thread; only runs here (the driver), not in the per-rank Ray workers.
    start_fd_monitor()
    exp = TerminalBenchExp(cfg)
    exp.run()


def run(cfg: DictConfig) -> None:
    run_ray_driver(cfg, skyrl_entrypoint)


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
