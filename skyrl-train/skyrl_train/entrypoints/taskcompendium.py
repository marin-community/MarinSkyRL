"""Train from unified TaskCompendium tasks with the Shellbox rollout engine."""

import hydra
import ray
from omegaconf import DictConfig

from skyrl_train.dataset.tasks import TaskDataset, TaskIdDataset
from skyrl_train.entrypoints.main_base import BasePPOExp, config_dir, run_ray_driver


def require_batch_concurrency(cfg: DictConfig) -> None:
    """Refuse a worker pool that cannot hold one training batch of rollouts in flight at once."""
    batch = int(cfg.trainer.train_batch_size) * int(cfg.generator.n_samples_per_prompt)
    slots = int(cfg.trajectory_runner.max_concurrent_tasks) * int(cfg.trajectory_runner.rollout_workers.num_workers)
    if slots < batch:
        raise ValueError(
            f"trajectory_runner.max_concurrent_tasks x rollout_workers.num_workers = {slots} rollout slots cannot hold "
            f"one batch of {batch} rollouts (train_batch_size x n_samples_per_prompt); raise max_concurrent_tasks"
        )


class TaskCompendiumExp(BasePPOExp):
    def get_train_dataset(self):
        require_batch_concurrency(self.cfg)
        dataset = TaskDataset(self.cfg.data.train_data, self.tokenizer, self.cfg.trainer.max_prompt_length)
        if len(dataset) < self.cfg.trainer.train_batch_size:
            raise ValueError("The task dataset is smaller than the training batch")
        return dataset

    def get_eval_dataset(self):
        if self.cfg.trainer.eval_interval > 0 and self.cfg.data.val_data:
            return TaskIdDataset(self.cfg.data.val_data, self.tokenizer, self.cfg.trainer.max_prompt_length)
        return None


@ray.remote(num_cpus=1, max_retries=0)
def skyrl_entrypoint(cfg: DictConfig):
    TaskCompendiumExp(cfg).run()


def run(cfg: DictConfig) -> None:
    run_ray_driver(cfg, skyrl_entrypoint)


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
