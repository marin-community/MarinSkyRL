"""Train on mixed TaskCompendium lowerings through their Harbor launches."""

from __future__ import annotations

import tempfile
from pathlib import Path

import hydra
import ray
from omegaconf import DictConfig

from skyrl_train.config.trajectory_runner_capabilities import TrajectoryRunnerMode
from skyrl_train.entrypoints.main_base import config_dir, run_ray_driver
from skyrl_train.entrypoints.terminal_bench import TerminalBenchExp


class TaskCompendiumExp(TerminalBenchExp):
    """Use per-task Harbor executions against one live policy endpoint."""

    def _api_base(self) -> str:
        configured = str(self.cfg.terminal_bench_config.get("agent_api_base", "") or "").strip()
        if configured:
            return configured.rstrip("/")
        return f"http://{self.cfg.generator.http_endpoint_host}:{self.cfg.generator.http_endpoint_port}/v1"

    def get_trajectory_runner(self, cfg, tokenizer, inference_engine_client):
        del inference_engine_client
        from taskcompendium.skyrl import TaskCompendiumTrajectoryRunner  # noqa: PLC0415

        output_dir = Path(tempfile.mkdtemp(prefix="taskcompendium-attempts-"))
        return TaskCompendiumTrajectoryRunner(
            tokenizer,
            output_dir,
            concurrency=int(cfg.taskcompendium.concurrency),
            archive_uri=str(cfg.taskcompendium.archive_uri) if cfg.taskcompendium.archive_uri else None,
        )

    def get_train_dataset(self):
        from skyrl_train.trajectory_runners.taskcompendium import TaskCompendiumTaskDataset  # noqa: PLC0415

        dataset = TaskCompendiumTaskDataset(
            self.cfg.data.train_data,
            api_base=self._api_base(),
            model_name=str(self.cfg.generator.model_name),
        )
        assert len(dataset) >= self.cfg.trainer.train_batch_size, (
            f"dataset should be at least as large as train_batch_size "
            f"{self.cfg.trainer.train_batch_size}, got {len(dataset)}"
        )
        return dataset

    def get_eval_dataset(self):
        return None


@ray.remote(num_cpus=1, max_retries=0)
def skyrl_entrypoint(cfg: DictConfig):
    TaskCompendiumExp(cfg).run()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    run_ray_driver(cfg, skyrl_entrypoint, TrajectoryRunnerMode.TASKCOMPENDIUM)


if __name__ == "__main__":
    main()
