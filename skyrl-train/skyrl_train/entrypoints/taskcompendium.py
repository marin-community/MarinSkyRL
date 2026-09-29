"""Train on TaskCompendium lowerings through native chat or Harbor."""

from __future__ import annotations

import tempfile
from pathlib import Path

import hydra
import ray
from omegaconf import DictConfig

from skyrl_train.config.trajectory_runner_capabilities import TrajectoryRunnerMode
from skyrl_train.entrypoints.main_base import BasePPOExp, config_dir, run_ray_driver
from skyrl_train.inference_engines.model_identity import served_model_name


class TaskCompendiumExp(BasePPOExp):
    """Route answer-only tasks to native chat and richer tasks to Harbor."""

    def _api_base(self) -> str:
        configured = str(self.cfg.terminal_bench_config.get("agent_api_base", "") or "").strip()
        if configured:
            return configured.rstrip("/")
        if not self.cfg.generator.enable_http_endpoint:
            raise ValueError(
                "TaskCompendium Harbor rollouts require terminal_bench_config.agent_api_base "
                "or generator.enable_http_endpoint=true"
            )
        return f"http://{self.cfg.generator.http_endpoint_host}:{self.cfg.generator.http_endpoint_port}/v1"

    def get_trajectory_runner(self, cfg, tokenizer, inference_engine_client):
        from skyrl_train.trajectory_runners.model_clients import DirectModelClient  # noqa: PLC0415
        from skyrl_train.trajectory_runners.taskcompendium import (  # noqa: PLC0415
            NativeTaskCompendiumRunner,
            TaskCompendiumHarborRunner,
            TaskCompendiumTrajectoryRouter,
        )
        from skyrl_train.utils.algorithm_registry import rollout_logprobs_enabled  # noqa: PLC0415

        output_dir = Path(tempfile.mkdtemp(prefix="taskcompendium-attempts-"))
        harbor_runner = TaskCompendiumHarborRunner(
            tokenizer,
            output_dir,
            concurrency=int(cfg.taskcompendium_config.concurrency),
            max_turns=int(cfg.taskcompendium_config.max_turns),
            timeout=float(cfg.taskcompendium_config.timeout),
        )
        native_runner = NativeTaskCompendiumRunner(cfg.generator, tokenizer, DirectModelClient(inference_engine_client))
        return TaskCompendiumTrajectoryRouter(
            native_runner=native_runner,
            harbor_runner=harbor_runner,
            require_rollout_logprobs=rollout_logprobs_enabled(cfg.trainer.algorithm),
            tis_lcs_alert_threshold=float(cfg.trainer.algorithm.tis_lcs_alert_threshold),
        )

    def get_train_dataset(self):
        from skyrl_train.trajectory_runners.taskcompendium import TaskCompendiumTaskDataset  # noqa: PLC0415

        dataset = TaskCompendiumTaskDataset(
            self.cfg.data.train_data,
            api_base=self._api_base(),
            model_name=served_model_name(self.cfg),
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
    from skyrl_train.utils.fd_monitor import start_fd_monitor  # noqa: PLC0415

    start_fd_monitor()
    TaskCompendiumExp(cfg).run()


def run(cfg: DictConfig) -> None:
    run_ray_driver(cfg, skyrl_entrypoint, TrajectoryRunnerMode.TASKCOMPENDIUM)


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
