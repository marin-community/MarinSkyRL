"""
Main entrypoint for evaluation-only.
"""

import asyncio
import time
from pathlib import Path
from typing import Any, Protocol

import hydra
import ray
from loguru import logger
from omegaconf import DictConfig

from skyrl_train.entrypoints.main_base import (
    BasePPOExp,
    config_dir,
    run_ray_driver,
)
from skyrl_train.config.trajectory_runner_capabilities import EntrypointOperation, TrajectoryRunnerMode
from skyrl_train.inference_engines.base import NamedWeightsUpdateRequest, lora_disk_load_request
from skyrl_train.inference_engines.vllm.stats import IntervalReadMode
from skyrl_train.inference_observability import (
    VLLM_GENERATION_TOKENS_TOTAL_METRIC,
    format_console_summary,
    trainer_metrics,
)
from skyrl_train.evaluate import evaluate
from skyrl_train.utils.trainer_utils import build_dataloader


class PolicyAdapterClient(Protocol):
    async def update_named_weights(self, request: NamedWeightsUpdateRequest) -> None: ...


async def load_initial_policy_adapter(inference_engine_client: PolicyAdapterClient, cfg: DictConfig) -> None:
    """Load a configured policy LoRA before an evaluation-only rollout."""
    adapter_path = cfg.trainer.policy.model.lora.adapter_path
    if adapter_path is None:
        return
    if not cfg.generator.run_engines_locally:
        raise ValueError("evaluation-only LoRA loading requires local inference engines")
    if cfg.generator.backend != "vllm":
        raise ValueError("evaluation-only LoRA loading currently requires the vLLM backend")
    path = Path(adapter_path)
    if not path.is_absolute() or not path.is_dir():
        raise ValueError("evaluation-only LoRA adapter_path must be an existing absolute directory")
    await inference_engine_client.update_named_weights(lora_disk_load_request(str(path)))


class EvalOnlyEntrypoint(BasePPOExp):
    def get_train_dataset(self):
        """Override to avoid requiring a train dataset for eval-only runs."""
        return None

    async def run(self) -> dict[str, Any]:
        assert self.eval_dataset is not None, "The evaluation only entrypoint requires an eval dataset is provided"

        inference_engine_client = self.create_inference_engine_client()
        trajectory_runner = self.get_trajectory_runner(self.cfg, self.tokenizer, inference_engine_client)
        try:
            await inference_engine_client.wake_up()
            await load_initial_policy_adapter(inference_engine_client, self.cfg)
            started_at = time.monotonic()
            results: dict[str, Any] = await evaluate(
                eval_dataloader=build_dataloader(self.cfg, self.eval_dataset, is_train=False),
                trajectory_runner=trajectory_runner,
                cfg=self.cfg,
                global_step=None,
                tokenizer=self.tokenizer,
            )
            elapsed = time.monotonic() - started_at
            snapshot = await inference_engine_client.get_stats(read_mode=IntervalReadMode.PEEK)
            inference_metrics = trainer_metrics(snapshot)
            generation_tokens = inference_metrics.get(VLLM_GENERATION_TOKENS_TOTAL_METRIC, 0.0)
            results.update(inference_metrics)
            results["eval/all/wall_time_seconds"] = elapsed
            results["eval/all/generation_tokens_per_second"] = generation_tokens / elapsed

            if inference_metrics:
                logger.info(format_console_summary(inference_metrics, step=0))
            tracker = self.get_tracker()
            tracker.log(results, step=0, commit=True)
            return results
        finally:
            inference_engine_client.shutdown_http_endpoint()
            await trajectory_runner.shutdown()
            await inference_engine_client.teardown()


@ray.remote(num_cpus=1, max_retries=0)
def eval_entrypoint(cfg: DictConfig) -> dict:
    exp = EvalOnlyEntrypoint(cfg)
    return asyncio.run(exp.run())


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    run_ray_driver(
        cfg,
        eval_entrypoint,
        TrajectoryRunnerMode.SKYRL_GYM,
        operation=EntrypointOperation.GENERATE,
        failure_message="Evaluation failed",
    )


if __name__ == "__main__":
    main()
