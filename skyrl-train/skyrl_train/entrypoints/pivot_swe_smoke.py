"""Multi-update SWE PivotRL smoke with fixed probes before and after training."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import fsspec
import hydra
import ray
from loguru import logger
from omegaconf import DictConfig

from infra.rl_data.pivot_swe import PROBE_PREFIXES, prepare_smoke_sample, write_smoke_report
from skyrl_train.config.trajectory_runner_capabilities import TrajectoryRunnerMode
from skyrl_train.entrypoints.main_base import BasePPOExp, config_dir, run_ray_driver


@ray.remote(num_cpus=1, max_retries=0)
def skyrl_entrypoint(cfg: DictConfig) -> None:
    with tempfile.TemporaryDirectory(prefix="pivot-swe-smoke-") as directory:
        sample_dir = Path(directory)
        manifest = prepare_smoke_sample(
            sample_dir,
            tokenizer_name=str(cfg.trainer.policy.model.path),
            chat_template_kwargs=dict(cfg.generator.chat_template_kwargs),
        )
        cfg.data.train_data = [str(sample_dir / "train.parquet")]
        cfg.data.val_data = [str(sample_dir / "probe.parquet")]
        diagnostics_root = f"{str(cfg.trainer.export_path).rsplit('/', 1)[0]}/diagnostics"
        with fsspec.open(f"{diagnostics_root}/sample-manifest.json", "w") as file:
            json.dump(manifest, file, indent=2, sort_keys=True)
        logger.info("Pivot SWE smoke artifacts: {}", diagnostics_root)
        retention_root = str(cfg.generator.trajectory_retention.output_path)
        final_step = int(cfg.trainer.max_steps)
        try:
            BasePPOExp(cfg).run()
        except BaseException:
            try:
                write_smoke_report(
                    str(cfg.trainer.export_path),
                    diagnostics_root,
                    retention_root,
                    manifest,
                    final_step=final_step,
                    training_completed=False,
                )
            except Exception:
                logger.exception("Could not write partial Pivot SWE smoke report")
            raise
        summary = write_smoke_report(
            str(cfg.trainer.export_path),
            diagnostics_root,
            retention_root,
            manifest,
            final_step=final_step,
            training_completed=True,
        )
        logger.info("Pivot SWE smoke result: {}", json.dumps(summary, sort_keys=True))
        eval_num_prompts = cfg.trainer.eval_num_prompts
        expected_probes = PROBE_PREFIXES if eval_num_prompts is None else min(PROBE_PREFIXES, int(eval_num_prompts))
        if (summary["before_probe_count"], summary["after_probe_count"], summary["training_response_count"]) != (
            expected_probes,
            expected_probes,
            final_step * int(cfg.trainer.train_batch_size) * int(cfg.generator.n_samples_per_prompt),
        ):
            raise RuntimeError("Pivot SWE smoke did not retain the expected probe and training responses")
        if len(summary["training_responses_per_step"]) != final_step:
            raise RuntimeError("Pivot SWE smoke did not retain responses from every training step")
        if summary["mixed_reward_groups"] == 0:
            raise RuntimeError("Pivot SWE smoke produced no mixed-reward groups for GRPO")


@hydra.main(config_path=config_dir, config_name="pivot_swe_smoke", version_base=None)
def main(cfg: DictConfig) -> None:
    run_ray_driver(cfg, skyrl_entrypoint, TrajectoryRunnerMode.SKYRL_GYM)


if __name__ == "__main__":
    main()
