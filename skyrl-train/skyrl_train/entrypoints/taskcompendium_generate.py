"""Evaluate a policy on prepared TaskCompendium rows and write per-task rewards.

Reads ``data.val_data`` (``task_rollout_spec`` parquet), samples ``generator.eval_n_samples_per_prompt``
rollouts per task, and logs the usual ``eval/*`` metrics. With ``trainer.dump_eval_results`` the per-rollout
dumps land under ``<trainer.export_path>/dumped_evals/eval_only/``; this entrypoint adds ``per_task.jsonl`` there,
one line per task id with its rollout count, mean reward and pass rate.
"""

import asyncio
import json
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import hydra
import ray
from loguru import logger
from omegaconf import DictConfig

from skyrl_train.config.rollout_validation import EntrypointOperation
from skyrl_train.dataset.tasks import TaskIdDataset
from skyrl_train.entrypoints.main_base import config_dir, run_ray_driver
from skyrl_train.entrypoints.main_generate import EvalOnlyEntrypoint
from skyrl_train.evaluate import evaluation_dump_dir
from skyrl_train.io import io
from marinskyrl.resource_locator import join_resource_path

PER_TASK_FILE = "per_task.jsonl"
AGGREGATED_FILE = "aggregated_results.jsonl"


def per_task_summary(entries: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fold per-rollout eval dump entries into one record per task id (the dump's ``uid``).

    ``reward`` is the grader's continuous score; a rollout without a verdict (infrastructure failure) counts in
    ``unscored`` and in neither mean. ``pass_rate`` uses the grader's ``passed`` verdict.
    """
    rollouts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        rollouts[entry["uid"]].append(entry)
    summary = []
    for uid, items in sorted(rollouts.items()):
        verdicts = [item["verification"] for item in items if item.get("verification") is not None]
        scored = [verdict for verdict in verdicts if verdict["score"] is not None]
        rewards = [float(verdict["score"]) for verdict in scored]
        passes = [bool(verdict["passed"]) for verdict in scored if verdict["passed"] is not None]
        summary.append(
            {
                "task_id": uid,
                "data_source": items[0]["data_source"],
                "rollouts": len(items),
                "scored": len(scored),
                "unscored": len(items) - len(scored),
                "mean_reward": sum(rewards) / len(rewards) if rewards else None,
                "pass_rate": sum(passes) / len(passes) if passes else None,
                "rewards": rewards,
                "statuses": sorted({verdict["status"] for verdict in verdicts}),
                "stop_reasons": sorted({str(item.get("stop_reason")) for item in items}),
            }
        )
    return summary


def _dump_entries(dump_dir: str) -> Iterable[dict[str, Any]]:
    scheme = dump_dir.split("://", 1)[0] + "://" if "://" in dump_dir else ""
    for path in sorted(io.find_files(dump_dir)):
        name = path.rsplit("/", 1)[-1]
        if not name.endswith(".jsonl") or name in (AGGREGATED_FILE, PER_TASK_FILE):
            continue
        location = path if "://" in path or not scheme else scheme + path
        with io.open_file(location, "r") as lines:
            for line in lines:
                if line.strip():
                    yield json.loads(line)


def write_per_task_summary(dump_dir: str) -> str:
    summary = per_task_summary(_dump_entries(dump_dir))
    if not summary:
        raise RuntimeError(f"No evaluation rollouts were dumped under {dump_dir}")
    destination = join_resource_path(dump_dir, PER_TASK_FILE)
    with io.open_file(destination, "w") as output:
        for record in summary:
            output.write(json.dumps(record) + "\n")
    logger.info("Wrote per-task rewards for {} tasks to {}", len(summary), destination)
    return destination


class TaskCompendiumGenerateExp(EvalOnlyEntrypoint):
    def get_eval_dataset(self):
        if not self.cfg.data.val_data:
            raise ValueError("taskcompendium_generate requires data.val_data")
        if not self.cfg.trainer.dump_eval_results:
            raise ValueError("taskcompendium_generate writes per-task rewards from trainer.dump_eval_results")
        return TaskIdDataset(self.cfg.data.val_data, self.tokenizer, self.cfg.trainer.max_prompt_length)

    def _run(self) -> dict[str, Any]:
        results = asyncio.run(self._evaluate())
        write_per_task_summary(evaluation_dump_dir(str(self.cfg.trainer.export_path), None))
        return results


@ray.remote(num_cpus=1, max_retries=0)
def skyrl_entrypoint(cfg: DictConfig):
    return TaskCompendiumGenerateExp(cfg).run()


def run(cfg: DictConfig) -> None:
    run_ray_driver(cfg, skyrl_entrypoint, operation=EntrypointOperation.GENERATE, failure_message="Evaluation failed")


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
