"""End-to-end CPU DPO training through the production entrypoint on a tiny policy.

The experiment swaps the Megatron workers for CPU policy/reference workers and runs the static
preference-pair runner with no inference engines, so the real trainer, objective, pairing
plumbing, and reduction run on synthetic preference data.
"""

import argparse
import json
from pathlib import Path

import ray
from omegaconf import DictConfig, OmegaConf
from skyrl_train.config.utils import get_default_config
from skyrl_train.entrypoints.main_base import BasePPOExp
from skyrl_train.rollouts.loader import EpochTail

from tests.cpu.tiny_training.cpu_backend import CPUPolicyWorker, CPURefWorker
from tests.cpu.tiny_training.experiment import (
    METRICS_FILE,
    JsonlTracker,
    run_with_exp,
)
from tests.cpu.tiny_training.tiny_model import build_tiny_policy
from tests.cpu.util import DPO_OVERRIDES

# Correct completions are preferred over off-by-one distractors. The pairs are separable under
# any tiny model because the completions differ, and DPO must widen the margin without generation.
PREFERENCE_ROWS = [
    {
        "prompt": [{"role": "user", "content": f"Q: {a} + {b}\nA:"}],
        "chosen": f" {a + b}",
        "rejected": f" {a + b + 1}",
    }
    for a, b in [(1, 1), (2, 3), (4, 5), (6, 1), (2, 2), (3, 3), (5, 4), (7, 2)] * 8
]


def write_preference_rows(path: Path, num_pairs: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in PREFERENCE_ROWS[:num_pairs]:
            handle.write(json.dumps(row) + "\n")
    return path


def preference_pair_training_config(
    root: Path,
    model_dir: Path,
    *,
    steps: int,
    beta: float = 0.1,
    num_pairs: int = len(PREFERENCE_ROWS),
    epoch_tail: EpochTail = EpochTail.DROP,
) -> DictConfig:
    overrides = {
        "data": {
            "train_data": [str(write_preference_rows(root / "data" / "preferences.jsonl", num_pairs))],
            "val_data": [],
            "shuffle": False,
            "epoch_tail": epoch_tail.value,
        },
        "trainer": {
            "debug_mode": "off",
            "placement": {"policy_num_gpus_per_node": 1, "ref_num_gpus_per_node": 1},
            "policy": {"model": {"path": str(model_dir)}, "optimizer_config": {"lr": 5e-4, "weight_decay": 0.0}},
            "algorithm": {"group_admission": {"stall_timeout": 120}},
            "rollout_buffer": {"max_staleness_steps": 0, "max_in_flight": 8, "object_store_root": None},
            "train_batch_size": 4,
            "policy_mini_batch_size": 4,
            "micro_train_batch_size_per_gpu": 4,
            "micro_forward_batch_size_per_gpu": 4,
            "max_prompt_length": 64,
            "max_steps": steps,
            "eval_before_train": False,
            "eval_interval": -1,
            "ckpt_interval": -1,
            "hf_save_interval": -1,
            "resume_mode": "none",
            "ckpt_path": str(root / "ckpts"),
            "export_path": str(root / "exports"),
            "training_metrics": False,
        },
        "generator": {
            "num_inference_engines": 0,
            "inference_engine_tensor_parallel_size": 1,
            "max_turns": 1,
            "inference_stats_interval": 0,
            "enable_http_endpoint": False,
            "sampling_params": {"max_generate_length": 16, "logprobs": None},
            "trajectory_retention": {"enabled": False},
        },
        "trajectory_runner": {"rollout_workers": {"num_workers": 2, "cpus_per_worker": 1, "start_interval_seconds": 0}},
    }
    cfg = OmegaConf.merge(get_default_config(), OmegaConf.create(DPO_OVERRIDES), OmegaConf.create(overrides))
    cfg.trainer.algorithm.dpo.beta = beta
    cfg.trainer.placement.ref_num_gpus_per_node = 1
    cfg.trainer.logger = "console"
    return cfg


class PreferencePairCPUExp(BasePPOExp):
    """The production entrypoint with CPU policy and frozen reference workers."""

    def get_worker_classes(self):
        return ray.remote(num_gpus=1)(CPUPolicyWorker), None, ray.remote(num_gpus=1)(CPURefWorker)

    def get_tracker(self):
        return JsonlTracker(Path(self.cfg.trainer.export_path) / METRICS_FILE)


def run_dpo_experiment(
    root: Path,
    model_dir: Path,
    *,
    steps: int,
    num_pairs: int = len(PREFERENCE_ROWS),
    epoch_tail: EpochTail = EpochTail.DROP,
) -> Path:
    cfg = preference_pair_training_config(root, model_dir, steps=steps, num_pairs=num_pairs, epoch_tail=epoch_tail)
    run_with_exp(cfg, PreferencePairCPUExp)
    return root / "exports" / METRICS_FILE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    args = parser.parse_args()
    model_dir = args.model or build_tiny_policy(args.root / "model")
    run_dpo_experiment(args.root, model_dir, steps=args.steps)


if __name__ == "__main__":
    main()
