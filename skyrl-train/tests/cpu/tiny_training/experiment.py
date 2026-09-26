"""Run the production training entrypoints end to end on CPU with a tiny GSM8K policy.

The experiments swap only the Megatron policy worker and the vLLM engines for the CPU backend.
Ray runs locally with logical GPUs so placement code runs unchanged. Synchronous training runs the standard
entrypoint at staleness 0; asynchronous training runs the Gym worker-pool entrypoint at positive staleness, so
the two modes also cover both rollout-worker topologies.

Usage::

    uv run --frozen --no-sync python -m tests.cpu.tiny_training.experiment --mode async --steps 20
"""

import argparse
import json
import os
from enum import StrEnum
from pathlib import Path

import ray
from omegaconf import DictConfig, OmegaConf

from skyrl_train.config.trajectory_runner_capabilities import (
    TrajectoryRunnerMode,
    validate_trajectory_runner_capabilities,
)
from skyrl_train.config.utils import get_default_config
from skyrl_train.dataset import PromptDataset
from skyrl_train.entrypoints.gym_worker_pool import GymWorkerPoolExp
from skyrl_train.entrypoints.main_base import BasePPOExp, EntrypointOperation
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.ray_wrapped_inference_engine import RayWrappedInferenceEngine
from skyrl_train.utils import validate_cfg
from tests.cpu.tiny_training.cpu_backend import CPUInferenceEngine, CPUPolicyWorker
from tests.cpu.tiny_training.tiny_model import build_tiny_policy, write_gsm8k_dataset

LOGICAL_GPUS = 4
METRICS_FILE = "metrics.jsonl"
WORKER_ENV_VARS = {"HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "4"}


class TrainingMode(StrEnum):
    SYNC = "sync"
    ASYNC = "async"


MAX_STALENESS_STEPS = {TrainingMode.SYNC: 0, TrainingMode.ASYNC: 1}


def tiny_training_config(root: Path, mode: TrainingMode, *, max_steps: int, num_prompts: int = 64) -> DictConfig:
    """Build a complete training config for the tiny policy under ``root``."""
    model_dir = build_tiny_policy(root / "model")
    cfg = get_default_config()
    overrides = {
        "data": {
            "train_data": [str(write_gsm8k_dataset(root / "data" / "train.jsonl", num_prompts))],
            "val_data": [str(write_gsm8k_dataset(root / "data" / "validation.jsonl", 8))],
        },
        "trainer": {
            "debug_mode": "off",
            "placement": {"colocate_all": False, "policy_num_gpus_per_node": 1},
            "policy": {"model": {"path": str(model_dir)}, "optimizer_config": {"lr": 1.0e-3}},
            "algorithm": {"use_kl_loss": False},
            "rollout_buffer": {"max_staleness_steps": MAX_STALENESS_STEPS[mode], "max_in_flight": 8},
            "train_batch_size": 4,
            "policy_mini_batch_size": 4,
            "micro_train_batch_size_per_gpu": 8,
            "micro_forward_batch_size_per_gpu": 8,
            "use_sample_packing": False,
            "max_steps": max_steps,
            "eval_before_train": False,
            "eval_interval": -1,
            "ckpt_interval": -1,
            "hf_save_interval": -1,
            "resume_mode": "none",
            "ckpt_path": str(root / "ckpts"),
            "export_path": str(root / "exports"),
        },
        "generator": {
            "num_inference_engines": 1,
            "inference_engine_tensor_parallel_size": 1,
            "n_samples_per_prompt": 4,
            "inference_stats_interval": 0,
            "sampling_params": {"max_generate_length": 16},
            # Each retention storage operation spawns a process that re-imports the entrypoint.
            "trajectory_retention": {"enabled": False},
        },
        "trajectory_runner": {"process_pool": {"num_coordinators": 2, "cpus_per_coordinator": 1}},
    }
    return OmegaConf.merge(cfg, overrides)


class JsonlTracker:
    """Append every logged metrics record to a JSON Lines file."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path

    def log(self, data, step, commit=True):
        with self.path.open("a") as handle:
            handle.write(json.dumps({"step": step, **data}, default=float) + "\n")


def read_metrics(root: Path) -> list[dict]:
    """Return the metrics records an experiment under ``root`` logged, in order."""
    with (root / "exports" / METRICS_FILE).open() as handle:
        return [json.loads(line) for line in handle]


class TinyTrainingExp(BasePPOExp):
    """The standard entrypoint with CPU policy workers and CPU inference engines."""

    def get_train_dataset(self):
        # Filtering a few dozen prompts in one process beats spawning preprocessing workers.
        return PromptDataset(
            datasets=self.cfg.data.train_data,
            tokenizer=self.tokenizer,
            max_prompt_length=self.cfg.trainer.max_prompt_length,
            num_workers=1,
        )

    def get_tracker(self):
        return JsonlTracker(Path(self.cfg.trainer.export_path) / METRICS_FILE)

    def get_worker_classes(self):
        if self.cfg.trainer.critic.model.path or self.cfg.trainer.algorithm.use_kl_loss:
            raise ValueError("the CPU backend provides only a policy worker")
        return ray.remote(num_gpus=1)(CPUPolicyWorker), None, None

    def create_inference_engine_client(
        self, *, operation: EntrypointOperation = EntrypointOperation.TRAIN
    ) -> InferenceEngineClient:
        engine_actor = ray.remote(CPUInferenceEngine)
        engines = [
            RayWrappedInferenceEngine(
                engine_actor.options(num_cpus=1).remote(
                    self.cfg.trainer.policy.model.path, self.cfg.trainer.seed + index
                )
            )
            for index in range(self.cfg.generator.num_inference_engines)
        ]
        return InferenceEngineClient(engines, self.tokenizer, self.cfg)


class TinyWorkerPoolTrainingExp(TinyTrainingExp, GymWorkerPoolExp):
    """The Gym worker-pool entrypoint with the CPU backend."""


EXPERIMENTS = {TrainingMode.SYNC: TinyTrainingExp, TrainingMode.ASYNC: TinyWorkerPoolTrainingExp}


def run_tiny_training(cfg: DictConfig, mode: TrainingMode) -> None:
    """Validate the config as the production driver does, then run in a fresh local Ray session."""
    validate_cfg(cfg)
    validate_trajectory_runner_capabilities(cfg, TrajectoryRunnerMode.SKYRL_GYM, EntrypointOperation.TRAIN)
    ray.init(num_cpus=os.cpu_count(), num_gpus=LOGICAL_GPUS, runtime_env={"env_vars": WORKER_ENV_VARS})
    try:
        EXPERIMENTS[mode](cfg).run()
    finally:
        ray.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", type=TrainingMode, choices=list(TrainingMode), required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    run_tiny_training(tiny_training_config(args.root, args.mode, max_steps=args.steps), args.mode)


if __name__ == "__main__":
    main()
