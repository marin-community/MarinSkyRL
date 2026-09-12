"""Run shipped Megatron update methods with CPU model/collective boundaries."""

import ast
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import torch
from loguru import logger
from omegaconf import OmegaConf

from skyrl_train.config.utils import get_default_config
from skyrl_train.megatron_timing import (
    FINAL_BARRIER,
    OPTIMIZER_STEP,
    WORLD_METRIC_REDUCTION,
    MegatronTrainTimings,
    publish_megatron_train_timings,
)
from skyrl_train.training_batch import (
    GLOBAL_LOSS_DENOM_METADATA_KEY,
    TrainingBatchIterator,
    TrainingOutputBatch,
    gradient_accumulation_steps,
)
from skyrl_train.utils.gradient_direction import gradient_direction_summary
from skyrl_train.utils.metrics import policy_progress_metrics, policy_training_metrics
from skyrl_train.utils.progress import tqdm
from skyrl_train.utils.type_c_staleness import optimizer_success_counts
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.weight_sync.policy_weight_access import PolicyWeightAccess
from tests.cpu.test_async_prepared_cohort import make_cohort


def native_worker_methods():
    # Follow the existing weight-publication CPU fixture: execute shipped method
    # bodies without importing the unavailable optional Megatron/CUDA package.
    root = Path(__file__).parents[2] / "skyrl_train/workers/megatron"
    micro = next(
        n
        for n in ast.parse((root / "megatron_model_wrapper.py").read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "MegatronPolicyMicroBatch"
    )
    worker = next(
        n
        for n in ast.parse((root / "megatron_worker.py").read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "MegatronPolicyWorkerBase"
    )
    worker.bases = []
    worker.body = [
        n for n in worker.body if isinstance(n, ast.FunctionDef) and n.name in {"ppo_train", "_ppo_train_with_timings"}
    ]
    namespace = {
        "dataclass": dataclass,
        "Optional": Optional,
        "torch": torch,
        "OmegaConf": OmegaConf,
        "logger": logger,
        "defaultdict": defaultdict,
        "tqdm": tqdm,
        "TrainingBatchIterator": TrainingBatchIterator,
        "TrainingOutputBatch": TrainingOutputBatch,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "MegatronTrainTimings": MegatronTrainTimings,
        "publish_megatron_train_timings": publish_megatron_train_timings,
        "GLOBAL_LOSS_DENOM_METADATA_KEY": GLOBAL_LOSS_DENOM_METADATA_KEY,
        "FINAL_BARRIER": FINAL_BARRIER,
        "OPTIMIZER_STEP": OPTIMIZER_STEP,
        "WORLD_METRIC_REDUCTION": WORLD_METRIC_REDUCTION,
        "policy_training_metrics": policy_training_metrics,
        "policy_progress_metrics": policy_progress_metrics,
        "gradient_direction_summary": gradient_direction_summary,
        "optimizer_success_counts": optimizer_success_counts,
    }
    exec(
        compile(ast.Module(body=[micro, worker], type_ignores=[]), str(root / "megatron_worker.py"), "exec"), namespace
    )
    return namespace["MegatronPolicyWorkerBase"]()


def test_actual_worker_preserves_cohort_index_and_successful_optimizer_clock(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    worker = native_worker_methods()
    worker._policy_weight_access = PolicyWeightAccess()
    worker.cfg = get_default_config()
    worker.cfg.trainer.micro_train_batch_size_per_gpu = 32
    worker.cfg.trainer.policy.megatron_config.check_train_eval_parity = False
    worker.policy_mini_batch_size_per_gpu = 64
    worker.profiler = None
    worker.empty_cuda_cache = False
    worker._memory = SimpleNamespace(span=lambda *args, **kwargs: nullcontext())
    worker._optimizer_state_observer = SimpleNamespace(enabled=False)
    worker._gradient_observer = lambda **kwargs: None
    worker.actor_module = [SimpleNamespace(zero_grad_buffer=lambda: None)]
    worker.optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=0.01)
    worker.scheduler = None
    worker.strategy = SimpleNamespace(
        is_rank_0=lambda: False,
        optimizer_step=lambda *args, **kwargs: 1.0,
        last_grad_metrics={},
        last_optimizer_step_succeeded=True,
        all_reduce=lambda data: data,
    )
    received = []

    def forward_backward(**kwargs):
        received.extend(micro.old_action_log_probs.clone() for micro in kwargs["micro_batches"])
        return [{"policy_loss": 0.2, "policy_entropy": 0.5, "policy_kl": 0.0} for _ in kwargs["micro_batches"]]

    worker.model = SimpleNamespace(train=lambda: None, forward_backward_mini_batch=forward_backward)
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.cfg = worker.cfg
    trainer.colocate_all = False
    trainer.critic_model = None
    trainer._successful_policy_updates = 0
    trainer._training_metrics_enabled = True
    trainer.all_timings = {}
    monkeypatch.setattr("skyrl_train.trainer.collect_actor_results", lambda actors, refs, **kwargs: refs)
    monkeypatch.setattr("skyrl_train.trainer.ray.get", lambda refs: refs)
    monkeypatch.setattr("skyrl_train.trainer.record_event", lambda *args, **kwargs: None)
    cohort = make_cohort()
    for index in range(2):
        # Actual DP4 dispatch sends one 64-response rank strip per update.
        batch = cohort.partition(1 + index).chunk(64)[0]
        batch.metadata["global_step"] = 1 + index
        result = worker.ppo_train(batch)
        assert worker._completed_update == 1 + index
        status = result.metadata["train_status"]
        assert status["policy_update_steps"] == status["policy_successful_update_steps"] == 1
        assert status["policy_successful_update_steps_valid"] == 1
        assert status["update_age_mean"] == status["update_age_max"] == index
        rows = result.metadata["train_status_by_update"]
        assert len(rows) == 1 and rows[0]["update_index"] == rows[0]["update_age"] == index
        assert torch.equal(torch.cat(received[-2:]), batch["action_log_probs"])
        trainer.global_step = 1 + index
        trainer.all_metrics = {}
        trainer.policy_model = SimpleNamespace(
            actor_infos=[SimpleNamespace(rank=SimpleNamespace(dp_size=1))],
            async_run_ray_method=lambda dispatch, method, *args: [result] if method == "ppo_train" else [],
        )
        trainer.train_critic_and_policy(batch)
        assert trainer.all_metrics["policy/updates_attempted"] == index + 1
        assert trainer.all_metrics["policy/updates_completed"] == index + 1
        assert trainer.all_metrics[f"policy/by_update/{index}/update_age"] == index
        assert f"policy/by_update/{1 - index}/update_age" not in trainer.all_metrics
        cohort = cohort.advanced()
