"""Keep actual optimizer-boundary diagnostics through worker and driver logging."""

from types import SimpleNamespace

import pytest
import torch

from skyrl_train.config.utils import get_default_config
from skyrl_train.learner_memory import LearnerMemory
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.workers.fsdp.fsdp_worker import FSDPPolicyWorkerBase


def test_optimizer_statuses_survive_worker_mean_and_driver_logging(monkeypatch):
    class CpuPolicy(FSDPPolicyWorkerBase):
        def training_step(self, experience, global_step, local_step, accumulation_steps):
            assert experience.rollout_age is not None
            if (local_step + 1) % accumulation_steps == 0:
                self.update_count += 1
            return {
                "log_ratio_abs_mean": float(self.update_count),
                "policy_loss": 0.5,
                "response_length": 1,
                "policy_lr": 1e-6,
                "policy_entropy": 0.0,
            }

    cfg = get_default_config()
    cfg.trainer.micro_train_batch_size_per_gpu = 1
    cfg.trainer.update_epochs_per_batch = 2
    cfg.trainer.policy.grug_query_bias_update_mode = "frozen"
    worker = object.__new__(CpuPolicy)
    worker.cfg = cfg
    worker._rank = 0
    worker._is_lora = False
    worker.update_count = 0
    worker.policy_mini_batch_size_per_gpu = 2
    worker.model = SimpleNamespace(model=torch.ones(1))
    worker.strategy = SimpleNamespace(is_rank_0=lambda: False, all_reduce=lambda status: status)
    worker._memory = LearnerMemory(enabled=False, rank=0, backend="fsdp2")
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    batch = TrainingInputBatch(
        {
            key: torch.ones(4, 1)
            for key in (
                "sequences",
                "attention_mask",
                "response_mask",
                "loss_mask",
                "action_log_probs",
                "base_action_log_probs",
                "values",
                "returns",
                "advantages",
            )
        }
    )
    batch["rollout_logprobs"] = None
    batch["rollout_age"] = torch.arange(4, dtype=torch.int32)
    batch.metadata = {"global_step": 7, "response_length": 1}
    output = worker.ppo_train(batch)
    assert output.metadata["train_status"]["policy_update_steps"] == 4
    assert output.metadata["train_status"]["log_ratio_abs_mean"] == pytest.approx(2)
    updates = output.metadata["train_status_by_update"]
    assert [row["update_age"] for row in updates] == [0, 1, 2, 3]
    assert [row["log_ratio_abs_mean"] for row in updates] == [1, 2, 3, 4]

    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.cfg = cfg
    trainer.global_step = 7
    trainer.all_metrics = {}
    trainer.all_timings = {}
    trainer.colocate_all = False
    trainer.critic_model = None
    trainer._training_metrics_enabled = True
    trainer.policy_model = SimpleNamespace(actor_infos=[], async_run_ray_method=lambda *args: [output])
    monkeypatch.setattr("skyrl_train.trainer.collect_actor_results", lambda infos, refs, **kwargs: refs)
    monkeypatch.setattr("skyrl_train.trainer.ray.get", lambda refs: refs)
    events = []
    monkeypatch.setattr("skyrl_train.trainer.record_event", lambda *args, **kwargs: events.append((args, kwargs)))
    mean = trainer.train_critic_and_policy(batch)
    assert mean["log_ratio_abs_mean"] == pytest.approx(2)
    assert [trainer.all_metrics[f"policy/by_update/{k}/update_age"] for k in range(4)] == [0, 1, 2, 3]
    assert [event[0][1]["update_age"] for event in events] == [0, 1, 2, 3]
    assert all(event[0][0] == "policy_update" and event[1]["attributes"]["step"] == "7" for event in events)
