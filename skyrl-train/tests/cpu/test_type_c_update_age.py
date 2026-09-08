"""Exercise actual worker update retention with a deterministic CPU optimizer."""

from types import SimpleNamespace

import pytest
import torch

from skyrl_train.config.utils import get_default_config
from skyrl_train.learner_memory import LearnerMemory
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.utils.importance_ratio_diagnostics import LogRatioMonitor
from skyrl_train.workers.fsdp.fsdp_worker import FSDPPolicyWorkerBase


def test_four_real_optimizer_updates_retain_age_and_increasing_stale_ratio(monkeypatch):
    class CpuPolicy(FSDPPolicyWorkerBase):
        def training_step(self, experience, global_step, local_step, accumulation_steps):
            logits = self.tiny_model(torch.ones(1, 1))
            current = torch.log_softmax(logits, dim=-1)[:, :1]
            monitor = LogRatioMonitor(torch.device("cpu"))
            monitor.add(current.detach(), experience.action_log_probs, torch.ones_like(current))
            (-current.mean() / accumulation_steps).backward()
            if (local_step + 1) % accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.update_count += 1
            return {
                **monitor.metrics(),
                "policy_loss": -current.item(),
                "response_length": 1,
                "policy_lr": 0.1,
                "policy_entropy": 0.0,
            }

    cfg = get_default_config()
    cfg.trainer.micro_train_batch_size_per_gpu = 1
    cfg.trainer.update_epochs_per_batch = 1
    cfg.trainer.policy.grug_query_bias_update_mode = "frozen"
    worker = object.__new__(CpuPolicy)
    worker.cfg = cfg
    worker._rank = 0
    worker._is_lora = False
    worker.update_count = 0
    worker.policy_mini_batch_size_per_gpu = 2
    worker.tiny_model = torch.nn.Linear(1, 2, bias=False)
    torch.nn.init.zeros_(worker.tiny_model.weight)
    worker.optimizer = torch.optim.SGD(worker.tiny_model.parameters(), lr=0.1)
    worker.model = SimpleNamespace(model=worker.tiny_model)
    worker.strategy = SimpleNamespace(is_rank_0=lambda: False, all_reduce=lambda status: status)
    worker._memory = LearnerMemory(enabled=False, rank=0, backend="fsdp2")
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    old = torch.log_softmax(worker.tiny_model(torch.ones(8, 1)), dim=-1)[:, :1].detach()
    batch = TrainingInputBatch(
        {
            key: torch.ones(8, 1)
            for key in (
                "sequences",
                "attention_mask",
                "response_mask",
                "loss_mask",
                "base_action_log_probs",
                "values",
                "returns",
                "advantages",
            )
        }
    )
    batch["action_log_probs"] = old
    batch["rollout_logprobs"] = None
    batch["rollout_age"] = torch.zeros(8, dtype=torch.int32)
    batch.metadata = {"global_step": 1, "response_length": 1}
    output = worker.ppo_train(batch)
    mean = output.metadata["train_status"]
    updates = output.metadata["train_status_by_update"]
    assert worker.update_count == mean["policy_update_steps"] == 4
    assert [row["update_age"] for row in updates] == [0, 1, 2, 3]
    assert mean["update_age_max"] == 3
    stale = [row["stale/abs_log_ratio_mean"] for row in updates]
    assert stale[0] == pytest.approx(0, abs=1e-8)
    assert stale[-1] > stale[0]
    assert stale == sorted(stale)
