"""Keep actual optimizer-boundary diagnostics through worker and driver logging."""

from types import SimpleNamespace

import pytest
import torch
from rigging.telemetry.serialization import EventBody, event_fields

from skyrl_train.config.utils import get_default_config
from skyrl_train.learner_memory import LearnerMemory
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.workers.fsdp.fsdp_worker import FSDPPolicyWorkerBase
from skyrl_train.utils.importance_ratio_diagnostics import LogRatioMonitor
from skyrl_train.utils.gradient_direction import GradientDirectionTracker


@pytest.mark.parametrize("position_window", [256, 128])
@pytest.mark.parametrize("grad_enabled", [False, True])
def test_optimizer_statuses_survive_worker_mean_and_driver_logging(monkeypatch, position_window, grad_enabled):
    class CpuPolicy(FSDPPolicyWorkerBase):
        def training_step(self, experience, global_step, local_step, accumulation_steps):
            assert experience.rollout_age is not None
            gradients = {}
            if (local_step + 1) % accumulation_steps == 0:
                self.update_count += 1
                if grad_enabled:
                    gradients = self.grad_tracker.observe([torch.tensor([float(self.update_count)])])
            monitor = LogRatioMonitor(torch.device("cpu"), position_window=position_window)
            monitor.add(torch.ones(1, 600), torch.zeros(1, 600), torch.ones(1, 600))
            return {
                **monitor.metrics(),
                **gradients,
                "optimizer_step_succeeded": 1.0,
                "log_ratio_abs_mean": float(self.update_count),
                "policy_loss": 0.5,
                "raw_grad_norm": float(self.update_count + 10),
                "ppo_clip_ratio": self.update_count / 10,
                "response_length": 1,
                "policy_lr": 1e-6,
                "policy_entropy": 0.0,
            }

    cfg = get_default_config()
    cfg.trainer.algorithm.ratio_diagnostics.position_window = position_window
    cfg.trainer.micro_train_batch_size_per_gpu = 1
    cfg.trainer.update_epochs_per_batch = 2
    cfg.trainer.policy.grug_query_bias_update_mode = "frozen"
    worker = object.__new__(CpuPolicy)
    worker.cfg = cfg
    worker._rank = 0
    worker._is_lora = False
    worker.update_count = 0
    worker.grad_tracker = GradientDirectionTracker("gpu_fp32", torch.device("cpu"))
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
    if grad_enabled:
        assert [row["grad_cosine_valid"] for row in updates] == [0, 1, 1, 1]
        assert output.metadata["train_status"]["grad_cosine_min"] == 1
        assert output.metadata["train_status"]["grad_cosine_max"] == 1
    assert len(updates[0]) > 64
    with pytest.raises(ValueError, match="at most 64 fields"):
        event_fields(EventBody(updates[0]), budget=100_000)

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

    def validated_event(*args, **kwargs):
        fields = {key: value for key, value in args[1].items() if value is not None}
        assert event_fields(EventBody(fields), budget=100_000) == fields
        assert len(fields) == (37 if grad_enabled else 32)
        if grad_enabled:
            for key in ("grad_cosine", "grad_cosine_valid", "grad_norm_reduced", "grad_norm_valid", "grad_dot"):
                assert fields[key] == updates[fields["update_index"]][key]
        assert fields["raw_grad_norm"] == updates[fields["update_index"]]["raw_grad_norm"]
        assert fields["ppo_clip_ratio"] == updates[fields["update_index"]]["ppo_clip_ratio"]
        assert fields[f"stale/pos_first{position_window}/selected_tokens"] > 0
        assert fields[f"stale/pos_last{position_window}/selected_tokens"] > 0
        events.append((args, kwargs))

    monkeypatch.setattr("skyrl_train.trainer.record_event", validated_event)
    mean = trainer.train_critic_and_policy(batch)
    assert mean["log_ratio_abs_mean"] == pytest.approx(2)
    assert trainer.all_metrics["policy/updates_attempted"] == 28
    assert trainer.all_metrics["policy/updates_completed"] == 4
    assert trainer.all_metrics["policy/updates_completed_valid"] == 1
    assert [trainer.all_metrics[f"policy/by_update/{k}/update_age"] for k in range(4)] == [0, 1, 2, 3]
    assert [event[0][1]["update_age"] for event in events] == [0, 1, 2, 3]
    assert all(event[0][0] == "policy_update" and event[1]["attributes"]["step"] == "7" for event in events)
