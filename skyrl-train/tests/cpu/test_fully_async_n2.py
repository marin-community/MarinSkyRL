"""Actual async driver N2 preparation, publication and resume boundaries."""

import asyncio
from copy import deepcopy
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from skyrl_train.config.utils import get_default_config
from skyrl_train.fully_async_trainer import FullyAsyncRayPPOTrainer, _GenerationQueues
from skyrl_train.data_order import source_order_checkpoint, validate_source_order_checkpoint
from skyrl_train.utils.policy_losses import ppo_policy_loss
from skyrl_train.utils.trainer_utils import ResumeMode
from tests.cpu.test_fully_async_publication_cadence import (
    DriverWithCpuLearner,
    FirstTokenRunner,
    TimedInferenceService,
    make_driver,
)
from tests.cpu.test_fully_async_staleness import _batch_assembly_state, _generated_group


class N2CpuDriver(DriverWithCpuLearner):
    """Replace only model forward/update I/O, preserving preparation and driver."""

    def fwd_logprobs_values_reward(self, data):
        self.preparations.append((self.global_step, list(data.metadata["uids"])))
        data["action_log_probs"] = torch.full_like(data["response_mask"], -self.global_step / 10, dtype=torch.float)
        data["base_action_log_probs"] = None
        data["values"] = None
        return data

    def train_critic_and_policy(self, data):
        self.inputs.append(deepcopy(data))
        self.policy_model.completed_update += 1
        self._successful_policy_updates = self.policy_model.completed_update
        self.all_metrics["policy/updates_completed"] = self._successful_policy_updates
        return {
            "learner/update": self.policy_model.completed_update,
            "policy_successful_update_steps_valid": 1,
            "policy_successful_update_steps": 1,
        }


def n2_driver(driver_type=N2CpuDriver, **kwargs):
    trainer = make_driver(
        interval=1,
        age=0,
        steps=4,
        runner_type=FirstTokenRunner,
        first_token_admission=None,
        minibatches=2,
        driver_type=driver_type,
        **kwargs,
    )
    engine = TimedInferenceService()
    trainer.inference_engine_client = engine
    trainer.trajectory_runner.engine = engine
    trainer.policy_model.actor_infos = [SimpleNamespace(rank=SimpleNamespace(dp_size=1))]
    trainer.preparations, trainer.inputs = [], []
    return trainer


@pytest.mark.asyncio
async def test_actual_n2_a0_prepares_once_per_cohort_and_publishes_each_update(monkeypatch):
    events = []
    monkeypatch.setattr(
        "skyrl_train.fully_async_trainer.record_event",
        lambda name, body, **kwargs: events.append((name, body, kwargs)),
    )
    trainer = n2_driver()
    try:
        await asyncio.wait_for(trainer._train_loop(), timeout=15)
    finally:
        await trainer._cancel_trajectory_tasks()
    assert [step for step, _ in trainer.preparations] == [1, 3]
    assert trainer.inference_engine_client.publications == [0, 1, 2, 3, 4]
    assert trainer.data_tracker.total_samples_consumed == 8
    assert len(trainer.inputs) == 4
    for index, data in enumerate(trainer.inputs):
        assert data["action_log_probs"].unique().tolist() == pytest.approx([-(1 + 2 * (index // 2)) / 10])
        assert data["rollout_age"].unique().tolist() == [index % 2]
        assert data.metadata["async_cohort_update_index"] == index % 2
    uids = [uid for data in trainer.inputs for uid in data.metadata["uids"][::2]]
    assert len(set(uids)) == len(uids) == 8
    ages = [body for name, body, _ in events if name == "cohort_consumption"]
    assert len(ages) == 8
    assert all(body["admission_age"] == 0 and body["consume_age"] == body["within_cohort_lag"] for body in ages)
    prepared = [body for name, body, _ in events if name == "cohort_prepared"]
    assert [body["admission_step"] for body in prepared] == [1, 3]
    assert all(body["groups"] == 4 and body["sequences"] == 8 and body["updates"] == 2 for body in prepared)


@pytest.mark.asyncio
async def test_actual_n2_resume_second_partition_keeps_prepared_old_weights(tmp_path):
    trainer = n2_driver(stop_step=1, save_step=1)
    trainer.cfg.trainer.ckpt_path = str(tmp_path)
    try:
        await asyncio.wait_for(trainer._train_loop(), timeout=15)
    finally:
        await trainer._cancel_trajectory_tasks()
    resumed = n2_driver()
    resumed.resume_mode = ResumeMode.LATEST
    resumed.cfg.trainer.resume_path = str(tmp_path / "global_step_1")
    artifact = tmp_path / "global_step_1" / "generation_buffer_state.pt"
    original = artifact.read_bytes()
    broken = torch.load(artifact, weights_only=False)
    broken.pop("prepared_cohort")
    torch.save(broken, artifact)
    resumed.global_step = 2
    queues = _GenerationQueues(asyncio.Queue(maxsize=64), asyncio.Queue(), asyncio.Condition())
    try:
        with pytest.raises(ValueError, match="requires the saved prepared cohort"):
            resumed._restore_buffer_from_checkpoint(queues, resumed.cfg.trainer.resume_path)
        assert queues.snapshot().pending_uids() == set()
        assert resumed.async_train_dataloader._pending_uids == set()
    finally:
        artifact.write_bytes(original)
    try:
        await asyncio.wait_for(resumed._train_loop(), timeout=15)
    finally:
        await resumed._cancel_trajectory_tasks()
    assert [step for step, _ in resumed.preparations] == [3]
    assert resumed.inputs[0]["action_log_probs"].unique().tolist() == pytest.approx([-0.1])
    assert resumed.inputs[0]["rollout_age"].unique().tolist() == [1]
    assert resumed.inference_engine_client.publications == [1, 2, 3, 4]
    assert resumed.data_tracker.total_samples_consumed == 8


@pytest.mark.asyncio
async def test_n2_cohort_admission_rejects_stale_source_before_preparation():
    trainer, queues = _batch_assembly_state(mini_batch_size=2, accepted=5)
    trainer.cohort_size = 4
    trainer.updates_per_cohort = 2
    queues.completed.put_nowait(_generated_group("stale", earliest_model_step=7))
    for index in range(4):
        queues.completed.put_nowait(_generated_group(str(index), earliest_model_step=8))
    groups = await asyncio.wait_for(trainer._get_admitted_generation_group_mini_batch(queues), timeout=1)
    assert [group.uid for group in groups] == ["0", "1", "2", "3"]
    assert queues.retries.get_nowait() == [{"uid": "stale"}]
    assert trainer._staleness_manager._stat.accepted == 4
    assert trainer.global_step == 10


@pytest.mark.asyncio
async def test_unsuccessful_optimizer_does_not_advance_n2_cursor_or_publish(monkeypatch):
    trainer = n2_driver()
    monkeypatch.setattr(
        trainer,
        "train_critic_and_policy",
        lambda data: {
            "policy_successful_update_steps_valid": 1,
            "policy_successful_update_steps": 0,
        },
    )
    try:
        with pytest.raises(RuntimeError, match="exactly one successful"):
            await asyncio.wait_for(trainer._train_loop(), timeout=15)
    finally:
        await trainer._cancel_trajectory_tasks()
    assert trainer.global_step == 1
    assert trainer._generation_queues.prepared_cohort.next_update == 0
    assert trainer.inference_engine_client.publications == [0]
    assert trainer.data_tracker.total_samples_consumed == 0


@pytest.mark.parametrize(
    "path,value",
    [
        ("trainer.algorithm.loss_reduction", "seq_mean_token_sum_norm_global"),
        ("trainer.update_epochs_per_batch", 2),
        ("trainer.algorithm.use_kl_loss", True),
        ("trainer.algorithm.dynamic_sampling.type", "filter"),
        ("trainer.strategy", "fsdp2"),
    ],
)
def test_unsupported_n2_numerical_geometry_rejects_before_model_initialization(path, value):
    cfg = get_default_config()
    cfg.trainer.train_batch_size = 128
    cfg.trainer.policy_mini_batch_size = 64
    cfg.trainer.strategy = "megatron"
    cfg.trainer.algorithm.use_kl_loss = False
    OmegaConf.update(cfg, path, value)
    with pytest.raises(ValueError, match="async N2 currently requires"):
        FullyAsyncRayPPOTrainer(cfg=cfg)


class DifferentiableN2Driver(N2CpuDriver):
    """A real CPU parameter/AdamW boundary with production GRPO and REG loss."""

    def __init__(self, *args, **kwargs):
        kwargs["cfg"].data.epoch_seeded_shuffle = True
        kwargs["cfg"].trainer.algorithm.policy_loss_type = "regular"
        OmegaConf.update(kwargs["cfg"], "trainer.algorithm.max_seq_len", 1, force_add=True)
        super().__init__(*args, **kwargs)
        self.parameter = torch.nn.Parameter(torch.tensor(0.15, dtype=torch.float64))
        self.optimizer = torch.optim.AdamW([self.parameter], lr=0.01)
        self.gradients = []

    def logprobs(self, data):
        features = (torch.arange(len(data)) % 2 * 2 - 1).to(torch.float64).unsqueeze(1)
        return torch.nn.functional.logsigmoid(self.parameter * features).expand_as(data["response_mask"])

    def fwd_logprobs_values_reward(self, data):
        self.preparations.append((self.global_step, list(data.metadata["uids"])))
        data["action_log_probs"] = self.logprobs(data).detach()
        data["base_action_log_probs"] = None
        data["values"] = None
        return data

    def train_critic_and_policy(self, data):
        self.optimizer.zero_grad()
        loss, _ = ppo_policy_loss(
            self.logprobs(data),
            data["action_log_probs"],
            data["advantages"],
            config=self.cfg.trainer.algorithm,
            loss_mask=data["loss_mask"],
            rollout_logprobs=data["rollout_logprobs"],
        )
        loss.backward()
        self.gradients.append(self.parameter.grad.detach().clone())
        self.optimizer.step()
        return super().train_critic_and_policy(data)

    def save_checkpoints(self):
        super().save_checkpoints()
        torch.save(
            {
                "parameter": self.parameter.detach(),
                "optimizer": self.optimizer.state_dict(),
                "source_order": source_order_checkpoint(self.train_dataloader, self.global_step),
            },
            Path(self.cfg.trainer.ckpt_path) / f"global_step_{self.global_step}" / "cpu-optimizer.pt",
        )

    def load_checkpoints(self):
        step, checkpoint = super().load_checkpoints()
        saved = torch.load(Path(checkpoint) / "cpu-optimizer.pt", weights_only=False)
        with torch.no_grad():
            self.parameter.copy_(saved["parameter"])
        self.optimizer.load_state_dict(saved["optimizer"])
        validate_source_order_checkpoint(self.train_dataloader, saved["source_order"], step)
        return step, checkpoint


@pytest.mark.asyncio
async def test_real_loss_optimizer_resume_matches_uninterrupted_parameters_and_state(tmp_path):
    full = n2_driver(driver_type=DifferentiableN2Driver)
    partial = n2_driver(driver_type=DifferentiableN2Driver, stop_step=1, save_step=1)
    partial.cfg.trainer.ckpt_path = str(tmp_path)
    for trainer in (full, partial):
        try:
            await asyncio.wait_for(trainer._train_loop(), timeout=15)
        finally:
            await trainer._cancel_trajectory_tasks()
    resumed = n2_driver(driver_type=DifferentiableN2Driver)
    resumed.resume_mode = ResumeMode.LATEST
    resumed.cfg.trainer.resume_path = str(tmp_path / "global_step_1")
    try:
        await asyncio.wait_for(resumed._train_loop(), timeout=15)
    finally:
        await resumed._cancel_trajectory_tasks()
    assert [step for step, _ in resumed.preparations] == [3]
    assert torch.equal(full.parameter, resumed.parameter)
    assert not torch.equal(full.parameter, torch.tensor(0.15, dtype=torch.float64))
    assert len(full.gradients) == 4 and all(torch.isfinite(g) and abs(g) > 1e-8 for g in full.gradients)
    full_state, resumed_state = full.optimizer.state_dict(), resumed.optimizer.state_dict()
    assert full_state["param_groups"] == resumed_state["param_groups"]
    for key, value in full_state["state"][0].items():
        assert torch.equal(value, resumed_state["state"][0][key])
    expected = [data.metadata["uids"] for data in full.inputs]
    actual = [data.metadata["uids"] for data in [*partial.inputs, *resumed.inputs]]
    assert actual == expected
    assert full.data_tracker.total_samples_consumed == resumed.data_tracker.total_samples_consumed == 8
