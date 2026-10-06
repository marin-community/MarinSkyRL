import asyncio
from types import SimpleNamespace

import pytest
import torch

from skyrl_train.config.utils import get_default_config
from skyrl_train.rollouts.buffer import RolloutGroup
from skyrl_train.trajectory_runners.base import TrajectoryID


def _group(uid: str, policy_step: int, *, rewards: list[float] | None = None) -> RolloutGroup:
    batch = {
        "prompt_token_ids": [[1], [1]],
        "response_ids": [[2], [3]],
        "rewards": rewards or [0.0, 1.0],
        "loss_masks": [[1], [1]],
        "stop_reasons": ["stop", "stop"],
        "rollout_metrics": {},
        "rollout_logprobs": None,
        "trajectory_ids": [TrajectoryID(instance_id=uid, repetition_id=index) for index in range(2)],
    }
    return RolloutGroup(batch, uid, policy_step, {"uid": uid})


@pytest.mark.parametrize("reason", ["initial", "training_step"])
@pytest.mark.parametrize("offload_enabled", [False, True])
def test_weight_sync_respects_optimizer_offload_policy(reason, offload_enabled, driver_trainer_factory):
    cfg = get_default_config()
    cfg.trainer.offload_optimizer_during_rollouts = offload_enabled
    trainer = driver_trainer_factory(cfg)
    events = []

    class Policy:
        optimizer_on_gpu = True

        def offload_to_cpu(self, *, offload_optimizer, offload_model):
            assert offload_optimizer and not offload_model
            self.optimizer_on_gpu = False
            events.append("offload")

    class Engine:
        async def pause_generation(self):
            events.append("pause")

        async def resume_generation(self):
            assert trainer.policy_model.optimizer_on_gpu != offload_enabled
            events.append("resume")

    async def sync_weights():
        events.append("sync")

    trainer.policy_model = Policy()
    trainer.inference_engine_client = Engine()
    trainer.sync_policy_weights_to_inference_engines = sync_weights

    asyncio.run(trainer._sync_policy_for_rollouts(reason=reason))

    assert trainer.policy_model.optimizer_on_gpu != offload_enabled
    paused = reason == "training_step"
    assert events == (["pause"] if paused else []) + (["offload"] if offload_enabled else []) + ["sync"] + (
        ["resume"] if paused else []
    )


def test_rollout_batch_conversion_reports_staleness_and_stage_timings(monkeypatch, driver_trainer_factory):
    cfg = get_default_config()
    cfg.trainer.train_batch_size = 2
    cfg.trainer.rollout_buffer.max_staleness_steps = 2
    cfg.generator.n_samples_per_prompt = 2
    cfg.trainer.algorithm.policy_loss_type = "regular"
    cfg.trainer.algorithm.off_policy_correction = "none"
    trainer = driver_trainer_factory(cfg, tokenizer=SimpleNamespace(decode=str, pad_token_id=0))
    trainer.global_step = 10
    ticks = iter((0.0, 0.0, 0.0, 18.0, 18.0, 21.0))
    monkeypatch.setattr("skyrl_train.utils.utils.time", SimpleNamespace(monotonic=lambda: next(ticks)))

    result = trainer.convert_rollout_groups_to_training_input([_group("fresh", 10), _group("stale", 8)])

    torch.testing.assert_close(result["rewards"], torch.tensor([[0.0], [1.0], [0.0], [1.0]]))
    assert result.metadata["uids"] == ["fresh", "fresh", "stale", "stale"]
    assert result["rollout_staleness"].tolist() == [0, 0, 2, 2]
    assert trainer.all_metrics["async/staleness_max"] == 2
    assert trainer.all_metrics["async/staleness_ratio"] == 0.5
    assert trainer.all_timings == {
        "assemble_generation_group_mini_batch": 0.0,
        "postprocess_trajectory_batch": 18.0,
        "convert_to_training_input": 3.0,
    }


def test_rollout_batch_conversion_records_domain_reward_metrics(driver_trainer_factory):
    cfg = get_default_config()
    cfg.trainer.train_batch_size = 3
    cfg.trainer.rollout_buffer.max_staleness_steps = 0
    cfg.trainer.algorithm.policy_loss_type = "regular"
    cfg.trainer.algorithm.off_policy_correction = "none"
    cfg.generator.n_samples_per_prompt = 2
    trainer = driver_trainer_factory(cfg, tokenizer=SimpleNamespace(decode=str, pad_token_id=0))
    math = _group("math", 0, rewards=[0.2, 0.6])
    tools = _group("tools", 0, rewards=[0.6, 1.0])
    missing = _group("missing", 0, rewards=[0.4, 0.6])
    math.trajectory_batch["data_sources"] = ["math", "math"]
    tools.trajectory_batch["data_sources"] = ["tools", "tools"]

    trainer.convert_rollout_groups_to_training_input([math, tools, missing])

    assert trainer.all_metrics["reward/domain/math/avg_raw_reward"] == pytest.approx(0.4)
    assert trainer.all_metrics["reward/domain/tools/avg_raw_reward"] == pytest.approx(0.8)
    assert trainer.all_metrics["reward/domain/_missing/avg_raw_reward"] == pytest.approx(0.5)


class _RecordingDistillationRuntime:
    domain_balancer = None

    def __init__(self):
        self.submitted = []

    async def submit_before_batch_assembly(self, trajectory_batch):
        self.submitted.append(trajectory_batch)
        return object()


def test_teacher_scores_only_the_rows_the_learner_selects(driver_trainer_factory):
    cfg = get_default_config()
    cfg.generator.n_samples_per_prompt = 2
    cfg.trainer.trajectory_selector.type = "best_of_n"
    cfg.trainer.algorithm.advantage_estimator = "uniform"
    trainer = driver_trainer_factory(cfg)
    runtime = _RecordingDistillationRuntime()
    trainer.configure_distillation(runtime)

    asyncio.run(trainer._submit_admitted_groups_for_teacher_scoring([_group("best", 10, rewards=[0.25, 0.75])]))

    (submitted,) = runtime.submitted
    assert submitted["response_ids"] == [[3]]
    assert submitted["trajectory_ids"][0].repetition_id == 1
