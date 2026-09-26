import asyncio
from types import SimpleNamespace

import pytest

from skyrl_train.rollouts.buffer import RolloutGroup
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.trajectory_runners.base import TrajectoryID
from skyrl_train.trajectory_selection import BestOfNTrajectorySelector


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
    return RolloutGroup(batch, uid, policy_step, {"uid": uid}, {})


@pytest.mark.parametrize("reason", ["initial", "training_step"])
@pytest.mark.parametrize("offload_enabled", [False, True])
def test_weight_sync_respects_optimizer_offload_policy(reason, offload_enabled):
    trainer = object.__new__(RayPPOTrainer)
    trainer.cfg = SimpleNamespace(trainer=SimpleNamespace(offload_optimizer_during_rollouts=offload_enabled))
    trainer.colocate_all = False
    trainer.global_step = 0
    trainer.all_startup_timings = {}
    trainer.all_timings = {}
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


def test_rollout_batch_conversion_reports_staleness_and_stage_timings(monkeypatch):
    trainer = object.__new__(RayPPOTrainer)
    trainer.context = SimpleNamespace(config=SimpleNamespace(batch_size=2, max_staleness_steps=2))
    trainer.cfg = SimpleNamespace(
        trainer=SimpleNamespace(algorithm=SimpleNamespace(policy_loss_type="pg", tis_lcs_alert_threshold=0.0))
    )
    trainer.global_step = 10
    trainer.all_metrics = {}
    trainer.all_timings = {}
    trainer.tokenizer = SimpleNamespace(decode=lambda response: str(response))
    now = [0.0]
    monkeypatch.setattr("skyrl_train.utils.utils.time", SimpleNamespace(monotonic=lambda: now[0]))

    def postprocess(batch, uids):
        now[0] += 7.0
        return batch

    def select(batch, uids):
        now[0] += 11.0
        return batch, uids

    def convert(batch, uids, *, rollout_staleness):
        now[0] += 3.0
        return {"rewards": batch["rewards"], "uids": uids, "rollout_staleness": rollout_staleness}

    trainer.postprocess_trajectory_batch = postprocess
    trainer.select_trajectories = select
    trainer.convert_to_training_input = convert

    result = trainer.convert_rollout_groups_to_training_input([_group("fresh", 10), _group("stale", 8)])

    assert result == {
        "rewards": [0.0, 1.0, 0.0, 1.0],
        "uids": ["fresh", "fresh", "stale", "stale"],
        "rollout_staleness": [0, 0, 2, 2],
    }
    assert trainer.all_metrics["async/staleness_max"] == 2
    assert trainer.all_metrics["async/staleness_ratio"] == 0.5
    assert trainer.all_timings == {
        "assemble_generation_group_mini_batch": 0.0,
        "postprocess_trajectory_batch": 18.0,
        "convert_to_training_input": 3.0,
    }


class _RecordingDistillationRuntime:
    def __init__(self):
        self.submitted = []

    async def submit_before_batch_assembly(self, trajectory_batch):
        self.submitted.append(trajectory_batch)
        return object()


def test_teacher_scores_only_the_rows_the_learner_selects():
    trainer = object.__new__(RayPPOTrainer)
    runtime = _RecordingDistillationRuntime()
    trainer._distillation_runtime = runtime
    trainer._distillation_tickets = {}
    trainer.trajectory_selector = BestOfNTrajectorySelector(2)

    asyncio.run(trainer._submit_admitted_groups_for_teacher_scoring([_group("best", 10, rewards=[0.25, 0.75])]))

    (submitted,) = runtime.submitted
    assert submitted["response_ids"] == [[3]]
    assert submitted["trajectory_ids"][0].repetition_id == 1
    assert set(trainer._distillation_tickets) == {"best"}
