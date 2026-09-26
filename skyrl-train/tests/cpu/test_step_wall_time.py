"""Exclusive async optimizer-step accounting and its boundaries."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from omegaconf import OmegaConf

from skyrl_train.fully_async_trainer import FullyAsyncRayPPOTrainer
from skyrl_train.timing_observability import STEP_WALL_PHASES, StepWallTime, phase_timing_observations
from skyrl_train.trainer import RayPPOTrainer


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def _wall(clock):
    return StepWallTime({phase: 1.0 for phase in STEP_WALL_PHASES}, clock=clock)


def test_ordinary_step_is_exclusive_and_reports_each_budget_and_overrun():
    clock = Clock()
    wall = _wall(clock)
    wall.start("group_admission")
    clock.advance(2)
    wall.start("batch_assembly")
    clock.advance(3)
    wall.start("training_preparation")
    clock.advance(4)
    wall.start("advantages")
    clock.advance(5)
    wall.start("policy_training")
    clock.advance(6)
    wall.start("group_bookkeeping")
    clock.advance(1)
    wall.start("weight_sync")
    clock.advance(2)
    wall.start("step_end_bookkeeping")
    clock.advance(1)

    metrics = wall.finish(25)

    assert sum(metrics[f"timing/step_wall/{phase}"] for phase in STEP_WALL_PHASES) == pytest.approx(25)
    assert metrics["timing/step_wall/unaccounted"] == 1
    assert metrics["timing/step_wall_overrun/unaccounted"] == 0
    assert metrics["timing/step_wall_overrun/policy_training"] == 5
    assert {
        key.removeprefix("timing/step_wall_budget/") for key in metrics if key.startswith("timing/step_wall_budget/")
    } == set(STEP_WALL_PHASES)


def test_step_end_checkpoint_and_evaluation_are_distinct_exclusive_phases():
    clock = Clock()
    wall = _wall(clock)
    control = SimpleNamespace(should_save=True, should_save_hf_model=True, should_evaluate=True, reset=lambda: None)
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer._control = control
    trainer.eval_dataset = object()
    trainer.all_timings = {}
    trainer.all_metrics = {}

    async def callback(event, _state, current_control, **_kwargs):
        clock.advance(1)
        return current_control

    async def save(_state):
        clock.advance(2)

    async def drain():
        clock.advance(3)
        return True

    async def evaluate():
        clock.advance(4)
        return {"eval/score": 0.5}

    trainer.callback_handler = SimpleNamespace(call_event_async=callback)
    trainer._save_intermediate_checkpoint = save
    trainer._drain_checkpoint_upload = drain
    trainer.handle_hf_export = lambda: clock.advance(1)
    trainer.eval = evaluate
    wall.start("step_end_bookkeeping")

    asyncio.run(trainer._run_step_end_callbacks(SimpleNamespace(), step_wall=wall))
    metrics = wall.finish(clock.now)

    assert metrics["timing/step_wall/checkpoint_work"] == 6
    assert metrics["timing/step_wall/evaluation"] == 5
    assert metrics["timing/step_wall/step_end_bookkeeping"] == 1
    assert sum(metrics[f"timing/step_wall/{phase}"] for phase in STEP_WALL_PHASES) == 12


def test_background_upload_elapsed_is_not_a_step_child_but_blocking_wait_is():
    observations = phase_timing_observations({"step": 10, "checkpoint_upload": 100, "checkpoint_upload_blocking": 2})
    by_name = {item.name: item for item in observations}
    assert by_name["checkpoint_upload"].root == "checkpoint_upload"
    assert by_name["checkpoint_upload_blocking"].parent == "step"


def test_checkpoint_drain_separates_background_elapsed_from_trainer_wait():
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.all_timings = {}

    async def drain():
        upload = asyncio.get_running_loop().create_future()
        upload.set_result((100.0, 0.0))
        trainer._pending_checkpoint_upload = (upload, SimpleNamespace())
        return await trainer._drain_checkpoint_upload()

    assert asyncio.run(drain())
    assert trainer.all_timings["checkpoint_upload"] == 100
    assert 0 <= trainer.all_timings["checkpoint_upload_blocking"] < 1


def test_pretraining_evaluation_is_startup_only():
    trainer = FullyAsyncRayPPOTrainer.__new__(FullyAsyncRayPPOTrainer)
    trainer.global_step = 0
    trainer.all_timings = {}
    trainer.cfg = OmegaConf.create({"trainer": {"tracker_commit_each_step": True}})
    trainer.eval = AsyncMock(return_value={"eval/score": 0.5})
    trainer.tracker = SimpleNamespace(log=lambda metrics, **kwargs: logged.append((metrics, kwargs)))
    trainer._log_metrics_stdout = lambda metrics, **kwargs: mirrored.append((metrics, kwargs))
    logged, mirrored = [], []

    asyncio.run(trainer._run_pretraining_evaluation())

    assert trainer.all_timings == {}
    assert logged[0] == ({"eval/score": 0.5}, {"step": 0, "commit": True})
    assert logged[1][0]["startup/eval_before_train"] >= 0
    assert logged[1][1] == {"step": 0, "commit": False}
    assert mirrored[1][1]["kind"] == "startup"


def test_invalid_budgets_and_excess_wall_time_are_rejected():
    with pytest.raises(ValueError, match="exactly"):
        StepWallTime({"policy_training": 1})
    clock = Clock()
    wall = _wall(clock)
    wall.start("policy_training")
    clock.advance(2)
    with pytest.raises(ValueError, match="exceed"):
        wall.finish(1)
