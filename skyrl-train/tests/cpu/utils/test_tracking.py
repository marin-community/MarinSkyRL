from pathlib import Path
from contextlib import nullcontext

import ray
import pytest
import wandb
from omegaconf import OmegaConf

from skyrl_train.config.utils import get_default_config
from skyrl_train.utils.tracking import Tracking
from skyrl_train.entrypoints.main_base import BasePPOExp


class _SharedRun:
    def __init__(self):
        self.id = "shared-run"
        self.history = []
        self.pending = {}
        self.summary = {}
        self.axes = {}
        self.requested_steps = []
        self.state = "running"
        self.events = []
        self.finish_error = None

    def define_metric(self, name, step_metric=None):
        self.axes[name] = step_metric

    def log(self, data, *, step=None, commit=None):
        self.requested_steps.append(step)
        self.pending.update(data)
        if commit:
            self.history.append(self.pending.copy())
            self.summary.update(self.pending)
            self.pending.clear()

    def finish(self, exit_code=0):
        self.events.append("finish")
        if self.finish_error is not None:
            raise self.finish_error
        self.state = "finished" if exit_code == 0 else "failed"


def test_default_shared_wandb_run_commits_eval_and_train_metrics(monkeypatch):
    run = _SharedRun()
    monkeypatch.setattr(wandb, "init", lambda **kwargs: run)
    monkeypatch.setattr(ray, "is_initialized", lambda: False)
    config_path = Path(__file__).parents[3] / "skyrl_train/config/ppo_base_config.yaml"
    default_config = OmegaConf.load(config_path)
    commit = default_config.trainer.tracker_commit_each_step
    with Tracking("project", "run", backends="wandb", config=OmegaConf.create({})) as tracker:
        tracker.log({"eval/reward": 0.25}, step=0, commit=commit)
        tracker.log({"train/loss": 0.5}, step=1, commit=commit)

    assert run.history == [
        {"eval/reward": 0.25, "trainer/global_step": 0},
        {"train/loss": 0.5, "trainer/global_step": 1},
    ]
    assert run.requested_steps == [None, None]
    assert run.axes["*"] == "trainer/global_step"
    assert run.summary["eval/reward"] == 0.25
    assert run.summary["train/loss"] == 0.5
    assert run.state == "finished"


class _Trainer:
    def __init__(self, tracker, failure, events):
        self.tracker = tracker
        self.failure = failure
        self.events = events

    async def train(self):
        self.tracker.log({"train/loss": 0.5}, step=1, commit=True)
        if self.failure is not None:
            raise self.failure

    async def shutdown(self):
        self.events.append("shutdown")


class _Experiment(BasePPOExp):
    def __init__(self, events, *, setup_failure=None, training_failure=None):
        self.setup_failure = setup_failure
        self.training_failure = training_failure
        self.cfg = get_default_config()
        self.cfg.trainer.project_name = "project"
        self.cfg.trainer.run_name = "run"
        self.cfg.trainer.logger = "wandb"
        self.trainer = None
        self.startup_timings = {}
        self.events = events

    def _setup_trainer(self, tracker):
        if self.setup_failure is not None:
            raise self.setup_failure
        self.trainer = _Trainer(tracker, self.training_failure, self.events)
        return self.trainer


@pytest.mark.parametrize(
    ("setup_failure", "training_failure", "finish_failure", "expected_state", "expected_events"),
    [
        (None, None, None, "finished", ["shutdown", "finish"]),
        (RuntimeError("setup failed"), None, None, "failed", ["finish"]),
        (None, RuntimeError("training failed"), None, "failed", ["shutdown", "finish"]),
        (
            None,
            RuntimeError("training failed"),
            BrokenPipeError("tracker disconnected"),
            "running",
            ["shutdown", "finish"],
        ),
    ],
)
def test_experiment_finalizes_wandb_with_workload_status(
    monkeypatch, setup_failure, training_failure, finish_failure, expected_state, expected_events
):
    run = _SharedRun()
    monkeypatch.setattr(wandb, "init", lambda **kwargs: run)
    monkeypatch.setattr(ray, "is_initialized", lambda: False)
    run.finish_error = finish_failure
    failure = setup_failure or training_failure
    experiment = _Experiment(run.events, setup_failure=setup_failure, training_failure=training_failure)
    outcome = pytest.raises(RuntimeError) if failure is not None else nullcontext()
    with outcome as error:
        experiment.run()

    assert (error.value if error is not None else None) is failure
    assert run.state == expected_state
    assert run.events == expected_events
