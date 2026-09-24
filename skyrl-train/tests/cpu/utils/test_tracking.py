from pathlib import Path

import ray
import wandb
from omegaconf import OmegaConf

from skyrl_train.utils.tracking import Tracking


class _SharedRun:
    def __init__(self):
        self.id = "shared-run"
        self.history = []
        self.pending = {}
        self.summary = {}
        self.axes = {}
        self.requested_steps = []

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
        pass


def test_default_shared_wandb_run_commits_eval_and_train_metrics(monkeypatch):
    run = _SharedRun()
    monkeypatch.setattr(wandb, "init", lambda **kwargs: run)
    monkeypatch.setattr(ray, "is_initialized", lambda: False)
    config_path = Path(__file__).parents[3] / "skyrl_train/config/ppo_base_config.yaml"
    default_config = OmegaConf.load(config_path)
    commit = default_config.trainer.tracker_commit_each_step
    tracker = Tracking("project", "run", backends="wandb", config=OmegaConf.create({}))

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
