from pathlib import Path

import ray
import wandb
from omegaconf import OmegaConf
import pytest

from skyrl_train.utils import tracking
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


@pytest.mark.parametrize("backends", ["wandb", ["wandb", "console"]])
def test_default_shared_wandb_run_commits_eval_and_train_metrics(monkeypatch, generated_recipe_schema, backends):
    root = Path(__file__).resolve().parents[4]
    assert Path(tracking.__file__).resolve() == root / "skyrl-train/skyrl_train/utils/tracking.py"
    run = _SharedRun()
    monkeypatch.setattr(wandb, "init", lambda **kwargs: run)
    monkeypatch.setattr(ray, "is_initialized", lambda: False)
    config_path = Path(__file__).parents[3] / "skyrl_train/config/ppo_base_config.yaml"
    default_config = OmegaConf.load(config_path)
    commit = default_config.trainer.tracker_commit_each_step
    recipe_type, base = generated_recipe_schema
    recipe = recipe_type.from_document({"trainer": {"logger": backends}})
    with pytest.raises(ValueError):
        recipe.with_settings(['trainer.logger=["unknown-logging-backend"]'])
    authored_config = OmegaConf.merge(base, recipe.to_skyrl())
    tracker = Tracking("project", "run", backends=authored_config.trainer.logger, config=OmegaConf.create({}))

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

    console_recipe = recipe.with_settings(['trainer.logger=["console"]'])
    console_config = OmegaConf.merge(base, console_recipe.to_skyrl())
    console_tracker = Tracking("project", "run", backends=console_config.trainer.logger, config=OmegaConf.create({}))
    console_tracker.log({"train/loss": 0.75}, step=2, commit=commit)
    assert run.history == [
        {"eval/reward": 0.25, "trainer/global_step": 0},
        {"train/loss": 0.5, "trainer/global_step": 1},
    ]
    assert run.summary["train/loss"] == 0.5
