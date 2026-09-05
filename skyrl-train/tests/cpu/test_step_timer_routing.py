import io
from unittest.mock import Mock

import pytest
import wandb
from loguru import logger
from omegaconf import OmegaConf

from ci.marin_nightly.gate import StepMetrics, parse_metrics
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.utils.tracking import Tracking


def test_shared_wandb_history_retains_optimizer_steps(monkeypatch, tmp_path):
    # Exercise the real shared-mode SDK, replacing only its service transport and Ray discovery.
    run = wandb.sdk.wandb_run.Run(settings=wandb.Settings(mode="shared", run_id="test", root_dir=str(tmp_path)))
    transport = Mock()
    run._interface = transport
    monkeypatch.setattr(wandb, "init", lambda **_kwargs: run)
    monkeypatch.setattr("ray.is_initialized", lambda: False)
    tracker = Tracking("test", "test", backends="wandb", config=OmegaConf.create({}))
    payload = {"reward/mean": 0.25}
    try:
        tracker.log({"eval/all/avg_score": 0.1}, step=0, commit=True)
        tracker.log(payload, step=1, commit=True)
        tracker.log({"reward/mean": 0.5}, step=2, commit=True)
        tracker.log({"eval/all/avg_score": 0.3}, step=2, commit=True)
        history = transport.publish_partial_history.call_args_list
        assert [call.args[1] for call in history] == [
            {"eval/all/avg_score": 0.1, "global_step": 0},
            {"reward/mean": 0.25, "global_step": 1},
            {"reward/mean": 0.5, "global_step": 2},
            {"eval/all/avg_score": 0.3, "global_step": 2},
        ]
        assert all(call.kwargs["flush"] is True and call.kwargs["step"] is None for call in history)
        definitions = [call.args[0] for call in transport._publish_metric.call_args_list]
        assert any(metric.glob_name == "*" and metric.step_metric == "global_step" for metric in definitions)
        assert payload == {"reward/mean": 0.25}
    finally:
        tracker.logger.clear()


@pytest.fixture
def mirror_lines():
    """Loguru writes to its own sinks, so pytest's caplog does not see the mirror line."""
    stream = io.StringIO()
    sink_id = logger.add(stream, format="{message}")
    try:
        yield stream
    finally:
        logger.remove(sink_id)


class _RecordingTracker:
    """Captures what the trainer publishes, in place of a live wandb run."""

    def __init__(self):
        self.calls = []

    def log(self, data, step, commit):
        self.calls.append((data, step, commit))


def _trainer(startup_timings, step_timings=None):
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.all_startup_timings = startup_timings
    trainer.all_timings = step_timings if step_timings is not None else {}
    trainer.global_step = 0
    trainer.tracker = _RecordingTracker()
    return trainer


def test_the_startup_payload_reaches_the_gate_parser_as_its_own_kind(mirror_lines):
    trainer = _trainer({"load_checkpoints": 12.5, "init_weight_sync_state": 3.0})

    trainer._log_startup_timings()

    expected = {"startup/load_checkpoints": 12.5, "startup/init_weight_sync_state": 3.0}
    assert parse_metrics(mirror_lines.getvalue()) == [StepMetrics(kind="startup", step=0, values=expected)]
    assert trainer.tracker.calls == [(expected, 0, False)]


def test_telemetry_failure_keeps_the_stdout_metric_payload(mirror_lines, monkeypatch):
    trainer = _trainer({})
    trainer._training_metrics_enabled = True

    def broken_publication(*_args, **_kwargs):
        raise RuntimeError("invalid telemetry record")

    monkeypatch.setattr("skyrl_train.trainer.record_training_metrics", broken_publication)
    with pytest.raises(RuntimeError, match="invalid telemetry record"):
        trainer._log_metrics_stdout({"reward/avg_raw_reward": 0.5}, step=2)

    assert parse_metrics(mirror_lines.getvalue()) == [
        StepMetrics(kind="train", step=2, values={"reward/avg_raw_reward": 0.5})
    ]


def test_startup_work_recorded_into_the_step_payload_is_moved_out_of_it(mirror_lines):
    """The weight sync at startup writes to all_timings, and Timer accumulates into it.

    Left there, step 1's ``timing/sync_weights`` would be the startup sync plus its own.
    """
    trainer = _trainer({"load_checkpoints": 1.0}, step_timings={"sync_weights": 8.0})

    trainer._log_startup_timings()

    assert trainer.all_timings == {}
    published = parse_metrics(mirror_lines.getvalue())[0].values
    assert published["startup/sync_weights"] == 8.0
    assert not any(key.startswith("timing/") for key in published)


def test_nothing_is_published_when_no_startup_span_ran(mirror_lines):
    trainer = _trainer({})

    trainer._log_startup_timings()

    assert parse_metrics(mirror_lines.getvalue()) == []
    assert trainer.tracker.calls == []
