import io

import pytest
from loguru import logger

from ci.marin_nightly.gate import StepMetrics, parse_metrics
from skyrl_train.trainer import RayPPOTrainer


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
