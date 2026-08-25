"""Every ``Timer`` in the trainers records into a payload, and startup spans use the startup one.

``Timer(name)`` without a dict logs one loguru line and nothing else -- the duration never reaches
the step payload, and so never reaches wandb, the ``WANDB_MIRROR`` line or anything downstream.
Only ``Timer(name, some_dict)`` records it, and no runtime assertion can see the missing argument,
so the source checks below read the trainers.

The two payloads are not interchangeable. ``all_timings`` is cleared after every step, so a span
that runs once before the step loop must not record there: its duration would be reported as part
of step 1. Those spans record into ``all_startup_timings``, which is published once by
``_log_startup_timings``.
"""

import ast
import io
from pathlib import Path
from typing import NamedTuple

import pytest
from loguru import logger

from ci.marin_nightly.gate import StepMetrics, parse_metrics
from infra.rl_cleanup.parse_skyrl_metrics import TIMING_PARENTS
from skyrl_train.trainer import RayPPOTrainer

TRAINER_SOURCES = ("trainer.py", "fully_async_trainer.py")

# Spans that run once, before the step loop.
STARTUP_SPANS = frozenset(
    {
        "init_weight_sync_state",
        "load_checkpoints",
        "sync_weights_to_inference_engines",
    }
)

_SKYRL_TRAIN = Path(__file__).resolve().parents[2] / "skyrl_train"


def _timer_calls(source: Path):
    """Every ``Timer(...)`` call in one module, excluding ``threading.Timer``."""
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute):  # threading.Timer and friends
            continue
        if isinstance(func, ast.Name) and func.id == "Timer":
            yield node


class _Span(NamedTuple):
    name: str
    destination: str | None  # the dict the duration is recorded into; None for a bare ``Timer(name)``
    line: int


def _named_spans(source: Path) -> list[_Span]:
    """Every literally-named ``Timer`` span in one module."""
    spans = []
    for call in _timer_calls(source):
        arguments = dict(zip(("message", "update_dict"), call.args))
        arguments.update({kw.arg: kw.value for kw in call.keywords if kw.arg})
        name = arguments.get("message")
        if not isinstance(name, ast.Constant):
            continue
        destination = arguments.get("update_dict")
        spans.append(_Span(name.value, ast.unparse(destination) if destination else None, call.lineno))
    return spans


@pytest.mark.parametrize("filename", TRAINER_SOURCES)
def test_every_trainer_timer_records_into_a_payload(filename):
    leaked = [(s.name, s.line) for s in _named_spans(_SKYRL_TRAIN / filename) if s.destination is None]
    assert not leaked, f"{filename}: spans measured but recorded nowhere: {leaked}"


@pytest.mark.parametrize("filename", TRAINER_SOURCES)
def test_startup_spans_use_the_startup_payload(filename):
    misrouted = [
        s
        for s in _named_spans(_SKYRL_TRAIN / filename)
        if s.name in STARTUP_SPANS and s.destination != "self.all_startup_timings"
    ]
    assert not misrouted, f"{filename}: startup spans in the per-step payload would report as step 1: {misrouted}"


def test_every_startup_span_name_is_still_used():
    """Guard the list above against a rename leaving a startup span silently unchecked."""
    in_source = {s.name for filename in TRAINER_SOURCES for s in _named_spans(_SKYRL_TRAIN / filename)}
    unused = STARTUP_SPANS - in_source
    assert not unused, f"startup span names no longer matching any span: {sorted(unused)}"


def test_every_step_span_is_declared_to_the_step_time_analyzer():
    """An undeclared span is reported with no parent and counted on top of the step total.

    ``summarize_timing_spans`` subtracts only declared children from the step remainder, so a span
    that records into ``all_timings`` without an entry here makes the rows sum past 100%.
    """
    recorded = {
        s.name
        for filename in TRAINER_SOURCES
        for s in _named_spans(_SKYRL_TRAIN / filename)
        if s.destination == "self.all_timings"
    }
    assert not recorded - set(TIMING_PARENTS), (
        f"step spans missing from TIMING_PARENTS: {sorted(recorded - set(TIMING_PARENTS))}"
    )


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
