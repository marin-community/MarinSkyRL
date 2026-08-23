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
from pathlib import Path
from typing import NamedTuple
from unittest.mock import Mock

import pytest

from skyrl_train.trainer import RayPPOTrainer

TRAINER_SOURCES = ("trainer.py", "fully_async_trainer.py")

# Spans that run once, before the step loop.
STARTUP_SPANS = frozenset(
    {
        "init_weight_sync_state",
        "load_checkpoints",
        "sync_policy_for_rollouts",
        "sync_weights_to_inference_engines",
    }
)

_SKYRL_TRAIN = Path(__file__).resolve().parents[2] / "skyrl_train"


def _timer_calls(source: Path):
    """Every ``Timer(...)`` call in one module, excluding ``threading.Timer``."""
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
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
        if not isinstance(call.args[0], ast.Constant):
            continue
        destination = ast.unparse(call.args[1]) if len(call.args) > 1 else None
        spans.append(_Span(call.args[0].value, destination, call.lineno))
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


def _trainer_with_startup_timings(timings):
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.all_startup_timings = timings
    trainer.global_step = 0
    trainer.tracker = Mock()
    trainer._log_metrics_stdout = Mock()
    return trainer


def test_startup_payload_is_published_under_its_own_prefix_and_kind():
    trainer = _trainer_with_startup_timings({"load_checkpoints": 12.5, "init_weight_sync_state": 3.0})

    trainer._log_startup_timings()

    expected = {"startup/load_checkpoints": 12.5, "startup/init_weight_sync_state": 3.0}
    trainer.tracker.log.assert_called_once_with(expected, step=0)
    # gate.py parses `kind` off the mirror line and reads only `train`; a startup payload
    # must not arrive labelled as a training step.
    payload, kwargs = trainer._log_metrics_stdout.call_args
    assert payload[0] == expected
    assert kwargs["kind"] == "startup"


def test_nothing_is_published_when_no_startup_span_ran():
    trainer = _trainer_with_startup_timings({})

    trainer._log_startup_timings()

    trainer.tracker.log.assert_not_called()
    trainer._log_metrics_stdout.assert_not_called()
