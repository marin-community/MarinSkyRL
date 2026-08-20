"""Every per-step ``Timer`` in the trainers records into ``all_timings``, and every exemption is live.

``Timer(name)`` without a dict logs one loguru line and nothing else -- the duration never reaches
``timing/<name>`` in the step payload, and so never reaches wandb, the ``WANDB_MIRROR`` line or
anything downstream. Only ``Timer(name, self.all_timings)`` records it, and no runtime assertion can
see the missing argument, so these checks read the sources.

``STARTUP_ONLY`` names spans that run once, before the step loop. ``all_timings`` is cleared after
every step, so recording one there would report startup cost as part of step 1; those need a startup
payload instead. The exemption check runs over both trainers at once because one exempt span exists
only in the async trainer.
"""

import ast
from pathlib import Path

import pytest

TRAINER_SOURCES = ("trainer.py", "fully_async_trainer.py")

STARTUP_ONLY = frozenset({"init_weight_sync_state", "load_checkpoints", "sync_weights_to_inference_engines"})

_SKYRL_TRAIN = Path(__file__).resolve().parents[2] / "skyrl_train"


def _skyrl_timer_calls(source: Path):
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


def _unrouted_spans(source: Path) -> list[tuple[str, int]]:
    """The ``Timer("name")`` spans in one module that pass no dict, as ``(name, line)``."""
    return [
        (call.args[0].value, call.lineno)
        for call in _skyrl_timer_calls(source)
        if len(call.args) == 1 and isinstance(call.args[0], ast.Constant)
    ]


@pytest.mark.parametrize("filename", TRAINER_SOURCES)
def test_per_step_timers_record_into_all_timings(filename):
    leaked = [(name, line) for name, line in _unrouted_spans(_SKYRL_TRAIN / filename) if name not in STARTUP_ONLY]
    assert not leaked, f"{filename}: per-step spans not recorded into all_timings: {leaked}"


def test_every_startup_exemption_is_still_used():
    exempt_in_source = {name for filename in TRAINER_SOURCES for name, _ in _unrouted_spans(_SKYRL_TRAIN / filename)}
    unused = STARTUP_ONLY - exempt_in_source
    assert not unused, f"exemptions no longer matching any span: {sorted(unused)}"
