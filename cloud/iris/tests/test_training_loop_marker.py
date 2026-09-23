from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cloud.iris.rl_config_translation import (  # noqa: E402
    RLEntrypoint,
    inert_fully_async_settings,
    training_loop_for_entrypoint,
)
from marinskyrl.environment_contract import TrainingLoop  # noqa: E402

COLOCATED = {"trainer": {"placement": {"colocate_all": True}}}
SPLIT = {"trainer": {"placement": {"colocate_all": False}}}


@pytest.mark.parametrize(
    ("entrypoint", "raw", "overrides", "expected"),
    [
        (RLEntrypoint.FULLY_ASYNC, {}, (), TrainingLoop.ASYNC),
        (RLEntrypoint.SYNC, {}, (), TrainingLoop.SYNC),
        (RLEntrypoint.MINI_SWE, {}, (), TrainingLoop.SYNC),
        (RLEntrypoint.TERMINAL_BENCH, SPLIT, (), TrainingLoop.ASYNC),
        (RLEntrypoint.TERMINAL_BENCH, COLOCATED, (), TrainingLoop.SYNC),
        (RLEntrypoint.TERMINAL_BENCH, {}, (), TrainingLoop.SYNC),
        (RLEntrypoint.GENERATE, {}, (), None),
        (RLEntrypoint.TERMINAL_BENCH_GENERATE, SPLIT, (), None),
    ],
)
def test_the_loop_follows_the_trainer_the_entrypoint_builds(entrypoint, raw, overrides, expected):
    assert training_loop_for_entrypoint(entrypoint, raw, overrides) is expected


def test_fully_async_keys_are_inert_only_under_a_synchronous_trainer():
    raw = {"trainer": {"fully_async": {"max_staleness_steps": 2, "pause_mode": "keep"}}}
    assert inert_fully_async_settings(raw, RLEntrypoint.SYNC) == (
        "trainer.fully_async.max_staleness_steps",
        "trainer.fully_async.pause_mode",
    )
    assert inert_fully_async_settings(raw, RLEntrypoint.FULLY_ASYNC) == ()
    assert inert_fully_async_settings({**raw, **SPLIT}, RLEntrypoint.TERMINAL_BENCH) == ()
