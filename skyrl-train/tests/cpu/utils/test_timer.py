"""`Timer` reports one duration per block, measured on a monotonic clock.

Run with:
uv run --frozen pytest tests/cpu/utils/test_timer.py
"""

from collections.abc import Iterable

import pytest
from loguru import logger

import skyrl_train.utils.utils as timer_module
from skyrl_train.utils import Timer


class FakeClock:
    """A stand-in for the `time` module, installed only on the module `Timer` lives in.

    Patching `time.monotonic` globally would also retime the asyncio event loop, which
    schedules off the same clock. Scoping the patch to one module leaves every other
    caller on the real clock, so each tick here is consumed by `Timer` and nothing else.
    `monotonic` and `time` advance independently, which is what lets a test step the wall
    clock backwards without disturbing the monotonic one.
    """

    def __init__(self, *monotonic_ticks: float, wall_clock_ticks: Iterable[float] = ()):
        self._monotonic_ticks = list(monotonic_ticks)
        self._wall_clock_ticks = list(wall_clock_ticks)

    def monotonic(self) -> float:
        assert self._monotonic_ticks, "Timer read the monotonic clock more times than the test scripted"
        return self._monotonic_ticks.pop(0)

    def time(self) -> float:
        assert self._wall_clock_ticks, "Timer measured a duration on the wall clock"
        return self._wall_clock_ticks.pop(0)


def test_accumulated_duration_comes_from_a_single_clock_read(monkeypatch):
    """The duration `Timer` accumulates is the one it measured, not a later re-read.

    Reading the clock once for the log line and again for `update_dict` charges the
    accumulator for the logging call on top of the block itself. The third tick is far
    enough away that a second read reports 800.0 rather than 1.0.
    """
    monkeypatch.setattr(timer_module, "time", FakeClock(100.0, 101.0, 900.0))

    timings = {}
    with Timer("generate", timings):
        pass

    assert timings["generate"] == 1.0


def test_repeated_blocks_accumulate_under_one_key(monkeypatch):
    monkeypatch.setattr(timer_module, "time", FakeClock(10.0, 12.0, 100.0, 103.0))

    timings = {}
    for _ in range(2):
        with Timer("generate", timings):
            pass

    assert timings == {"generate": 5.0}


def test_failed_block_is_not_logged_as_finished(monkeypatch):
    records = []
    monkeypatch.setattr(timer_module, "time", FakeClock(10.0, 12.0))
    sink = logger.add(lambda message: records.append(message.record))
    try:
        with pytest.raises(RuntimeError, match="checkpoint failed"):
            with Timer("save_checkpoints"):
                raise RuntimeError("checkpoint failed")
    finally:
        logger.remove(sink)

    assert [record["level"].name for record in records] == ["INFO", "ERROR"]


def test_duration_ignores_a_backwards_wall_clock_step(monkeypatch):
    """An NTP correction during a block must not produce a negative duration.

    `timing/*` values are read as fractions of `timing/step`, so a negative duration
    corrupts the phase breakdown rather than merely looking odd.
    """
    monkeypatch.setattr(timer_module, "time", FakeClock(100.0, 101.0, wall_clock_ticks=(5000.0, 1000.0, 1000.0)))

    timings = {}
    with Timer("generate", timings):
        pass

    assert timings["generate"] == 1.0


@pytest.mark.asyncio
async def test_async_block_accumulates_a_single_measured_duration(monkeypatch):
    monkeypatch.setattr(timer_module, "time", FakeClock(100.0, 101.0, 900.0))

    timings = {}
    async with Timer("generate", timings):
        pass

    assert timings["generate"] == 1.0
