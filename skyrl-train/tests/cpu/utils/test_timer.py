"""`Timer` reports one duration per block, measured on a monotonic clock.

Run with:
uv run --frozen pytest tests/cpu/utils/test_timer.py
"""

import time

import pytest

from skyrl_train.utils import Timer


def _clock(*ticks: float):
    """A clock returning each tick in turn, then holding the last value.

    Patching the module attribute reaches every caller, including the asyncio event loop,
    which reads `time.monotonic` to schedule. Holding the last value rather than running
    out keeps those callers working while still making an extra read inside `Timer` show
    up as a much larger duration. The timed blocks below never await, so the loop's own
    reads fall outside the window these ticks cover.
    """
    remaining = list(ticks)

    def read() -> float:
        return remaining.pop(0) if len(remaining) > 1 else remaining[0]

    return read


def test_accumulated_duration_comes_from_a_single_clock_read(monkeypatch):
    """The duration `Timer` accumulates is the one it measured, not a later re-read.

    Reading the clock once for the log line and again for `update_dict` charges the
    accumulator for the logging call on top of the block itself.
    """
    monkeypatch.setattr(time, "monotonic", _clock(100.0, 101.0, 900.0))

    timings = {}
    with Timer("generate", timings):
        pass

    assert timings["generate"] == 1.0


def test_repeated_blocks_accumulate_under_one_key(monkeypatch):
    monkeypatch.setattr(time, "monotonic", _clock(10.0, 12.0, 100.0, 103.0))

    timings = {}
    for _ in range(2):
        with Timer("generate", timings):
            pass

    assert timings == {"generate": 5.0}


def test_duration_ignores_a_backwards_wall_clock_step(monkeypatch):
    """An NTP correction during a block must not produce a negative duration.

    `timing/*` values are read as fractions of `timing/step`, so a negative duration
    corrupts the phase breakdown rather than merely looking odd.
    """
    monkeypatch.setattr(time, "monotonic", _clock(100.0, 101.0))
    monkeypatch.setattr(time, "time", _clock(5000.0, 1000.0))

    timings = {}
    with Timer("generate", timings):
        pass

    assert timings["generate"] == 1.0


@pytest.mark.asyncio
async def test_async_block_accumulates_a_single_measured_duration(monkeypatch):
    monkeypatch.setattr(time, "monotonic", _clock(100.0, 101.0, 900.0))

    timings = {}
    async with Timer("generate", timings):
        pass

    assert timings["generate"] == 1.0
