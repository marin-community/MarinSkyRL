import asyncio

import pytest

from skyrl_train.weight_sync.shard_interval import settled


@pytest.mark.asyncio
@pytest.mark.parametrize("cancellations", [1, 2, 3])
async def test_cancel_waits_for_every_native_call_before_cleanup(cancellations):
    entered = asyncio.Event()
    finish = asyncio.Event()
    completed = []

    async def native():
        entered.set()
        await finish.wait()
        completed.append("native")

    task = asyncio.create_task(settled(native()))
    await entered.wait()
    try:
        for _ in range(cancellations):
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done() and not completed
    finally:
        finish.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert completed == ["native"]


@pytest.mark.asyncio
async def test_primary_error_waits_for_other_calls_and_retains_their_failures():
    finish = asyncio.Event()
    entered = asyncio.Event()
    first = ValueError("initiating")

    async def primary():
        entered.set()
        raise first

    async def secondary():
        await finish.wait()
        raise RuntimeError("secondary")

    task = asyncio.create_task(settled(primary(), secondary()))
    await entered.wait()
    assert not task.done()
    finish.set()
    with pytest.raises(ValueError) as caught:
        await task
    assert caught.value is first
    assert any("secondary" in note for note in first.__notes__)
