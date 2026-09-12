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


@pytest.mark.asyncio
async def test_physical_counter_delay_is_outside_complete_native_install_rpc(monkeypatch):
    from types import SimpleNamespace
    from skyrl_train.weight_sync import shard_interval

    clock = [0.0]
    events = []
    monkeypatch.setattr(shard_interval.time, "perf_counter", lambda: clock[0])

    async def call(method, rank, manifest, publication):
        phase = {
            "begin_shard_publication": "frozen",
            "verify_shard_publication": "verified",
            "begin_shard_stream": "frozen",
            "run_shard_publication": "installed",
            "run_shard_stream": "installed",
            "close_shard_publication": "closed",
            "close_shard_stream": "closed",
        }[method]
        if phase == "installed":
            assert events.count("before") == 1 and "after" not in events
            await asyncio.sleep(0)
            clock[0] += 3 if rank == 0 else 4
            events.append(method)
        return {"rank": rank, "manifest_id": manifest, "publication_id": publication, "phase": phase}

    class Client:
        generation_paused_event = asyncio.Event()

        async def begin_shard_stream(self, *args):
            return await call("begin_shard_stream", 1, *args)

        async def run_shard_stream(self, *args):
            return await call("run_shard_stream", 1, *args)

        async def close_shard_stream(self, *args):
            return await call("close_shard_stream", 1, *args)

    async def observe(moment):
        if moment == "after":
            assert set(events[1:]) == {"run_shard_publication", "run_shard_stream"}
        clock[0] += 1009
        events.append(moment)
        return {"endpoint": moment}

    async def replay(manifest, publication):
        assert events[-1] == "after"
        return {
            "rank": 1,
            "manifest_id": manifest,
            "publication_id": publication,
            "phase": "verified",
            "mismatches": 0,
            "coverage": 1.0,
            "compared_bytes": 16,
        }

    client = Client()
    client.generation_paused_event.set()
    driver = SimpleNamespace(
        inference_engine_client=client,
        policy_model=SimpleNamespace(async_run_ray_method=lambda dispatch, method, *args: [call(method, 0, *args)]),
    )
    result = await shard_interval.run_shard_interval(
        driver,
        "manifest",
        1,
        replay=replay,
        policy_ranks=[0],
        receiver_ranks=[1],
        expected_receiver_bytes={1: 16},
        generation_boundary="driver",
        observe=observe,
    )
    assert result["phase_seconds"]["install"] == 7
    assert result["phase_seconds"]["observation_before"] == 1009
    assert result["phase_seconds"]["observation_after"] == 1009
    assert result["physical_before"] == {"endpoint": "before"}
    assert result["physical_after"] == {"endpoint": "after"}
    assert clock[0] == 2025
