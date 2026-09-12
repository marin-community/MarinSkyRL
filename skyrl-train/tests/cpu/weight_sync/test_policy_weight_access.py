import asyncio
import tempfile
from concurrent.futures import ThreadPoolExecutor
from threading import Event, get_ident

import pytest
import ray

from skyrl_train.weight_sync.policy_weight_access import PolicyWeightAccess


def test_session_ownership_crosses_threads_without_releasing_another_session():
    access = PolicyWeightAccess()
    released = Event()
    entered = Event()
    state = {}

    def begin():
        state.update(token=access.acquire("weight-sync"), thread=get_ident())
        entered.set()
        assert released.wait(5)

    def finish():
        assert entered.wait(5)
        assert state["thread"] != get_ident()
        with pytest.raises(RuntimeError, match="already owned"):
            with access.hold("ppo"):
                pytest.fail("PPO entered the frozen interval")
        with pytest.raises(RuntimeError, match="does not match"):
            access.release("wrong-session")
        assert access.owner == "weight-sync"
        access.release(state["token"])
        with access.hold("ppo"):
            with pytest.raises(RuntimeError, match="does not match"):
                access.release(state["token"])
        released.set()

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(begin)
        second = pool.submit(finish)
        second.result(timeout=10)
        first.result(timeout=10)
    assert access.owner is None


@ray.remote(num_cpus=0)
class SessionActor:
    def __init__(self):
        self.access = PolicyWeightAccess()
        self.started = asyncio.Event()
        self.closed = asyncio.Event()

    async def start(self):
        token = self.access.acquire("weight-sync")
        self.started.set()
        try:
            await asyncio.Event().wait()
        finally:
            self.access.release(token)
            self.closed.set()

    async def wait_started(self):
        await self.started.wait()
        return self.access.owner

    async def ppo(self):
        with self.access.hold("ppo"):
            return "entered"

    async def wait_closed(self):
        await self.closed.wait()
        return self.access.owner


@pytest.mark.asyncio
async def test_actual_ray_cancellation_releases_session_after_rejecting_ppo():
    assert not ray.is_initialized()
    with tempfile.TemporaryDirectory(prefix="ray-weight-lease-") as runtime_dir:
        ray.init(address="local", num_cpus=2, include_dashboard=False, _temp_dir=runtime_dir)
        actor = SessionActor.remote()
        try:
            pending = actor.start.remote()
            assert await asyncio.wait_for(actor.wait_started.remote(), 15) == "weight-sync"
            with pytest.raises(ray.exceptions.RayTaskError, match="already owned"):
                await actor.ppo.remote()
            ray.cancel(pending)
            assert await asyncio.wait_for(actor.wait_closed.remote(), 5) is None
            assert await actor.ppo.remote() == "entered"
        finally:
            ray.kill(actor)
            ray.shutdown()
