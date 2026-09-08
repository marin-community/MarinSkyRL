import asyncio
import tempfile

import pytest
import ray

from skyrl_train.inference_engines.ray_wrapped_inference_engine import RayWrappedInferenceEngine


class PendingReference:
    def __init__(self):
        self.started = asyncio.Event()
        self.cancelled = False

    def __await__(self):
        return self._wait().__await__()

    async def _wait(self):
        self.started.set()
        await asyncio.Event().wait()


class PendingReferenceGenerator:
    def __init__(self):
        self.started = asyncio.Event()
        self.cancelled = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        self.started.set()
        await asyncio.Event().wait()


class RemoteMethod:
    def __init__(self, result):
        self.result = result

    def remote(self, request_payload):
        return self.result


class InferenceActor:
    def __init__(self, chat_completion_result, stream_result=None):
        self.chat_completion = RemoteMethod(chat_completion_result)
        self.chat_completion_stream = RemoteMethod(stream_result)


@pytest.fixture
def record_ray_cancellation(monkeypatch):
    def cancel(reference):
        reference.cancelled = True

    monkeypatch.setattr("skyrl_train.inference_engines.ray_wrapped_inference_engine.ray.cancel", cancel)


@pytest.mark.asyncio
async def test_cancelled_chat_completion_cancels_ray_actor_task(record_ray_cancellation):
    reference = PendingReference()
    engine = RayWrappedInferenceEngine(InferenceActor(reference))

    request = asyncio.create_task(engine.chat_completion({"json": {}}))
    await reference.started.wait()
    request.cancel()

    with pytest.raises(asyncio.CancelledError):
        await request
    assert reference.cancelled


@pytest.mark.asyncio
async def test_abandoned_chat_stream_cancels_ray_actor_task(record_ray_cancellation):
    reference = PendingReferenceGenerator()
    engine = RayWrappedInferenceEngine(InferenceActor(PendingReference(), reference))

    async def consume_stream():
        return [chunk async for chunk in engine.chat_completion_stream({"json": {"stream": True}})]

    request = asyncio.create_task(consume_stream())
    await reference.started.wait()
    request.cancel()

    with pytest.raises(asyncio.CancelledError):
        await request
    assert reference.cancelled


@ray.remote(num_cpus=0)
class PendingGenerationActor:
    def __init__(self):
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.active = False

    async def generate(self, input_batch):
        self.active = True
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.active = False
            self.cancelled.set()
            raise

    async def wait_started(self):
        await self.started.wait()
        return self.active

    async def wait_cancelled(self):
        await self.cancelled.wait()
        return self.active


@pytest.fixture
def local_ray():
    assert not ray.is_initialized(), "Cancellation test must own its local Ray runtime"
    with tempfile.TemporaryDirectory(prefix="ray-cancel-") as runtime_dir:
        ray.init(address="local", num_cpus=2, include_dashboard=False, _temp_dir=runtime_dir)
        try:
            yield
        finally:
            ray.shutdown()


@pytest.mark.asyncio
async def test_cancelled_token_generation_stops_actual_ray_actor(local_ray):
    actor = PendingGenerationActor.remote()
    engine = RayWrappedInferenceEngine(actor)
    try:
        request = asyncio.create_task(engine.generate({"prompt_token_ids": [[1, 2, 3]]}))
        assert await asyncio.wait_for(actor.wait_started.remote(), timeout=15)
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        # Remote readiness and cancellation acknowledgements prove the boundary,
        # without assuming that cancellation of a local ObjectRef cancels its actor.
        assert not await asyncio.wait_for(actor.wait_cancelled.remote(), timeout=2)
    finally:
        ray.kill(actor)
