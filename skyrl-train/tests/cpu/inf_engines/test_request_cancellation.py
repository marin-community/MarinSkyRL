import asyncio

import pytest

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
