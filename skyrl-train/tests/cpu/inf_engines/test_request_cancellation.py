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

    def remote(self, *args, **kwargs):
        return self.result


class InferenceActor:
    def __init__(self, chat_completion_result, stream_result=None, tokenize_result=None, generate_result=None):
        self.chat_completion = RemoteMethod(chat_completion_result)
        self.chat_completion_stream = RemoteMethod(stream_result)
        self.tokenize = RemoteMethod(tokenize_result)
        self.generate = RemoteMethod(generate_result)


class ResolvedReference:
    def __init__(self, value):
        self.value = value

    def __await__(self):
        async def resolve():
            return self.value

        return resolve().__await__()


class RecordingRemoteMethod:
    def __init__(self, name, calls):
        self.name = name
        self.calls = calls

    def remote(self, *args):
        self.calls.append((self.name, args))
        return ResolvedReference({"method": self.name})


class OnlineEagleActor:
    def __init__(self):
        self.calls = []
        for name in (
            "begin_online_eagle_capture",
            "seal_online_eagle_capture",
            "update_draft_weights",
        ):
            setattr(self, name, RecordingRemoteMethod(name, self.calls))


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
async def test_cancelled_tokenization_cancels_ray_actor_task(record_ray_cancellation):
    reference = PendingReference()
    engine = RayWrappedInferenceEngine(InferenceActor(PendingReference(), tokenize_result=reference))

    request = asyncio.create_task(engine.tokenize({"json": {"messages": []}}))
    await reference.started.wait()
    request.cancel()

    with pytest.raises(asyncio.CancelledError):
        await request
    assert reference.cancelled


@pytest.mark.asyncio
async def test_cancelled_generate_cancels_ray_actor_task(record_ray_cancellation):
    reference = PendingReference()
    engine = RayWrappedInferenceEngine(InferenceActor(PendingReference(), generate_result=reference))

    request = asyncio.create_task(engine.generate({"prompt_token_ids": [[1, 2, 3]], "sampling_params": {}}))
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


@pytest.mark.asyncio
async def test_online_eagle_methods_cross_the_ray_actor_boundary() -> None:
    actor = OnlineEagleActor()
    engine = RayWrappedInferenceEngine(actor)

    await engine.begin_online_eagle_capture({"step": 3})
    await engine.seal_online_eagle_capture("s3://bucket/captures/step-3")
    await engine.update_draft_weights("s3://bucket/drafts/draft-3/model.safetensors")

    assert actor.calls == [
        ("begin_online_eagle_capture", ({"step": 3},)),
        ("seal_online_eagle_capture", ("s3://bucket/captures/step-3",)),
        ("update_draft_weights", ("s3://bucket/drafts/draft-3/model.safetensors",)),
    ]
