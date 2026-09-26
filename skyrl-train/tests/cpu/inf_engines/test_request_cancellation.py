import asyncio
import threading

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


class BlockingRemoteMethod(RemoteMethod):
    def __init__(self, result):
        super().__init__(result)
        self.started = threading.Event()
        self.release = threading.Event()

    def remote(self, *args, **kwargs):
        self.started.set()
        assert self.release.wait(timeout=2)
        return self.result


class InferenceActor:
    def __init__(self, chat_completion_result=None, stream_result=None, tokenize_result=None, generate_result=None):
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
@pytest.mark.parametrize(
    "actor_result_kwarg, engine_method, payload",
    [
        ("chat_completion_result", "chat_completion", {"json": {}}),
        ("tokenize_result", "tokenize", {"json": {"messages": []}}),
        ("generate_result", "generate", {"prompt_token_ids": [[1, 2, 3]], "sampling_params": {}}),
    ],
)
async def test_cancelled_request_cancels_ray_actor_task(
    record_ray_cancellation, actor_result_kwarg, engine_method, payload
):
    reference = PendingReference()
    engine = RayWrappedInferenceEngine(InferenceActor(**{actor_result_kwarg: reference}))

    task = asyncio.create_task(getattr(engine, engine_method)(payload))
    await reference.started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert reference.cancelled


@pytest.mark.asyncio
async def test_slow_ray_submission_does_not_block_bridge_loop_and_cancels_abandoned_work(monkeypatch):
    reference = ResolvedReference({"tokens": [1]})
    method = BlockingRemoteMethod(reference)
    actor = InferenceActor(tokenize_result=reference)
    actor.tokenize = method
    engine = RayWrappedInferenceEngine(actor)
    cancelled = threading.Event()
    monkeypatch.setattr(
        "skyrl_train.inference_engines.ray_wrapped_inference_engine.ray.cancel", lambda _: cancelled.set()
    )

    started_at = asyncio.get_running_loop().time()
    task = asyncio.create_task(engine.tokenize({"json": {"messages": []}}))
    try:
        assert await asyncio.wait_for(asyncio.to_thread(method.started.wait, 1), timeout=1)
        assert asyncio.get_running_loop().time() - started_at < 0.5
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        method.release.set()

    assert await asyncio.wait_for(asyncio.to_thread(cancelled.wait, 1), timeout=1)


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
