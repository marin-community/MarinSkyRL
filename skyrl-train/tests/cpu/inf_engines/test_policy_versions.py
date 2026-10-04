import asyncio
from types import SimpleNamespace

import pytest

from skyrl_train.inference_engines.vllm.policy_versions import PolicyVersionRecorder


@pytest.mark.asyncio
async def test_resumed_request_keeps_old_versions_when_frontend_processing_lags():
    """A pause acknowledgement alone must not relabel buffered old tokens."""
    queue = asyncio.Queue()
    processor = SimpleNamespace(
        request_states={"internal": SimpleNamespace(external_req_id="chat", request_index=0)},
        process_outputs=lambda *args: None,
    )
    engine = SimpleNamespace(
        engine_core=SimpleNamespace(outputs_queue=queue, get_output_async=queue.get),
        output_processor=processor,
        output_handler=None,
    )
    recorder = PolicyVersionRecorder(engine)
    await recorder.install(3)
    processing = asyncio.Event()
    release_old = asyncio.Event()

    async def handler():
        while True:
            tokens = await engine.engine_core.get_output_async()
            if tokens == [10, 11]:
                # Model vLLM splitting a large output into processing chunks.
                processing.set()
                await release_old.wait()
            processor.process_outputs([SimpleNamespace(request_id="internal", new_token_ids=tokens)])

    engine.output_handler = asyncio.create_task(handler())
    try:
        queue.put_nowait([10, 11])
        await processing.wait()
        installing = asyncio.create_task(recorder.install(4))
        await asyncio.sleep(0)
        assert not installing.done()
        release_old.set()
        await installing
        queue.put_nowait([12])
        # A second publication waits for the newly emitted token too.
        await recorder.install(5)
        assert recorder.take("chat", 0, [10, 11, 12]) == [3, 3, 4]
    finally:
        engine.output_handler.cancel()
        with pytest.raises(asyncio.CancelledError):
            await engine.output_handler


@pytest.mark.asyncio
async def test_token_versions_refuse_misaligned_response_and_unknown_initial_weights():
    queue = asyncio.Queue()
    engine = SimpleNamespace(
        engine_core=SimpleNamespace(outputs_queue=queue, get_output_async=queue.get),
        output_processor=SimpleNamespace(
            request_states={"raw": SimpleNamespace(external_req_id="chat", request_index=0)},
            process_outputs=lambda *args: None,
        ),
        output_handler=None,
    )
    recorder = PolicyVersionRecorder(engine)
    output = SimpleNamespace(request_id="raw", new_token_ids=[7, 8, 9])
    with pytest.raises(RuntimeError, match="before its policy version"):
        engine.output_processor.process_outputs([output])
    await recorder.install(0)
    engine.output_processor.process_outputs([output])
    with pytest.raises(ValueError, match="does not match"):
        recorder.take("chat", 0, [7, 4])
    engine.output_processor.process_outputs([output])
    assert recorder.take("chat", 0, [7, 8]) == [0, 0]
    with pytest.raises(ValueError, match="no generating-policy trace"):
        recorder.take("chat", 0, [7])
