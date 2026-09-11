import asyncio
import socket

import httpx
import pytest
import uvicorn
from skyrl_train.inference_engines.inference_engine_client_http_endpoint import create_app, set_global_state
from skyrl_train.inference_engines.vllm.stats import HTTPBridgeStatsAccumulator, IntervalReadMode


TEST_MODEL_NAME = "test-model"


class _Backend:
    model_name = TEST_MODEL_NAME

    async def chat_completion(self, _request):
        return {"choices": [{"message": {"content": "ok"}}]}

    async def completion(self, _request):
        return {"choices": [{"text": "ok"}]}

    async def chat_completion_stream(self, _request):
        yield "data: [DONE]\n\n"

    async def tokenize(self, _request):
        raise AssertionError("tokenization is not expected in this test")


class _NativeTokenizationBackend(_Backend):
    def __init__(self):
        self.request_payload = None

    async def tokenize(self, request_payload):
        self.request_payload = request_payload
        return {
            "tokens": [41, 42, 43],
            "count": 3,
            "max_model_len": 32768,
            "token_strs": ["openai", "content", "parts"],
        }


class _RejectingTokenizationBackend(_Backend):
    async def tokenize(self, _request_payload):
        return {
            "error": {
                "message": "invalid tokenize request",
                "type": "Bad Request",
                "code": 400,
            }
        }


async def _wait_until_started(server: uvicorn.Server) -> None:
    while not server.started:
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_tokenize_preserves_harbor_request_and_native_vllm_response():
    backend = _NativeTokenizationBackend()
    set_global_state(backend, None)
    transport = httpx.ASGITransport(app=create_app())
    request_json = {
        "model": TEST_MODEL_NAME,
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": [{"type": "text", "text": "continue"}]},
        ],
        "add_generation_prompt": False,
        "continue_final_message": True,
        "chat_template": "custom-template",
        "chat_template_kwargs": {"enable_thinking": True},
        "media_io_kwargs": {"image": {"num_crops": 4}},
        "tools": [{"type": "function", "function": {"name": "shell"}}],
        "return_token_strs": True,
    }
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/tokenize", json=request_json)

    assert response.status_code == 200
    assert response.json() == {
        "tokens": [41, 42, 43],
        "count": 3,
        "max_model_len": 32768,
        "token_strs": ["openai", "content", "parts"],
    }
    assert backend.request_payload["json"] == request_json


@pytest.mark.asyncio
async def test_tokenize_preserves_native_vllm_validation_error():
    set_global_state(_RejectingTokenizationBackend(), None)
    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/tokenize", json={"messages": "not-a-message-list"})

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "message": "invalid tokenize request",
            "type": "Bad Request",
            "code": 400,
        }
    }


@pytest.mark.asyncio
async def test_real_uvicorn_bridge_records_96_concurrent_requests():
    accumulator = HTTPBridgeStatsAccumulator()
    set_global_state(_Backend(), None)
    app = create_app(accumulator, event_loop_lag_interval_seconds=0.001)
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="on"))
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    server_task = asyncio.create_task(server.serve(sockets=[listener]))
    try:
        await asyncio.wait_for(_wait_until_started(server), timeout=5)
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=10) as client:
            responses = await asyncio.gather(
                *(
                    client.post(
                        "/v1/chat/completions",
                        json={"model": "test-model", "messages": [{"role": "user", "content": str(index)}]},
                    )
                    for index in range(96)
                )
            )
        assert all(response.status_code == 200 for response in responses)

        snapshot = accumulator.snapshot(IntervalReadMode.PEEK)
        assert snapshot.response_bytes.count == 96
        assert snapshot.json_serialization_seconds.count == 96
        assert snapshot.event_loop_lag_seconds.count > 0
    finally:
        server.should_exit = True
        await asyncio.wait_for(server_task, timeout=5)
        listener.close()
