import asyncio
import socket

import httpx
import pytest
import uvicorn
from omegaconf import OmegaConf

from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
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

    def tokenize(self, _request):
        raise AssertionError("tokenization is not expected in this test")


class _Tokenizer:
    def apply_chat_template(
        self,
        messages,
        *,
        add_generation_prompt,
        chat_template,
        continue_final_message,
        tokenize,
        tools=None,
        **template_kwargs,
    ):
        assert not tokenize
        return ":".join(
            str(value)
            for value in (
                len(messages),
                sum(len(message["content"]) for message in messages),
                int(add_generation_prompt),
                int(continue_final_message),
                int(chat_template == "custom-template"),
                int(template_kwargs["enable_thinking"]),
                len(tools or []),
            )
        )

    def encode(self, prompt, *, add_special_tokens):
        assert not add_special_tokens
        return [int(value) for value in prompt.split(":")]


def _tokenizing_backend() -> InferenceEngineClient:
    config = OmegaConf.create(
        {
            "trainer": {"policy": {"model": {"path": TEST_MODEL_NAME}}},
            "generator": {
                "backend": "vllm",
                "enable_http_endpoint": False,
                "http_endpoint_host": "127.0.0.1",
                "http_endpoint_port": 0,
                "engine_init_kwargs": {},
            },
        }
    )
    return InferenceEngineClient(engines=[], tokenizer=_Tokenizer(), full_config=config)


async def _wait_until_started(server: uvicorn.Server) -> None:
    while not server.started:
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
@pytest.mark.parametrize("add_generation_prompt", [False, True])
async def test_tokenize_renders_chat_with_inference_client_tokenizer(add_generation_prompt):
    set_global_state(_tokenizing_backend(), None)
    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/tokenize",
            json={
                "model": TEST_MODEL_NAME,
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "continue"},
                ],
                "add_generation_prompt": add_generation_prompt,
                "continue_final_message": False,
                "chat_template": "custom-template",
                "chat_template_kwargs": {"enable_thinking": True},
                "tools": [{"type": "function", "function": {"name": "shell"}}],
            },
        )

    assert response.status_code == 200
    assert response.json() == {"tokens": [2, 14, int(add_generation_prompt), 0, 1, 1, 1], "count": 7}


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
