from unittest.mock import AsyncMock

import pytest
from aiohttp import web

from skyrl_train.trajectory_runners.model_clients import DirectModelClient, OpenAIHTTPModelClient


@pytest.mark.asyncio
@pytest.mark.parametrize("engine_indices", [None, [7]])
async def test_direct_model_client_preserves_engine_tokens(engine_indices):
    engine_output = {
        "responses": ["answer"],
        "response_ids": [[3, 4]],
        "stop_reasons": ["stop"],
        "response_logprobs": [[-0.1, -0.2]],
        "prompt_logprobs": None,
    }
    if engine_indices is not None:
        engine_output["generator_engine_indices"] = engine_indices
    engine = AsyncMock()
    engine.generate.return_value = engine_output

    output = await DirectModelClient(engine).generate({"prompt_token_ids": [[1, 2]]})

    assert output == {**engine_output, "token_provenance": "engine"}


@pytest.mark.asyncio
async def test_http_model_client_normalizes_chat_completion():
    requests = []

    async def complete(request):
        requests.append(await request.json())
        return web.json_response({"choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}]})

    app = web.Application()
    app.router.add_post("/v1/chat/completions", complete)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            assert text == "answer"
            assert add_special_tokens is False
            return [7, 8]

    try:
        client = OpenAIHTTPModelClient(base_url=f"http://127.0.0.1:{port}", model_name="policy", tokenizer=Tokenizer())
        output = await client.generate(
            {
                "prompts": [[{"role": "user", "content": "question"}]],
                "session_ids": ["trajectory-1"],
                "sampling_params": {"temperature": 0.7},
            }
        )
    finally:
        await runner.cleanup()

    assert requests == [
        {
            "model": "policy",
            "messages": [{"role": "user", "content": "question"}],
            "session_id": "trajectory-1",
            "temperature": 0.7,
        }
    ]
    assert output == {
        "responses": ["answer"],
        "response_ids": [[7, 8]],
        "stop_reasons": ["stop"],
        "response_logprobs": None,
        "prompt_logprobs": None,
        "token_provenance": "reconstructed",
    }
