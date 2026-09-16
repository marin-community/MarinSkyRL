from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import web

from skyrl_train.trajectory_runners.model_clients import DirectModelClient, OpenAIHTTPModelClient


@pytest.mark.asyncio
async def test_direct_model_client_preserves_engine_tokens():
    engine_output = {
        "responses": ["answer"],
        "response_ids": [[3, 4]],
        "stop_reasons": ["stop"],
        "response_logprobs": [[-0.1, -0.2]],
        "prompt_logprobs": None,
    }
    engine = AsyncMock()
    engine.generate.return_value = engine_output

    output = await DirectModelClient(engine).generate({"prompt_token_ids": [[1, 2]]})

    assert output == {**engine_output, "token_provenance": "engine"}


@pytest.mark.asyncio
async def test_direct_model_client_uses_vllm_chat_rendering_for_row_request_options():
    engine = AsyncMock()
    engine.model_name = "snowball"
    engine.tokenizer = MagicMock()
    engine.tokenizer.decode.return_value = "<tool-call tokens>"
    engine.tokenize.return_value = {"tokens": [11, 12, 13]}
    engine.chat_completion.return_value = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-2",
                            "type": "function",
                            "function": {"name": "search", "arguments": '{"query":"x"}'},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
                "token_ids": [21, 22],
                "logprobs": {"content": [{"logprob": -0.1}, {"logprob": -0.2}]},
            }
        ]
    }
    client = DirectModelClient(engine)

    output = await client.generate(
        {
            "prompts": [[{"role": "user", "content": "look it up"}]],
            "session_ids": ["trajectory-2"],
            "sampling_params": {"temperature": 0.7, "logprobs": 0},
            "chat_completion_params": [
                {
                    "tools": [
                        {
                            "type": "function",
                            "name": "search",
                            "description": "Search",
                            "parameters": {"type": "object"},
                            "strict": True,
                        }
                    ],
                    "parallel_tool_calls": False,
                    "max_output_tokens": 128,
                }
            ],
        }
    )

    expected_tools = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search",
                "parameters": {"type": "object"},
                "strict": True,
            },
        }
    ]
    tokenize_body = engine.tokenize.await_args.args[0]["json"]
    assert tokenize_body == {
        "model": "snowball",
        "messages": [{"role": "user", "content": "look it up"}],
        "tools": expected_tools,
        "add_generation_prompt": True,
    }
    chat_body = engine.chat_completion.await_args.args[0]["json"]
    assert chat_body == {
        "model": "snowball",
        "messages": [{"role": "user", "content": "look it up"}],
        "session_id": "trajectory-2",
        "temperature": 0.7,
        "tools": expected_tools,
        "parallel_tool_calls": False,
        "max_completion_tokens": 128,
        "return_token_ids": True,
        "logprobs": True,
    }
    assert output["prompt_ids"] == [[11, 12, 13]]
    assert output["response_ids"] == [[21, 22]]
    assert output["response_logprobs"] == [[-0.1, -0.2]]
    assert output["assistant_messages"] == [engine.chat_completion.return_value["choices"][0]["message"]]
    assert output["token_provenance"] == "engine"


def test_direct_model_client_omits_empty_tools_from_vllm_request():
    options = DirectModelClient._chat_options({"tools": [], "temperature": 0.4}, {})

    assert options == {"temperature": 0.4}


@pytest.mark.asyncio
async def test_direct_chat_client_captures_exact_student_topk_ids():
    engine = AsyncMock()
    engine.model_name = "teacher"
    engine.tokenizer = MagicMock()
    engine.tokenizer.decode.return_value = "answer"
    engine.tokenize.return_value = {"tokens": [1, 2]}
    engine.chat_completion.return_value = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "answer"},
                "finish_reason": "stop",
                "token_ids": [9],
                "logprobs": {
                    "content": [
                        {
                            "logprob": -7.0,
                            "top_logprobs": [
                                {"token": "token_id:9", "logprob": -7.0},
                                {"token": "token_id:3", "logprob": -0.2},
                                {"token": "token_id:2", "logprob": -0.1},
                            ],
                        }
                    ]
                },
            }
        ]
    }

    output = await DirectModelClient(engine).generate(
        {
            "prompts": [[{"role": "user", "content": "question"}]],
            "session_ids": ["test"],
            "sampling_params": {"logprobs": 2},
            "chat_completion_params": [{}],
        }
    )

    body = engine.chat_completion.await_args.args[0]["json"]
    assert body["top_logprobs"] == 3
    assert body["return_tokens_as_token_ids"] is True
    assert output["student_topk_indices"] == [[[2, 3]]]
    assert output["behavior_topk_logprobs"] == [[[-0.1, -0.2]]]


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
                "sampling_params": {"temperature": 0.7, "max_generate_length": 256},
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
            "max_completion_tokens": 256,
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


@pytest.mark.asyncio
async def test_http_model_client_preserves_exact_chat_tokens_and_logprobs():
    requests = []

    async def tokenize(request):
        requests.append(("tokenize", await request.json()))
        return web.json_response({"tokens": [11, 12]})

    async def complete(request):
        requests.append(("complete", await request.json()))
        return web.json_response(
            {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "answer"},
                        "finish_reason": "stop",
                        "token_ids": [21, 22],
                        "logprobs": {"content": [{"logprob": -0.1}, {"logprob": -0.2}]},
                    }
                ]
            }
        )

    app = web.Application()
    app.router.add_post("/tokenize", tokenize)
    app.router.add_post("/v1/chat/completions", complete)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]

    try:
        client = OpenAIHTTPModelClient(base_url=f"http://127.0.0.1:{port}", model_name="policy", tokenizer=MagicMock())
        output = await client.generate(
            {
                "prompts": [[{"role": "user", "content": "question"}]],
                "session_ids": ["trajectory-1"],
                "sampling_params": {"temperature": 0.7, "max_generate_length": 256, "logprobs": 0},
                "chat_completion_params": [{}],
            }
        )
    finally:
        await runner.cleanup()

    assert requests == [
        (
            "tokenize",
            {
                "model": "policy",
                "messages": [{"role": "user", "content": "question"}],
                "add_generation_prompt": True,
            },
        ),
        (
            "complete",
            {
                "model": "policy",
                "messages": [{"role": "user", "content": "question"}],
                "session_id": "trajectory-1",
                "temperature": 0.7,
                "max_completion_tokens": 256,
                "return_token_ids": True,
                "logprobs": True,
            },
        ),
    ]
    assert output["prompt_ids"] == [[11, 12]]
    assert output["response_ids"] == [[21, 22]]
    assert output["response_logprobs"] == [[-0.1, -0.2]]
    assert output["token_provenance"] == "engine"


@pytest.mark.asyncio
async def test_http_model_client_rejects_exact_chat_response_without_token_ids():
    async def tokenize(_request):
        return web.json_response({"tokens": [11, 12]})

    async def complete(_request):
        return web.json_response(
            {"choices": [{"message": {"role": "assistant", "content": "answer"}, "finish_reason": "stop"}]}
        )

    app = web.Application()
    app.router.add_post("/tokenize", tokenize)
    app.router.add_post("/v1/chat/completions", complete)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]

    try:
        client = OpenAIHTTPModelClient(base_url=f"http://127.0.0.1:{port}", model_name="policy", tokenizer=MagicMock())
        with pytest.raises(RuntimeError, match="did not return exact response token IDs"):
            await client.generate(
                {
                    "prompts": [[{"role": "user", "content": "question"}]],
                    "sampling_params": {"logprobs": 0},
                    "chat_completion_params": [{}],
                }
            )
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_http_model_client_preserves_server_error_details():
    async def reject(_request):
        return web.json_response({"error": {"message": "unsupported field"}}, status=400)

    app = web.Application()
    app.router.add_post("/v1/chat/completions", reject)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]

    try:
        client = OpenAIHTTPModelClient(base_url=f"http://127.0.0.1:{port}", model_name="policy", tokenizer=MagicMock())
        with pytest.raises(RuntimeError, match="HTTP 400.*unsupported field"):
            await client.generate({"prompts": [[{"role": "user", "content": "question"}]]})
    finally:
        await runner.cleanup()
