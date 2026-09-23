import base64
import contextlib
import io
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from aiohttp import web

from skyrl_train.trajectory_runners.model_clients import (
    DirectModelClient,
    OpenAIHTTPModelClient,
    _choice_routed_experts,
)


def _encoded_routes(rows):
    with io.BytesIO() as buffer:
        np.save(buffer, np.asarray(rows, dtype=np.uint8), allow_pickle=False)
        return base64.b64encode(buffer.getvalue()).decode("ascii")


def test_vllm_base64_routes_strip_prompt_and_pad_uncaptured_last_response_token():
    # vLLM includes prompt forwards and omits the last sampled token's forward.
    routes = _encoded_routes([[[1, 2]], [[3, 4]], [[5, 6]], [[7, 8]]])

    assert _choice_routed_experts({"routed_experts": routes}, [11, 12], [21, 22, 23]) == [
        [[5, 6]],
        [[7, 8]],
        [[0, 0]],
    ]


def test_vllm_base64_routes_reject_wrong_token_count_without_shifting_router_evidence():
    routes = _encoded_routes([[[1, 2]], [[3, 4]], [[5, 6]]])

    with pytest.raises(ValueError, match="served prompt and response token IDs"):
        _choice_routed_experts({"routed_experts": routes}, [11, 12], [21, 22, 23])


def test_vllm_base64_routes_reject_invalid_encoding_and_shape():
    with pytest.raises(ValueError, match="valid base64 .npy"):
        _choice_routed_experts({"routed_experts": "not-base64"}, [11], [21])
    with pytest.raises(ValueError, match="served prompt and response token IDs"):
        _choice_routed_experts({"routed_experts": _encoded_routes([[1, 2]])}, [11], [21])


@pytest.mark.asyncio
async def test_direct_chat_client_decodes_vllm_router_capture_without_masking_valid_response():
    engine = AsyncMock()
    engine.model_name = "snowball"
    engine.tokenizer = MagicMock()
    engine.tokenizer.decode.return_value = "answer"
    engine.tokenize.return_value = {"tokens": [11, 12]}
    engine.chat_completion.return_value = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "answer"},
                "finish_reason": "stop",
                "token_ids": [21, 22],
                "routed_experts": _encoded_routes([[[1, 2]], [[3, 4]], [[5, 6]]]),
            }
        ]
    }

    output = await DirectModelClient(engine).generate(
        {
            "prompts": [[{"role": "user", "content": "question"}]],
            "session_ids": ["trajectory-1"],
            "sampling_params": {},
            "chat_completion_params": [{}],
        }
    )

    assert output["response_ids"] == [[21, 22]]
    assert output["routed_experts"] == [[[[5, 6]], [[0, 0]]]]


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


@pytest.mark.asyncio
async def test_direct_chat_continuation_preserves_sampled_tool_call_tokens():
    engine = AsyncMock()
    engine.model_name = "snowball"
    engine.tokenizer = MagicMock()
    engine.tokenizer.decode.return_value = "answer"

    async def tokenize(request):
        messages = request["json"]["messages"]
        if len(messages) == 1:
            return {"tokens": [11, 12]}
        if len(messages) == 2 and messages[1]["content"] == "":
            return {"tokens": [11, 12, 30]}
        if len(messages) == 2:
            return {"tokens": [11, 12, 99, 22, 30]}
        return {"tokens": [11, 12, 99, 22, 30, 40, 41]}

    engine.tokenize.side_effect = tokenize
    engine.chat_completion.return_value = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "answer"},
                "finish_reason": "stop",
                "token_ids": [50],
            }
        ]
    }
    messages = [
        {"role": "user", "content": "run this"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "execute_python", "arguments": '{"code":"print(1)"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "1"},
    ]

    output = await DirectModelClient(engine).generate(
        {
            "prompts": [messages],
            "session_ids": ["trajectory-1"],
            "chat_completion_params": [{"tools": []}],
            "chat_continuations": [{"served_prefix_token_ids": [11, 12, 21, 22], "assistant_message_index": 1}],
        }
    )

    assert output["prompt_ids"] == [[11, 12, 21, 22, 30, 40, 41]]
    assert engine.chat_completion.await_args.args[0]["json"]["_skyrl_exact_prompt_token_ids"] == [
        11,
        12,
        21,
        22,
        30,
        40,
        41,
    ]


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
                "provider_specific_fields": {"routed_experts": [[[4, 7]]]},
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
    assert output["routed_experts"] == [[[[4, 7]]]]


@pytest.fixture
def http_client():
    @contextlib.asynccontextmanager
    async def serve(complete, tokenize=None, tokenizer=None):
        app = web.Application()
        app.router.add_post("/v1/chat/completions", complete)
        if tokenize is not None:
            app.router.add_post("/tokenize", tokenize)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            yield OpenAIHTTPModelClient(
                base_url=f"http://127.0.0.1:{port}", model_name="policy", tokenizer=tokenizer or MagicMock()
            )
        finally:
            await runner.cleanup()

    return serve


def _plain_request(**sampling_params):
    return {
        "prompts": [[{"role": "user", "content": "question"}]],
        "session_ids": ["trajectory-1"],
        "sampling_params": {"temperature": 0.7, "max_generate_length": 256, **sampling_params},
    }


@pytest.mark.asyncio
async def test_http_model_client_returns_the_engines_tokens_logprobs_version_spans_and_routes(http_client):
    requests = []

    async def complete(request):
        requests.append(await request.json())
        return web.json_response(
            {
                "prompt_token_ids": [1, 2],
                "choices": [
                    {
                        "message": {"content": "answer"},
                        "finish_reason": "stop",
                        "token_ids": [7, 8],
                        "logprobs": {"content": [{"logprob": -0.1}, {"logprob": -0.2}]},
                        "policy_version_segments": [{"start": 0, "token_count": 2, "policy_version": 3}],
                        "provider_specific_fields": {"routed_experts": [[[4, 7]], [[5, 6]]]},
                    }
                ],
            }
        )

    async with http_client(complete) as client:
        output = await client.generate(_plain_request(logprobs=0))

    assert requests == [
        {
            "model": "policy",
            "messages": [{"role": "user", "content": "question"}],
            "session_id": "trajectory-1",
            "temperature": 0.7,
            "max_completion_tokens": 256,
            "return_token_ids": True,
            "logprobs": True,
        }
    ]
    assert output == {
        "responses": ["answer"],
        "prompt_ids": [[1, 2]],
        "response_ids": [[7, 8]],
        "stop_reasons": ["stop"],
        "response_logprobs": [[-0.1, -0.2]],
        "prompt_logprobs": None,
        "token_provenance": "engine",
        "response_policy_version_segments": [[{"start": 0, "token_count": 2, "policy_version": 3}]],
        "routed_experts": [[[[4, 7]], [[5, 6]]]],
    }


@pytest.mark.asyncio
async def test_http_model_client_refuses_a_response_without_exact_tokens(http_client):
    async def complete(request):
        return web.json_response({"choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}]})

    async with http_client(complete) as client:
        with pytest.raises(RuntimeError, match="exact response token IDs"):
            await client.generate(_plain_request())


@pytest.mark.asyncio
async def test_http_model_client_refuses_a_response_without_the_served_prompt(http_client):
    async def complete(request):
        return web.json_response(
            {"choices": [{"message": {"content": "answer"}, "finish_reason": "stop", "token_ids": [7, 8]}]}
        )

    async with http_client(complete) as client:
        with pytest.raises(RuntimeError, match="served prompt token IDs"):
            await client.generate(_plain_request())


@pytest.mark.asyncio
async def test_http_model_client_refuses_a_response_missing_requested_logprobs(http_client):
    async def complete(request):
        return web.json_response(
            {
                "prompt_token_ids": [1, 2],
                "choices": [{"message": {"content": "answer"}, "finish_reason": "stop", "token_ids": [7, 8]}],
            }
        )

    async with http_client(complete) as client:
        with pytest.raises(RuntimeError, match="requested logprobs"):
            await client.generate(_plain_request(logprobs=0))


@pytest.mark.asyncio
async def test_http_structured_chat_continues_from_sampled_tool_call_tokens(http_client):
    served_requests = []

    async def tokenize(request):
        messages = (await request.json())["messages"]
        if len(messages) == 1:
            tokens = [11, 12]
        elif len(messages) == 2 and messages[1]["content"] == "":
            tokens = [11, 12, 30]
        elif len(messages) == 2:
            tokens = [11, 12, 99, 22, 30]
        else:
            tokens = [11, 12, 99, 22, 30, 40, 41]
        return web.json_response({"tokens": tokens})

    tool_call = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "execute_python", "arguments": '{"code":"print(1)"}'},
            }
        ],
    }

    async def complete(request):
        body = await request.json()
        served_requests.append(body)
        message = tool_call if len(body["messages"]) == 1 else {"role": "assistant", "content": "done"}
        tokens = [21, 22] if len(body["messages"]) == 1 else [50]
        return web.json_response(
            {
                "choices": [
                    {
                        "message": message,
                        "finish_reason": "tool_calls" if message is tool_call else "stop",
                        "token_ids": tokens,
                        "logprobs": {"content": [{"logprob": -0.1}] * len(tokens)},
                    }
                ]
            }
        )

    tokenizer = MagicMock()
    tokenizer.decode.side_effect = lambda ids, **_: "tool call" if ids == [21, 22] else "done"

    async with http_client(complete, tokenize=tokenize, tokenizer=tokenizer) as client:
        first = await client.generate(
            {
                "prompts": [[{"role": "user", "content": "run this"}]],
                "session_ids": ["trajectory-1"],
                "chat_completion_params": [{"tools": []}],
                "sampling_params": {"logprobs": 0},
            }
        )
        second = await client.generate(
            {
                "prompts": [
                    [
                        {"role": "user", "content": "run this"},
                        first["assistant_messages"][0],
                        {"role": "tool", "tool_call_id": "call-1", "content": "1"},
                    ]
                ],
                "session_ids": ["trajectory-1"],
                "chat_completion_params": [{"tools": []}],
                "chat_continuations": [
                    {
                        "served_prefix_token_ids": first["prompt_ids"][0] + first["response_ids"][0],
                        "assistant_message_index": 1,
                    }
                ],
                "sampling_params": {"logprobs": 0},
            }
        )

    assert first["prompt_ids"] == [[11, 12]]
    assert first["response_ids"] == [[21, 22]]
    assert second["prompt_ids"] == [[11, 12, 21, 22, 30, 40, 41]]
    assert second["response_logprobs"] == [[-0.1]]
    assert served_requests[1]["_skyrl_exact_prompt_token_ids"] == second["prompt_ids"][0]


@pytest.mark.asyncio
async def test_http_model_client_preserves_server_error_details(http_client):
    async def reject(_request):
        return web.json_response({"error": {"message": "unsupported field"}}, status=400)

    async with http_client(reject) as client:
        with pytest.raises(RuntimeError, match="HTTP 400.*unsupported field"):
            await client.generate({"prompts": [[{"role": "user", "content": "question"}]]})
