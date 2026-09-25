import asyncio
import base64
import io
import json
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from aiohttp import web
from jinja2 import TemplateError

from skyrl_train.inference_engines.chat_template import SINGLE_TOOL_CALL_TEMPLATE_ERROR
from skyrl_train.trajectory_runners import model_clients
from skyrl_train.trajectory_runners.model_clients import DirectModelClient, OpenAIHTTPModelClient


def _encoded_routes(rows):
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(rows, dtype=np.uint16), allow_pickle=False)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


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
async def test_direct_model_client_recovers_strict_template_by_sequentializing_tool_calls():
    class WrappedValidationError(RuntimeError):
        def as_instanceof_cause(self):
            return TemplateError(SINGLE_TOOL_CALL_TEMPLATE_ERROR)

    engine = AsyncMock()
    engine.model_name = "strict-tool-model"
    engine.tokenizer = MagicMock()
    engine.tokenizer.decode.return_value = "answer"
    engine.tokenize.side_effect = [WrappedValidationError(), {"tokens": [11, 12, 13]}]
    engine.chat_completion.return_value = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "answer"},
                "finish_reason": "stop",
                "token_ids": [21],
            }
        ]
    }
    first_call = {"id": "call-1", "type": "function", "function": {"name": "python", "arguments": "one"}}
    second_call = {"id": "call-2", "type": "function", "function": {"name": "python", "arguments": "two"}}
    messages = [
        {"role": "user", "content": "calculate"},
        {"role": "assistant", "content": None, "tool_calls": [first_call, second_call]},
        {"role": "tool", "tool_call_id": "call-1", "content": "1"},
        {"role": "tool", "tool_call_id": "call-2", "content": "2"},
    ]

    output = await DirectModelClient(engine).generate(
        {
            "prompts": [messages],
            "chat_completion_params": [{}],
        }
    )

    served_messages = engine.chat_completion.await_args.args[0]["json"]["messages"]
    assert served_messages == [
        {"role": "user", "content": "calculate"},
        {"role": "assistant", "content": None, "tool_calls": [first_call]},
        {"role": "assistant", "content": None, "tool_calls": [second_call]},
        {"role": "tool", "tool_call_id": "call-1", "content": "1"},
        {"role": "tool", "tool_call_id": "call-2", "content": "2"},
    ]
    assert output["prompt_ids"] == [[11, 12, 13]]


@pytest.mark.asyncio
async def test_strict_template_recovery_preserves_exact_prefix_and_tool_results():
    class WrappedValidationError(RuntimeError):
        def as_instanceof_cause(self):
            return TemplateError(SINGLE_TOOL_CALL_TEMPLATE_ERROR)

    async def tokenize(request):
        messages = request["json"]["messages"]
        if any(len(message.get("tool_calls") or []) > 1 for message in messages):
            raise WrappedValidationError()
        if len(messages) == 5:
            return {"tokens": [10, 20, 30, 40, 50]}
        if len(messages) == 3 and messages[-1].get("tool_calls"):
            return {"tokens": [10, 20, 30]}
        if len(messages) == 2:
            return {"tokens": [10, 20]}
        return {"tokens": [10, 20, 99]}

    engine = AsyncMock()
    engine.model_name = "strict-tool-model"
    engine.tokenizer = MagicMock()
    engine.tokenizer.decode.return_value = "answer"
    engine.tokenize.side_effect = tokenize
    engine.chat_completion.return_value = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "answer"},
                "finish_reason": "stop",
                "token_ids": [21],
            }
        ]
    }
    calls = [
        {"id": "call-1", "type": "function", "function": {"name": "python", "arguments": "one"}},
        {"id": "call-2", "type": "function", "function": {"name": "python", "arguments": "two"}},
    ]
    messages = [
        {"role": "user", "content": "calculate"},
        {"role": "assistant", "content": None, "tool_calls": calls},
        {"role": "tool", "tool_call_id": "call-1", "content": "1"},
        {"role": "tool", "tool_call_id": "call-2", "content": "2"},
    ]

    output = await DirectModelClient(engine).generate(
        {
            "prompts": [messages],
            "chat_completion_params": [{}],
            "chat_continuations": [{"served_prefix_token_ids": [7, 8], "assistant_message_index": 1}],
        }
    )

    served = engine.chat_completion.await_args.args[0]["json"]
    assert served["_skyrl_exact_prompt_token_ids"] == [7, 8, 99, 40, 50]
    assert [message.get("tool_call_id") for message in served["messages"] if message["role"] == "tool"] == [
        "call-1",
        "call-2",
    ]
    assert output["prompt_ids"] == [[7, 8, 99, 40, 50]]


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
                "token_ids": [9, 10],
                "provider_specific_fields": {"routed_experts": _encoded_routes([[[1, 2]], [[3, 4]], [[4, 7]]])},
                "logprobs": {
                    "content": [
                        {
                            "logprob": -7.0,
                            "top_logprobs": [
                                {"token": "token_id:9", "logprob": -7.0},
                                {"token": "token_id:3", "logprob": -0.2},
                                {"token": "token_id:2", "logprob": -0.1},
                            ],
                        },
                        {
                            "logprob": -0.1,
                            "top_logprobs": [
                                {"token": "token_id:10", "logprob": -0.1},
                                {"token": "token_id:11", "logprob": -0.2},
                            ],
                        },
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
    assert output["student_topk_indices"] == [[[2, 3], [10, 11]]]
    assert output["behavior_topk_logprobs"] == [[[-0.1, -0.2], [-0.1, -0.2]]]
    assert output["routed_experts"] == [[[[4, 7]], [[0, 0]]]]


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
        client = OpenAIHTTPModelClient(
            base_url=f"http://127.0.0.1:{port}",
            model_name="policy",
            tokenizer=Tokenizer(),
            max_concurrent_requests=8,
        )
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
async def test_http_model_client_limits_requests_across_concurrent_generation_calls(monkeypatch):
    saturated = asyncio.Event()
    release = asyncio.Event()
    active_requests = 0
    peak_requests = 0

    class FakeResponse:
        status = 200

        async def __aenter__(self):
            nonlocal active_requests, peak_requests
            active_requests += 1
            peak_requests = max(peak_requests, active_requests)
            if active_requests == 2:
                saturated.set()
            await release.wait()
            return self

        async def __aexit__(self, *_args):
            nonlocal active_requests
            active_requests -= 1

        async def json(self):
            return {"choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}]}

    class FakeSession:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            pass

        def post(self, *_args, **_kwargs):
            return FakeResponse()

    monkeypatch.setattr(model_clients.aiohttp, "ClientSession", FakeSession)
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1]
    client = OpenAIHTTPModelClient(
        base_url="http://inference.test",
        model_name="policy",
        tokenizer=tokenizer,
        max_concurrent_requests=2,
    )
    generations = [
        asyncio.create_task(client.generate({"prompts": [[{"role": "user", "content": str(index)}]]}))
        for index in range(6)
    ]

    await saturated.wait()
    assert peak_requests == 2
    release.set()
    outputs = await asyncio.gather(*generations)

    assert [output["responses"] for output in outputs] == [["answer"]] * 6


@pytest.mark.asyncio
async def test_structured_chat_transport_parity_across_tool_continuation():
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

    app = web.Application()
    app.router.add_post("/tokenize", tokenize)
    app.router.add_post("/v1/chat/completions", complete)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    tokenizer = MagicMock()
    tokenizer.decode.side_effect = lambda ids, **_: "tool call" if ids == [21, 22] else "done"
    client = OpenAIHTTPModelClient(
        base_url=f"http://127.0.0.1:{port}",
        model_name="policy",
        tokenizer=tokenizer,
        max_concurrent_requests=8,
    )
    first_request = {
        "prompts": [[{"role": "user", "content": "run this"}]],
        "session_ids": ["trajectory-1"],
        "chat_completion_params": [{"tools": []}],
        "sampling_params": {"logprobs": 0},
    }

    try:
        first = await client.generate(first_request)
        second_request = {
            "prompts": [
                [
                    first_request["prompts"][0][0],
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
        second = await client.generate(second_request)
    finally:
        await runner.cleanup()

    async def direct_tokenize(payload):
        response = await tokenize(MagicMock(json=AsyncMock(return_value=payload["json"])))
        return json.loads(response.body)

    async def direct_complete(payload):
        response = await complete(MagicMock(json=AsyncMock(return_value=payload["json"])))
        return json.loads(response.body)

    backend = MagicMock(model_name="policy", tokenizer=tokenizer)
    backend.tokenize = direct_tokenize
    backend.chat_completion = direct_complete
    direct = DirectModelClient(backend)
    assert await direct.generate(first_request) == first
    assert await direct.generate(second_request) == second

    assert first["prompt_ids"] == [[11, 12]]
    assert first["response_ids"] == [[21, 22]]
    assert second["prompt_ids"] == [[11, 12, 21, 22, 30, 40, 41]]
    assert second["response_logprobs"] == [[-0.1]]
    assert served_requests[1]["_skyrl_exact_prompt_token_ids"] == second["prompt_ids"][0]


@pytest.mark.asyncio
async def test_http_structured_chat_recovers_strict_single_tool_call_template():
    served_requests = []

    async def tokenize(request):
        messages = (await request.json())["messages"]
        if any(len(message.get("tool_calls") or []) > 1 for message in messages):
            return web.json_response({"error": {"message": SINGLE_TOOL_CALL_TEMPLATE_ERROR}}, status=500)
        return web.json_response({"tokens": [11, 12]})

    async def complete(request):
        body = await request.json()
        served_requests.append(body)
        return web.json_response(
            {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "done"},
                        "finish_reason": "stop",
                        "token_ids": [21],
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
    tokenizer = MagicMock()
    tokenizer.decode.return_value = "done"
    client = OpenAIHTTPModelClient(
        base_url=f"http://127.0.0.1:{port}",
        model_name="strict-tool-model",
        tokenizer=tokenizer,
        max_concurrent_requests=4,
    )
    calls = [
        {"id": "call-1", "type": "function", "function": {"name": "python", "arguments": "one"}},
        {"id": "call-2", "type": "function", "function": {"name": "python", "arguments": "two"}},
    ]

    try:
        output = await client.generate(
            {
                "prompts": [
                    [
                        {"role": "user", "content": "calculate"},
                        {"role": "assistant", "content": None, "tool_calls": calls},
                        {"role": "tool", "tool_call_id": "call-1", "content": "1"},
                        {"role": "tool", "tool_call_id": "call-2", "content": "2"},
                    ]
                ],
                "chat_completion_params": [{}],
            }
        )
    finally:
        await runner.cleanup()

    assert [message["tool_calls"] for message in served_requests[0]["messages"] if message["role"] == "assistant"] == [
        [calls[0]],
        [calls[1]],
    ]
    assert output["response_ids"] == [[21]]


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
        client = OpenAIHTTPModelClient(
            base_url=f"http://127.0.0.1:{port}",
            model_name="policy",
            tokenizer=MagicMock(),
            max_concurrent_requests=8,
        )
        with pytest.raises(RuntimeError, match="HTTP 400.*unsupported field"):
            await client.generate({"prompts": [[{"role": "user", "content": "question"}]]})
    finally:
        await runner.cleanup()
