import json
from types import SimpleNamespace

import httpx
import pytest
from openai.types.chat import ChatCompletionToolParam
from pydantic import TypeAdapter
from rolloutengine.contracts import ModelRequest, ModelTurn
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from skyrl_train.inference_engines.inference_engine_client_http_endpoint import create_app, set_global_state
from skyrl_train.rollouts.agent_backend import AgentInferenceBackend, TRIAL_HEADER
from skyrl_train.rollouts.task_worker import _model_turn
from skyrl_train.trajectory_runners.model_clients import DirectModelClient


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("request_logprobs", [False, True])
async def test_http_agent_completion_retains_tool_call_and_exact_token_evidence(stream, request_logprobs):
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0, "tool": 1})))
    requests = []
    message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "shell", "arguments": "{}"}}],
    }

    async def complete(request):
        requests.append(request)
        return ModelTurn(message, (7, 8), (1,), (-0.25,), "stop")

    async def tokenize(payload):
        return {"count": 2, "max_model_len": 32, "tokens": [7, 8]}

    backend = AgentInferenceBackend("model", 32, tokenize, tokenizer)
    set_global_state(backend, None)
    try:
        with backend.session("trial", complete):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=create_app()), base_url="http://test"
            ) as client:
                response = await client.post(
                    "/v1/chat/completions",
                    headers={TRIAL_HEADER: "trial"},
                    json={
                        "model": "model",
                        "messages": [{"role": "user", "content": "Use shell."}],
                        "stream": stream,
                        **({"logprobs": True} if request_logprobs else {}),
                        "tool_choice": "auto",
                        "max_tokens": 16,
                    },
                )
        assert response.status_code == 200
        if stream:
            events = [line.removeprefix("data: ") for line in response.text.splitlines() if line.startswith("data: ")]
            assert events[-1] == "[DONE]"
            payload = json.loads(events[0])
            wire_message = payload["choices"][0]["delta"]
            assert wire_message["tool_calls"][0].pop("index") == 0
        else:
            payload = response.json()
            wire_message = payload["choices"][0]["message"]
        assert wire_message == message
        assert payload["prompt_token_ids"] == [7, 8]
        assert payload["choices"][0]["token_ids"] == [1]
        if request_logprobs:
            assert payload["choices"][0]["logprobs"]["content"][0]["logprob"] == -0.25
        else:
            assert payload["choices"][0]["logprobs"] is None
        assert payload["choices"][0]["finish_reason"] == "tool_calls"
        assert requests[0].options == {"tool_choice": "auto", "max_tokens": 16}
        with pytest.raises(ValueError, match="No active agent session"):
            await backend.chat_completion({"headers": {TRIAL_HEADER: "trial"}, "json": {}})
    finally:
        set_global_state(None, None)


@pytest.mark.asyncio
async def test_agent_chat_tools_reach_serving_without_double_wrapping():
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0, "done": 1})))
    tools = [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Execute a command.",
                "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
                "strict": True,
            },
        }
    ]
    serving_requests = []

    async def tokenize(payload):
        TypeAdapter(list[ChatCompletionToolParam]).validate_python(payload["json"]["tools"])
        serving_requests.append(payload["json"])
        return {"tokens": [7, 8]}

    async def chat_completion(payload):
        TypeAdapter(list[ChatCompletionToolParam]).validate_python(payload["json"]["tools"])
        serving_requests.append(payload["json"])
        return {
            "choices": [
                {"message": {"role": "assistant", "content": "done"}, "finish_reason": "stop", "token_ids": [1]}
            ]
        }

    client = DirectModelClient(
        SimpleNamespace(model_name="model", tokenizer=tokenizer, tokenize=tokenize, chat_completion=chat_completion)
    )

    async def complete(request):
        return await _model_turn(
            client,
            ModelRequest(request.messages, request.options, (), None),
            sampling_params={},
            max_context_length=32,
            max_prompt_length=None,
            chat_template_kwargs={},
            session_id="trial",
        )

    backend = AgentInferenceBackend("model", 32, tokenize, tokenizer)
    with backend.session("trial", complete):
        response = await backend.chat_completion(
            {
                "headers": {TRIAL_HEADER: "trial"},
                "json": {"messages": [{"role": "user", "content": "Run a command."}], "tools": tools},
            }
        )
    assert response["choices"][0]["message"]["content"] == "done"
    assert response["prompt_token_ids"] == [7, 8]
    assert response["choices"][0]["token_ids"] == [1]
    assert response["choices"][0]["logprobs"] is None
    assert [request["tools"] for request in serving_requests] == [tools, tools]
