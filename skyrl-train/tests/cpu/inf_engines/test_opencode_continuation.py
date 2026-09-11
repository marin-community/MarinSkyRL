import json
import asyncio

import httpx
import pytest

from skyrl_train.inference_engines.inference_engine_client_http_endpoint import create_app, set_global_state
from skyrl_train.inference_engines.opencode_continuation import (
    EXACT_PROMPT_TOKEN_IDS_KEY,
    OpenCodeContinuationManager,
    TRIAL_ID_HEADER,
)


class _ContinuationBackend:
    model_name = "test-model"
    max_model_len = 128

    def __init__(self, *, prompt_ids=None, completion_ids=None) -> None:
        self.chat_requests = []
        self.next_prompt_ids = prompt_ids or [[1, 2], [1, 2, 99, 77, 40, 41], [7, 8]]
        self.next_completion_ids = completion_ids or [[99], [100], [9]]

    async def chat_completion(self, _request):
        raise AssertionError("OpenCode uses streaming chat completions")

    async def completion(self, _request):
        raise AssertionError("OpenCode uses chat completions")

    async def tokenize(self, request):
        body = request["json"]
        messages = body["messages"]
        if body.get("add_generation_prompt") is True:
            return {"tokens": [10, 20], "count": 2, "max_model_len": 128}
        if messages[-1] == {"role": "assistant", "content": ""}:
            return {"tokens": [10, 20, 77], "count": 3, "max_model_len": 128}
        if body.get("add_generation_prompt") is False:
            return {"tokens": [10, 20, 30], "count": 3, "max_model_len": 128}
        return {"tokens": [10, 20, 30, 40, 41], "count": 5, "max_model_len": 128}

    async def chat_completion_stream(self, request):
        index = len(self.chat_requests)
        self.chat_requests.append(request)
        prompt_ids = self.next_prompt_ids[index]
        completion_ids = self.next_completion_ids[index]
        first = {
            "id": f"chat-{index}",
            "model": self.model_name,
            "prompt_token_ids": prompt_ids,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}],
        }
        token = {
            "id": f"chat-{index}",
            "model": self.model_name,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": "\ufffd" if index == 0 else "ok",
                        "provider_specific_fields": {"token_ids": completion_ids},
                    },
                }
            ],
        }
        finish = {
            "id": f"chat-{index}",
            "model": self.model_name,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "length" if index == 0 else "stop"}],
        }
        for payload in (first, token, finish):
            yield f"data: {json.dumps(payload)}\n\n"
        yield "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_opencode_continues_from_exact_served_ids_across_tool_turn():
    backend = _ContinuationBackend()
    set_global_state(backend, None)
    app = create_app(backend=backend, enable_opencode_exact_continuation=True)
    headers = {TRIAL_ID_HEADER: "trial-a"}
    first_messages = [{"role": "user", "content": "run it"}]

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        first = await client.post(
            "/v1/chat/completions",
            headers=headers,
            json={"model": backend.model_name, "messages": first_messages, "stream": True},
        )
        assert first.status_code == 200

        second_messages = [
            *first_messages,
            {"role": "assistant", "content": "\ufffd", "tool_calls": [{"id": "call-1", "type": "function"}]},
            {"role": "tool", "tool_call_id": "call-1", "content": "tool output"},
        ]
        second = await client.post(
            "/v1/chat/completions",
            headers=headers,
            json={"model": backend.model_name, "messages": second_messages, "stream": True},
        )
        assert second.status_code == 200

    assert backend.chat_requests[0]["json"]["session_id"] == "trial-a"
    assert EXACT_PROMPT_TOKEN_IDS_KEY not in backend.chat_requests[0]["json"]
    assert backend.chat_requests[1]["json"][EXACT_PROMPT_TOKEN_IDS_KEY] == [1, 2, 99, 77, 40, 41]
    assert backend.chat_requests[1]["json"]["session_id"] == "trial-a"


@pytest.mark.asyncio
async def test_opencode_compaction_restarts_exact_continuation_segment():
    backend = _ContinuationBackend()
    set_global_state(backend, None)
    app = create_app(backend=backend, enable_opencode_exact_continuation=True)
    headers = {TRIAL_ID_HEADER: "trial-compacted"}

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        await client.post(
            "/v1/chat/completions",
            headers=headers,
            json={"model": backend.model_name, "messages": [{"role": "user", "content": "original"}], "stream": True},
        )
        await client.post(
            "/v1/chat/completions",
            headers=headers,
            json={"model": backend.model_name, "messages": [{"role": "user", "content": "summary"}], "stream": True},
        )

    assert EXACT_PROMPT_TOKEN_IDS_KEY not in backend.chat_requests[1]["json"]


@pytest.mark.asyncio
async def test_exact_continuation_is_scoped_to_marked_requests():
    backend = _ContinuationBackend()
    set_global_state(backend, None)
    app = create_app(backend=backend, enable_opencode_exact_continuation=True)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        await client.post(
            "/v1/chat/completions",
            json={"model": backend.model_name, "messages": [{"role": "user", "content": "unmarked"}], "stream": True},
        )

    assert "session_id" not in backend.chat_requests[0]["json"]
    assert EXACT_PROMPT_TOKEN_IDS_KEY not in backend.chat_requests[0]["json"]


@pytest.mark.asyncio
async def test_concurrent_trial_histories_remain_isolated():
    backend = _ContinuationBackend(
        prompt_ids=[[1, 2], [7, 8], [1, 2, 91, 77, 40, 41], [7, 8, 92, 77, 40, 41]],
        completion_ids=[[91], [92], [93], [94]],
    )
    set_global_state(backend, None)
    app = create_app(backend=backend, enable_opencode_exact_continuation=True)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        for trial, initial in (("trial-a", "alpha"), ("trial-b", "beta")):
            await client.post(
                "/v1/chat/completions",
                headers={TRIAL_ID_HEADER: trial},
                json={"model": backend.model_name, "messages": [{"role": "user", "content": initial}], "stream": True},
            )
        for trial, initial in (("trial-a", "alpha"), ("trial-b", "beta")):
            await client.post(
                "/v1/chat/completions",
                headers={TRIAL_ID_HEADER: trial},
                json={
                    "model": backend.model_name,
                    "messages": [
                        {"role": "user", "content": initial},
                        {"role": "assistant", "content": "tool call"},
                        {"role": "tool", "content": "result"},
                    ],
                    "stream": True,
                },
            )

    assert backend.chat_requests[2]["json"][EXACT_PROMPT_TOKEN_IDS_KEY] == [1, 2, 91, 77, 40, 41]
    assert backend.chat_requests[3]["json"][EXACT_PROMPT_TOKEN_IDS_KEY] == [7, 8, 92, 77, 40, 41]


@pytest.mark.asyncio
async def test_cancelled_stream_releases_trial_for_timeout_retry():
    backend = _ContinuationBackend()
    manager = OpenCodeContinuationManager(backend)
    payload = {
        "headers": {TRIAL_ID_HEADER: "trial-timeout"},
        "json": {"model": backend.model_name, "messages": [{"role": "user", "content": "slow"}], "stream": True},
    }
    lease = await manager.begin(payload)
    assert lease is not None

    async def stalled_stream():
        yield 'data: {"prompt_token_ids":[1,2],"choices":[]}\n\n'
        await asyncio.Event().wait()

    captured = lease.capture(stalled_stream())
    await anext(captured)
    await captured.aclose()

    retry = await asyncio.wait_for(manager.begin(payload), timeout=0.1)
    assert retry is not None
    retry_stream = retry.capture(_empty_stream())
    await anext(retry_stream, None)


async def _empty_stream():
    if False:
        yield ""
