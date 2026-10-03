import asyncio
import json
import logging

import httpx
import pytest
from marinskyrl.harbor_agent_names import MINI_SWE_HARBOR_AGENT_NAME
from skyrl_train.inference_engines.inference_engine_client_http_endpoint import create_app, set_global_state
from skyrl_train.inference_engines.harbor_continuation import (
    EXACT_PROMPT_TOKEN_IDS_KEY,
    TRIAL_ID_HEADER,
    TASK_AGENT_HEADER,
    HarborContinuationManager,
)

TOOLS = [{"type": "function", "function": {"name": "bash", "parameters": {"type": "object"}}}]


class _ContinuationBackend:
    model_name = "test-model"
    max_model_len = 128

    def __init__(self, *, prompt_ids=None, completion_ids=None) -> None:
        self.chat_requests = []
        self.next_prompt_ids = prompt_ids or [[1, 2], [1, 2, 99, 77, 40, 41], [7, 8]]
        self.next_completion_ids = completion_ids or [[99], [100], [9]]

    async def chat_completion(self, request):
        index = len(self.chat_requests)
        self.chat_requests.append(request)
        return {
            "id": f"chat-{index}",
            "model": self.model_name,
            "prompt_token_ids": self.next_prompt_ids[index],
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "\ufffd" if index == 0 else "ok"},
                    "token_ids": self.next_completion_ids[index],
                    "finish_reason": "stop",
                }
            ],
        }

    async def completion(self, _request):
        raise AssertionError("OpenCode uses chat completions")

    async def tokenize(self, request):
        body = request["json"]
        messages = body["messages"]
        if body.get("add_generation_prompt") is True and len(messages) == 1:
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


@pytest.mark.parametrize("stream, native_tools", [(True, True), (False, False)])
@pytest.mark.asyncio
async def test_chat_continues_from_exact_served_ids_across_agent_turn(caplog, stream, native_tools):
    caplog.set_level(logging.INFO)
    backend = _ContinuationBackend()
    set_global_state(backend, None)
    app = create_app(backend=backend, enable_harbor_exact_continuation=True)
    headers = {TRIAL_ID_HEADER: "trial-a"}
    agent_fields = {"tools": TOOLS} if native_tools else {}
    if not native_tools:
        headers[TASK_AGENT_HEADER] = MINI_SWE_HARBOR_AGENT_NAME
    first_messages = [{"role": "user", "content": "run it"}]

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        first = await client.post(
            "/v1/chat/completions",
            headers=headers,
            json={"model": backend.model_name, "messages": first_messages, "stream": stream, **agent_fields},
        )
        assert first.status_code == 200

        second_messages = [
            *first_messages,
            {
                "role": "assistant",
                "content": "\ufffd",
                **({"tool_calls": [{"id": "call-1", "type": "function"}]} if native_tools else {}),
            },
            (
                {"role": "tool", "tool_call_id": "call-1", "content": "tool output"}
                if native_tools
                else {"role": "user", "content": "bash output"}
            ),
        ]
        if not native_tools:
            budget = await client.post(
                "/tokenize",
                headers=headers,
                json={
                    "model": backend.model_name,
                    "messages": second_messages,
                    "tools": [],
                    "chat_template_kwargs": {},
                    "add_generation_prompt": True,
                },
            )
            assert budget.status_code == 200
            assert budget.json()["count"] == 6
            assert budget.json()["tokens"] == [1, 2, 99, 77, 40, 41]
        second = await client.post(
            "/v1/chat/completions",
            headers=headers,
            json={"model": backend.model_name, "messages": second_messages, "stream": stream, **agent_fields},
        )
        assert second.status_code == 200

    assert backend.chat_requests[0]["json"]["session_id"] == "trial-a"
    assert EXACT_PROMPT_TOKEN_IDS_KEY not in backend.chat_requests[0]["json"]
    assert backend.chat_requests[1]["json"][EXACT_PROMPT_TOKEN_IDS_KEY] == [1, 2, 99, 77, 40, 41]
    assert backend.chat_requests[1]["json"]["session_id"] == "trial-a"
    if stream:
        assert any(
            record.message == "OpenCode task-agent response reached output limit: trial_id=trial-a"
            for record in caplog.records
        )
    else:
        assert first.json()["choices"][0]["token_ids"] == [99]
        assert second.json()["prompt_token_ids"] == [1, 2, 99, 77, 40, 41]


@pytest.mark.asyncio
async def test_opencode_compaction_restarts_exact_continuation_segment():
    backend = _ContinuationBackend()
    set_global_state(backend, None)
    app = create_app(backend=backend, enable_harbor_exact_continuation=True)
    headers = {TRIAL_ID_HEADER: "trial-compacted"}

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        await client.post(
            "/v1/chat/completions",
            headers=headers,
            json={
                "model": backend.model_name,
                "messages": [{"role": "user", "content": "original"}],
                "tools": TOOLS,
                "stream": True,
            },
        )
        await client.post(
            "/v1/chat/completions",
            headers=headers,
            json={
                "model": backend.model_name,
                "messages": [{"role": "user", "content": "summary"}],
                "tools": TOOLS,
                "stream": True,
            },
        )

    assert EXACT_PROMPT_TOKEN_IDS_KEY not in backend.chat_requests[1]["json"]


@pytest.mark.asyncio
async def test_exact_continuation_is_scoped_to_marked_requests():
    backend = _ContinuationBackend()
    set_global_state(backend, None)
    app = create_app(backend=backend, enable_harbor_exact_continuation=True)

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
    app = create_app(backend=backend, enable_harbor_exact_continuation=True)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        for trial, initial in (("trial-a", "alpha"), ("trial-b", "beta")):
            await client.post(
                "/v1/chat/completions",
                headers={TRIAL_ID_HEADER: trial},
                json={
                    "model": backend.model_name,
                    "messages": [{"role": "user", "content": initial}],
                    "tools": TOOLS,
                    "stream": True,
                },
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
                    "tools": TOOLS,
                    "stream": True,
                },
            )

    assert backend.chat_requests[2]["json"][EXACT_PROMPT_TOKEN_IDS_KEY] == [1, 2, 91, 77, 40, 41]
    assert backend.chat_requests[3]["json"][EXACT_PROMPT_TOKEN_IDS_KEY] == [7, 8, 92, 77, 40, 41]


@pytest.mark.asyncio
async def test_cancelled_stream_releases_trial_for_timeout_retry():
    backend = _ContinuationBackend()
    manager = HarborContinuationManager(backend)
    payload = {
        "headers": {TRIAL_ID_HEADER: "trial-timeout"},
        "json": {
            "model": backend.model_name,
            "messages": [{"role": "user", "content": "slow"}],
            "tools": TOOLS,
            "stream": True,
        },
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


@pytest.mark.asyncio
async def test_auxiliary_generation_does_not_replace_agent_continuation_state():
    backend = _ContinuationBackend(
        prompt_ids=[[1, 2], [7, 8], [1, 2, 99, 77, 40, 41]],
        completion_ids=[[99], [9], [100]],
    )
    set_global_state(backend, None)
    app = create_app(backend=backend, enable_harbor_exact_continuation=True)
    headers = {TRIAL_ID_HEADER: "trial-with-title"}
    first_messages = [{"role": "user", "content": "run it"}]

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        await client.post(
            "/v1/chat/completions",
            headers=headers,
            json={"model": backend.model_name, "messages": first_messages, "tools": TOOLS, "stream": True},
        )
        await client.post(
            "/v1/chat/completions",
            headers=headers,
            json={
                "model": backend.model_name,
                "messages": [{"role": "user", "content": "Generate a title"}],
                "stream": True,
            },
        )
        await client.post(
            "/v1/chat/completions",
            headers=headers,
            json={
                "model": backend.model_name,
                "messages": [
                    *first_messages,
                    {"role": "assistant", "content": "tool call"},
                    {"role": "tool", "content": "result"},
                ],
                "tools": TOOLS,
                "stream": True,
            },
        )

    assert "session_id" not in backend.chat_requests[1]["json"]
    assert backend.chat_requests[2]["json"][EXACT_PROMPT_TOKEN_IDS_KEY] == [1, 2, 99, 77, 40, 41]


@pytest.mark.asyncio
async def test_cancelled_nonstream_response_releases_trial_for_retry():
    backend = _ContinuationBackend()
    manager = HarborContinuationManager(backend)
    payload = {
        "headers": {TRIAL_ID_HEADER: "trial-nonstream", TASK_AGENT_HEADER: MINI_SWE_HARBOR_AGENT_NAME},
        "json": {"model": backend.model_name, "messages": [{"role": "user", "content": "run bash"}]},
    }
    lease = await manager.begin(payload)
    assert lease is not None
    started = asyncio.Event()

    async def response():
        started.set()
        await asyncio.Event().wait()

    task = asyncio.create_task(lease.capture_response(response()))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    retry = await asyncio.wait_for(manager.begin(payload), timeout=0.1)
    assert retry is not None
    result = await retry.capture_response(backend.chat_completion(payload))
    assert result["choices"][0]["token_ids"] == [99]
