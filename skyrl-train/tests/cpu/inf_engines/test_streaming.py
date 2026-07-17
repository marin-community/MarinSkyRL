"""
CPU-runnable tests for the streaming chat-completion feature.

Covers:
- ``ensure_token_ids_in_sse_chunk`` (pure function, no vllm dependency)
- HTTP endpoint streaming path via a mock ``CompletionBackend``
- Regression: non-streaming path still returns ``JSONResponse``

Run with:
  uv run --isolated --extra dev pytest tests/cpu/inf_engines/test_streaming.py
"""

import json
import pytest
from http import HTTPStatus

from skyrl_train.inference_engines.vllm.utils import ensure_token_ids_in_sse_chunk
from skyrl_train.inference_engines.inference_engine_client_http_endpoint import (
    create_app,
    set_global_state,
)


# ---------------------------------------------------------------------------
# ensure_token_ids_in_sse_chunk
# ---------------------------------------------------------------------------


class TestEnsureTokenIds:
    def test_remaps_new_token_ids(self):
        chunk = (
            'data: {"choices":[{"delta":{"content":"hi","provider_specific_fields":{"new_token_ids":[10,20]}}}]}\n\n'
        )
        result = ensure_token_ids_in_sse_chunk(chunk)
        data = json.loads(result[len("data: ") :].strip())
        psf = data["choices"][0]["delta"]["provider_specific_fields"]
        assert psf["token_ids"] == [10, 20]
        assert "new_token_ids" in psf  # original key preserved

    def test_passes_through_when_token_ids_already_present(self):
        chunk = 'data: {"choices":[{"delta":{"provider_specific_fields":{"token_ids":[1,2],"new_token_ids":[1,2]}}}]}'
        result = ensure_token_ids_in_sse_chunk(chunk)
        # Should be unchanged — token_ids already present
        data = json.loads(result[len("data: ") :].strip())
        psf = data["choices"][0]["delta"]["provider_specific_fields"]
        assert psf["token_ids"] == [1, 2]

    def test_passes_through_done(self):
        assert ensure_token_ids_in_sse_chunk("data: [DONE]\n\n") == "data: [DONE]\n\n"

    def test_passes_through_non_data_line(self):
        assert ensure_token_ids_in_sse_chunk(": keepalive\n\n") == ": keepalive\n\n"

    def test_passes_through_no_choices(self):
        chunk = 'data: {"id":"x","choices":[]}\n\n'
        assert ensure_token_ids_in_sse_chunk(chunk) == chunk

    def test_passes_through_no_psf(self):
        chunk = 'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
        assert ensure_token_ids_in_sse_chunk(chunk) == chunk

    def test_handles_malformed_json(self):
        chunk = "data: {bad json}\n\n"
        assert ensure_token_ids_in_sse_chunk(chunk) == chunk

    def test_preserves_other_fields(self):
        chunk = (
            'data: {"id":"abc","object":"chat.completion.chunk",'
            '"choices":[{"index":0,"delta":{"content":"world",'
            '"provider_specific_fields":{"new_token_ids":[42]}},'
            '"finish_reason":null}],"model":"test"}\n\n'
        )
        result = ensure_token_ids_in_sse_chunk(chunk)
        data = json.loads(result[len("data: ") :].strip())
        assert data["id"] == "abc"
        assert data["object"] == "chat.completion.chunk"
        assert data["choices"][0]["delta"]["content"] == "world"
        assert data["choices"][0]["delta"]["provider_specific_fields"]["token_ids"] == [42]


# ---------------------------------------------------------------------------
# HTTP endpoint streaming path
# ---------------------------------------------------------------------------


class MockStreamingBackend:
    """Minimal CompletionBackend that yields canned SSE chunks."""

    model_name = "test-model"

    def __init__(self, chunks=None, non_stream_response=None):
        self._chunks = chunks or [
            'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n',
            'data: {"choices":[{"delta":{"content":"hello","provider_specific_fields":'
            '{"token_ids":[1,2,3],"new_token_ids":[1,2,3]}}}]}\n\n',
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n',
            "data: [DONE]\n\n",
        ]
        self._non_stream_response = non_stream_response or {
            "id": "test",
            "choices": [{"message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
        }

    async def chat_completion(self, request_payload):
        return self._non_stream_response

    async def chat_completion_stream(self, request_payload):
        for chunk in self._chunks:
            yield chunk

    async def completion(self, request_payload):
        return self._non_stream_response


@pytest.fixture
def app_with_mock_backend():
    from starlette.testclient import TestClient

    backend = MockStreamingBackend()
    app = create_app()
    # We only need the client reference; the uvicorn_server arg is unused by tests
    set_global_state(backend, None)
    client = TestClient(app)
    yield client, backend


@pytest.fixture
def app_with_error_backend():
    from starlette.testclient import TestClient

    backend = MockStreamingBackend(
        chunks=[
            'data: {"error":{"message":"boom","type":"Internal Server Error","code":500}}\n\n',
            "data: [DONE]\n\n",
        ]
    )
    app = create_app()
    set_global_state(backend, None)
    client = TestClient(app)
    yield client


class TestHTTPEndpointStreaming:
    def test_streaming_returns_event_stream(self, app_with_mock_backend):
        client, _ = app_with_mock_backend
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

    def test_streaming_body_contains_sse_chunks(self, app_with_mock_backend):
        client, _ = app_with_mock_backend
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        body = resp.text
        assert "data: " in body
        assert "data: [DONE]" in body

    def test_streaming_forwards_token_ids(self, app_with_mock_backend):
        """SSE chunks carrying token_ids (post-engine remap) reach the client."""
        client, _ = app_with_mock_backend
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        found_psf = False
        for line in resp.text.split("\n"):
            if not line.startswith("data: ") or "[DONE]" in line:
                continue
            data = json.loads(line[len("data: ") :])
            delta = data.get("choices", [{}])[0].get("delta", {})
            psf = delta.get("provider_specific_fields")
            if psf:
                found_psf = True
                assert "token_ids" in psf, "token_ids must be present for harbor literal capture"
        assert found_psf, "Expected at least one chunk with provider_specific_fields"

    def test_non_streaming_unchanged(self, app_with_mock_backend):
        """Non-streaming requests still get a JSON response (regression guard)."""
        client, _ = app_with_mock_backend
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/json"
        data = resp.json()
        assert data["choices"][0]["message"]["content"] == "hello"

    def test_streaming_error_chunk_passes_through(self, app_with_error_backend):
        """Error SSE chunks from the engine are forwarded as-is."""
        resp = app_with_error_backend.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        assert resp.status_code == 200
        assert "data: [DONE]" in resp.text

    def test_streaming_validates_model_name(self, app_with_mock_backend):
        """Model-name validation still fires before the streaming branch."""
        client, _ = app_with_mock_backend
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "wrong-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        assert resp.status_code == HTTPStatus.BAD_REQUEST
        assert "Model name mismatch" in resp.json()["error"]["message"]
