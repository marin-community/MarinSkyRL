"""
CPU-runnable tests for the streaming chat-completion feature.

Covers:
- ``ensure_token_ids_in_sse_chunk`` (pure function, no vllm dependency)
- HTTP endpoint streaming path via a mock ``CompletionBackend``
- Regression: non-streaming path still returns ``JSONResponse``

Run with:
  uv run --isolated --group dev --extra cpu pytest tests/cpu/inf_engines/test_streaming.py
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
    """Tests for ensure_token_ids_in_sse_chunk.

    vLLM 0.20.2 puts per-chunk token IDs flat at choices[0].token_ids.
    Harbor reads choices[0].delta.provider_specific_fields.token_ids first.
    The function bridges this gap.
    """

    def test_copies_flat_token_ids_to_delta_psf(self):
        """The canonical case: vLLM emits token_ids on the choice, harbor
        reads from delta.provider_specific_fields.token_ids."""
        chunk = 'data: {"id":"c1","choices":[{"index":0,"delta":{"content":"hi"},"token_ids":[10,20]}]}\n\n'
        result = ensure_token_ids_in_sse_chunk(chunk)
        data = json.loads(result[len("data: ") :].strip())
        psf = data["choices"][0]["delta"]["provider_specific_fields"]
        assert psf["token_ids"] == [10, 20]

    def test_copies_control_token_chunk(self):
        """Empty-delta chunks (control tokens) must still carry token_ids."""
        chunk = 'data: {"id":"c1","choices":[{"index":0,"delta":{},"token_ids":[151665]}]}\n\n'
        result = ensure_token_ids_in_sse_chunk(chunk)
        data = json.loads(result[len("data: ") :].strip())
        psf = data["choices"][0]["delta"]["provider_specific_fields"]
        assert psf["token_ids"] == [151665]

    def test_passes_through_when_psf_already_present(self):
        """If delta.provider_specific_fields.token_ids already exists, don't overwrite."""
        chunk = 'data: {"choices":[{"delta":{"provider_specific_fields":{"token_ids":[1,2]}},"token_ids":[9,9]}]}\n\n'
        result = ensure_token_ids_in_sse_chunk(chunk)
        data = json.loads(result[len("data: ") :].strip())
        assert data["choices"][0]["delta"]["provider_specific_fields"]["token_ids"] == [1, 2]

    def test_passes_through_done(self):
        assert ensure_token_ids_in_sse_chunk("data: [DONE]\n\n") == "data: [DONE]\n\n"

    def test_passes_through_non_data_line(self):
        assert ensure_token_ids_in_sse_chunk(": keepalive\n\n") == ": keepalive\n\n"

    def test_passes_through_no_choices(self):
        chunk = 'data: {"id":"x","choices":[]}\n\n'
        assert ensure_token_ids_in_sse_chunk(chunk) == chunk

    def test_passes_through_no_token_ids(self):
        """Chunks without token_ids (e.g. first role chunk) pass through."""
        chunk = 'data: {"choices":[{"delta":{"role":"assistant","content":""}}]}\n\n'
        assert ensure_token_ids_in_sse_chunk(chunk) == chunk

    def test_handles_malformed_json(self):
        chunk = "data: {bad json}\n\n"
        assert ensure_token_ids_in_sse_chunk(chunk) == chunk

    def test_preserves_other_fields(self):
        """All existing fields survive the remap."""
        chunk = (
            'data: {"id":"abc","object":"chat.completion.chunk",'
            '"model":"test","prompt_token_ids":[1,2],'
            '"choices":[{"index":0,"delta":{"content":"world"},'
            '"logprobs":{"content":[{"token":"world","logprob":-0.5}]},'
            '"finish_reason":null,"token_ids":[42]}]}\n\n'
        )
        result = ensure_token_ids_in_sse_chunk(chunk)
        data = json.loads(result[len("data: ") :].strip())
        assert data["id"] == "abc"
        assert data["prompt_token_ids"] == [1, 2]
        assert data["choices"][0]["delta"]["content"] == "world"
        assert data["choices"][0]["logprobs"]["content"][0]["logprob"] == -0.5
        assert data["choices"][0]["token_ids"] == [42]  # original preserved
        assert data["choices"][0]["delta"]["provider_specific_fields"]["token_ids"] == [42]  # copied


# ---------------------------------------------------------------------------
# HTTP endpoint streaming path
# ---------------------------------------------------------------------------


class MockStreamingBackend:
    """Minimal CompletionBackend that yields canned SSE chunks."""

    model_name = "test-model"

    def __init__(self, chunks=None, non_stream_response=None):
        self._chunks = chunks or [
            # First chunk: role + prompt_token_ids (vLLM shape)
            'data: {"id":"c1","model":"test-model","object":"chat.completion.chunk",'
            '"choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}],'
            '"prompt_token_ids":[1,2,3]}\n\n',
            # Content chunk: token_ids at choice level (vLLM shape)
            'data: {"id":"c1","model":"test-model","object":"chat.completion.chunk",'
            '"choices":[{"index":0,"delta":{"content":"hello"},"token_ids":[10,20,30],"finish_reason":null}]}\n\n',
            # Final chunk: finish_reason + last token_ids
            'data: {"id":"c1","model":"test-model","object":"chat.completion.chunk",'
            '"choices":[{"index":0,"delta":{},"token_ids":[40],"finish_reason":"stop"}]}\n\n',
            "data: [DONE]\n\n",
        ]
        self._non_stream_response = non_stream_response or {
            "id": "test",
            "choices": [{"message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
        }

    async def chat_completion(self, request_payload):
        return self._non_stream_response

    async def chat_completion_stream(self, request_payload):
        # Apply the same enrichment the real vLLM engine applies per-chunk
        for chunk in self._chunks:
            yield ensure_token_ids_in_sse_chunk(chunk)

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


# ---------------------------------------------------------------------------
# _safe_sse_stream — ensures SSE body always completes
# ---------------------------------------------------------------------------


class ErrorMidStreamBackend:
    """Backend whose generator raises after the first chunk."""

    model_name = "test-model"

    async def chat_completion(self, request_payload):
        return {}

    async def chat_completion_stream(self, request_payload):
        yield 'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
        raise RuntimeError("vLLM engine exploded mid-generation")

    async def completion(self, request_payload):
        return {}


class NoDoneBackend:
    """Backend whose generator ends without [DONE]."""

    model_name = "test-model"

    async def chat_completion(self, request_payload):
        return {}

    async def chat_completion_stream(self, request_payload):
        yield 'data: {"choices":[{"delta":{"content":"hi"},"finish_reason":"stop"}]}\n\n'
        # NOTE: no data: [DONE]\n\n

    async def completion(self, request_payload):
        return {}


@pytest.fixture
def app_with_midstream_error():
    from starlette.testclient import TestClient

    app = create_app()
    set_global_state(ErrorMidStreamBackend(), None)
    yield TestClient(app)


@pytest.fixture
def app_with_no_done():
    from starlette.testclient import TestClient

    app = create_app()
    set_global_state(NoDoneBackend(), None)
    yield TestClient(app)


class TestSafeSSEStream:
    def test_mid_stream_error_yields_error_and_done(self, app_with_midstream_error):
        """Mid-generation exception must not produce an incomplete body."""
        resp = app_with_midstream_error.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        assert resp.status_code == 200
        body = resp.text
        assert "partial" in body  # first chunk was sent
        assert "Streaming error" in body  # error chunk injected
        assert "data: [DONE]" in body  # [DONE] appended

    def test_missing_done_appended(self, app_with_no_done):
        """Generator that ends without [DONE] still gets it appended."""
        resp = app_with_no_done.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        assert resp.status_code == 200
        assert "data: [DONE]" in resp.text

    def test_normal_stream_not_double_done(self, app_with_mock_backend):
        """Normal stream already containing [DONE] is not duplicated."""
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
        assert body.count("[DONE]") == 1
