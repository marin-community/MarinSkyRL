"""Shared scaffolding for the endpoint route tests.

`/tokenize`, `/v1/models` and the HTTP bridge metrics all exercise the same FastAPI app
over the same module-global backend, so each of them needs the same three things: a
stand-in for `InferenceEngineClient` carrying only what the routes read off it, a way to
put the module globals back afterwards, and a client over the app.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import skyrl_train.inference_engines.inference_engine_client_http_endpoint as endpoint_module
from skyrl_train.inference_engines.inference_engine_client_http_endpoint import create_app, set_global_state


MODEL_NAME = "Qwen/Qwen3-0.6B"


class StubBackend:
    """Stand-in for InferenceEngineClient: only what the endpoint's routes read off it."""

    def __init__(
        self,
        tokenizer=None,
        chat_template_content_format=None,
        custom_chat_template=None,
        max_model_len=None,
        model_name=MODEL_NAME,
    ):
        self.tokenizer = tokenizer
        self.chat_template_content_format = chat_template_content_format
        self.custom_chat_template = custom_chat_template
        self.max_model_len = max_model_len
        self.model_name = model_name

    async def chat_completion(self, _request):
        return {"choices": [{"message": {"content": "ok"}}]}

    async def completion(self, _request):
        return {"choices": [{"text": "ok"}]}

    async def chat_completion_stream(self, _request):
        yield "data: [DONE]\n\n"


class StubTokenizer:
    """Only what `/v1/models` reads off a tokenizer: the context length it declares."""

    def __init__(self, model_max_length):
        self.model_max_length = model_max_length

    def encode(self, text, add_special_tokens=True):
        return [1, 2, 3]


@pytest.fixture
def restore_global_state():
    """`set_global_state` writes module globals; put them back so tests stay independent."""
    previous_client = endpoint_module._global_inference_engine_client
    previous_server = endpoint_module._global_uvicorn_server
    yield
    endpoint_module._global_inference_engine_client = previous_client
    endpoint_module._global_uvicorn_server = previous_server


@pytest.fixture
def endpoint_app(restore_global_state):
    """Install a stand-in backend as the endpoint's global client and build its app."""

    def _app(app_kwargs=None, **backend_kwargs):
        set_global_state(StubBackend(**backend_kwargs), None)
        return create_app(**(app_kwargs or {}))

    return _app


@pytest.fixture
def endpoint_client(endpoint_app):
    """A `TestClient` over the endpoint, served by a stand-in backend."""

    def _client(**backend_kwargs):
        return TestClient(endpoint_app(**backend_kwargs))

    return _client
