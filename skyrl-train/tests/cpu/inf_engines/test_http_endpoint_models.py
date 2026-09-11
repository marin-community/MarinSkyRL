"""
CPU tests for the endpoint's vLLM-compatible ``GET /v1/models`` route.

The route exposes the served identity and context limit used by clients for proactive
context-window checks.

Run with:
  uv run --isolated --group dev --extra cpu pytest tests/cpu/inf_engines/test_http_endpoint_models.py
"""

from http import HTTPStatus
from types import SimpleNamespace

from fastapi.testclient import TestClient

import skyrl_train.inference_engines.inference_engine_client_http_endpoint as endpoint_module
from skyrl_train.inference_engines.inference_engine_client_http_endpoint import create_app


def test_serves_the_vllm_model_list_shape(monkeypatch):
    model_name = "Qwen/Qwen3-0.6B"
    backend = SimpleNamespace(model_name=model_name, max_model_len=32768)
    monkeypatch.setattr(endpoint_module, "_global_inference_engine_client", backend)

    response = TestClient(create_app()).get("/v1/models")

    assert response.status_code == HTTPStatus.OK.value, response.text
    body = response.json()
    assert body["object"] == "list"
    assert len(body["data"]) == 1
    entry = body["data"][0]
    assert entry["id"] == model_name
    assert entry["object"] == "model"
    assert entry["owned_by"] == "skyrl"
    assert isinstance(entry["created"], int)
    assert entry["max_model_len"] == 32768
