"""
CPU tests for the endpoint's vLLM-compatible ``GET /v1/models`` route.

harbor's proactive context guard learns the served context length from this route
(``harbor/src/harbor/llms/lite_llm.py::LiteLLM._get_vllm_max_model_len``: GET ``/v1/models``,
then the first entry in ``data`` with a truthy ``max_model_len``). When the probe fails it falls
back to litellm's model registry, which has no entry for an RL checkpoint served under a hashed
``served_model_name`` and answers 1e6 — the guard then lets an over-long prompt through and the
engine rejects it instead. So the tests below pin the wire shape, the exact field harbor reads,
and the invariant that the number agrees with the one ``/tokenize`` reports.

Run with:
  uv run --isolated --group dev --extra cpu pytest tests/cpu/inf_engines/test_http_endpoint_models.py
"""

from http import HTTPStatus

from fastapi.testclient import TestClient

from tests.cpu.inf_engines.conftest import MODEL_NAME, StubTokenizer


def _models(client: TestClient):
    response = client.get("/v1/models")
    assert response.status_code == HTTPStatus.OK.value, response.text
    return response.json()


def _harbor_reads_max_model_len(body):
    """The parse harbor performs, copied from `LiteLLM._get_vllm_max_model_len`."""
    limit = None
    for entry in body.get("data", []):
        mml = entry.get("max_model_len")
        if mml:
            limit = int(mml)
            break
    return limit


def test_serves_the_vllm_model_list_shape(endpoint_client):
    client = endpoint_client(tokenizer=StubTokenizer(40960), max_model_len=32768)

    body = _models(client)

    assert body["object"] == "list"
    assert len(body["data"]) == 1
    entry = body["data"][0]
    assert entry["id"] == MODEL_NAME
    assert entry["object"] == "model"
    assert entry["owned_by"] == "skyrl"
    assert isinstance(entry["created"], int) and entry["created"] > 0
    assert entry["max_model_len"] == 32768


def test_harbor_reads_the_engines_max_model_len(endpoint_client):
    """The field harbor's context guard actually consumes, read the way it reads it."""
    client = endpoint_client(tokenizer=StubTokenizer(40960), max_model_len=32768)

    assert _harbor_reads_max_model_len(_models(client)) == 32768


def test_reported_length_matches_the_tokenize_route(endpoint_client):
    """Both routes answer harbor about the same server, so they must not disagree: `/tokenize`
    feeds terminus-2's token counter and `/v1/models` feeds the context guard that counter is
    checked against."""
    client = endpoint_client(tokenizer=StubTokenizer(40960), max_model_len=32768)

    models_len = _models(client)["data"][0]["max_model_len"]
    tokenize_response = client.post("/tokenize", json={"model": MODEL_NAME, "prompt": "hi"})

    assert tokenize_response.status_code == HTTPStatus.OK.value, tokenize_response.text
    assert tokenize_response.json()["max_model_len"] == models_len


def test_served_model_name_is_the_id(endpoint_client):
    """RL checkpoints are served under a hashed `served_model_name`; that is the id harbor and
    litellm address the endpoint with, so it is the id reported here."""
    client = endpoint_client(tokenizer=StubTokenizer(40960), max_model_len=32768, model_name="a1b2c3d4")

    assert _models(client)["data"][0]["id"] == "a1b2c3d4"


def test_created_is_stable_across_requests(endpoint_client):
    """OpenAI's `created` is when the server started serving the model, not per request."""
    client = endpoint_client(tokenizer=StubTokenizer(40960), max_model_len=32768)

    assert _models(client)["data"][0]["created"] == _models(client)["data"][0]["created"]
