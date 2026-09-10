"""
CPU tests for the endpoint's vLLM-compatible ``POST /tokenize`` route.

The route exists because harbor's TITO (token-in-token-out) transport renders every
continuation prompt through it and has no fallback: pointed at a SkyRL endpoint without
it, every rollout dies on ``404 Not Found for url .../tokenize`` (job 1738272). So the
tests below pin two things — the wire contract vLLM defines, and the invariant that the
ids come from the SAME render the serving path uses (the engine's tokenizer, its custom
chat template, its pinned content format). If the render drifts, TITO silently feeds the
engines a prompt they would never have produced themselves.

Run with:
  uv run --isolated --group dev --extra cpu pytest tests/cpu/inf_engines/test_http_endpoint_tokenize.py
"""

from http import HTTPStatus

import pytest
from fastapi.testclient import TestClient
from transformers import AutoTokenizer

from tests.cpu.inf_engines.conftest import MODEL_NAME

# The body harbor's TITO transport POSTs, copied from
# harbor/src/harbor/llms/lite_llm.py (_TOKEN_IN_TOKEN_OUT_TEMPLATE_BASE + _FOLLOWUP, sent
# by harbor.llms.vllm_tokenization.tokenize_chat as
# {"model", "messages", "add_generation_prompt", **template_options}).
HARBOR_TITO_MESSAGES = [
    {"role": "user", "content": "I am a user."},
    {"role": "assistant", "content": "ok", "reasoning_content": "thinking"},
    {"role": "user", "content": "I am another user."},
]
HARBOR_TITO_MESSAGES_ALT = [
    {"role": "user", "content": "I am a user."},
    {"role": "assistant", "content": "different", "reasoning_content": "thinking"},
    {"role": "user", "content": "I am another user."},
]

# A template that is deliberately unlike Qwen3's, so "did the render use it?" is decidable.
CUSTOM_CHAT_TEMPLATE = (
    "{% for message in messages %}[{{ message['role'] }}]{{ message['content'] }}{% endfor %}"
    "{% if add_generation_prompt %}[assistant]{% endif %}"
)


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def _tokenize(client: TestClient, **body):
    response = client.post("/tokenize", json={"model": MODEL_NAME, **body})
    assert response.status_code == HTTPStatus.OK.value, response.text
    return response.json()


def _local_chat_ids(tokenizer, messages, chat_template=None, **template_kwargs):
    """The reference render: text from the chat template, then encode. What vLLM does."""
    rendered = tokenizer.apply_chat_template(
        messages,
        chat_template=chat_template,
        add_generation_prompt=True,
        tokenize=False,
        **template_kwargs,
    )
    return tokenizer.encode(rendered, add_special_tokens=False)


# ---------------------------------------------------------------------------
# Wire contract
# ---------------------------------------------------------------------------


def test_prompt_form_matches_encode(tokenizer, endpoint_client):
    """The completion form encodes the raw string, with add_special_tokens defaulting True."""
    client = endpoint_client(tokenizer=tokenizer)

    body = _tokenize(client, prompt="Hello there")

    assert body["tokens"] == tokenizer.encode("Hello there", add_special_tokens=True)
    assert body["count"] == len(body["tokens"])
    assert body["token_strs"] is None


class _RecordingTokenizer:
    """Records how it was called. Qwen3 adds no special tokens, so the flag needs a witness."""

    model_max_length = 4096

    def __init__(self):
        self.encode_calls = []
        self.template_calls = []

    def apply_chat_template(self, conversation, **kwargs):
        self.template_calls.append(kwargs)
        return "rendered"

    def encode(self, text, add_special_tokens=True):
        self.encode_calls.append({"text": text, "add_special_tokens": add_special_tokens})
        return [1, 2, 3]


@pytest.mark.parametrize(
    "body, expected",
    [
        # vLLM's TokenizeCompletionRequest defaults add_special_tokens to True, its
        # TokenizeChatRequest to False (the chat template emits them itself).
        ({"prompt": "hi"}, True),
        ({"prompt": "hi", "add_special_tokens": False}, False),
        ({"messages": HARBOR_TITO_MESSAGES}, False),
        ({"messages": HARBOR_TITO_MESSAGES, "add_special_tokens": True}, True),
    ],
)
def test_add_special_tokens_defaults_per_request_form(endpoint_client, body, expected):
    recorder = _RecordingTokenizer()
    client = endpoint_client(tokenizer=recorder)

    _tokenize(client, **body)

    assert recorder.encode_calls[-1]["add_special_tokens"] is expected


def test_response_reports_count_max_model_len_and_token_strs(tokenizer, endpoint_client):
    client = endpoint_client(tokenizer=tokenizer, max_model_len=32768)

    body = _tokenize(client, messages=HARBOR_TITO_MESSAGES, return_token_strs=True)

    assert body["count"] == len(body["tokens"])
    assert body["max_model_len"] == 32768
    assert body["token_strs"] == tokenizer.convert_ids_to_tokens(body["tokens"])


def test_max_model_len_falls_back_to_the_tokenizer(tokenizer, endpoint_client):
    """No engine-configured length: report the tokenizer's, which Qwen3 declares."""
    client = endpoint_client(tokenizer=tokenizer)

    body = _tokenize(client, prompt="hi")

    assert body["max_model_len"] == tokenizer.model_max_length


def test_max_model_len_is_null_for_a_sentinel_length(tokenizer, endpoint_client):
    """HF's "no limit" sentinel (1e30) is not a context length; report null instead."""

    class _NoLimitTokenizer:
        model_max_length = int(1e30)

        def encode(self, text, add_special_tokens=True):
            return [1, 2, 3]

    client = endpoint_client(tokenizer=_NoLimitTokenizer())

    body = _tokenize(client, prompt="hi")

    assert body["max_model_len"] is None


# ---------------------------------------------------------------------------
# The render must be the serving path's render
# ---------------------------------------------------------------------------


def test_messages_match_the_engine_clients_own_chat_render(tokenizer, endpoint_client):
    """The ids equal what InferenceEngineClient.generate() renders for the same messages.

    That call (inference_engine_client.py, `prompt_token_ids = self.tokenizer.
    apply_chat_template(...)`) is the trainer-side render of a chat prompt. `/tokenize`
    disagreeing with it would mean TITO splices a prompt the engines never serve.
    """
    client = endpoint_client(tokenizer=tokenizer)

    body = _tokenize(client, messages=HARBOR_TITO_MESSAGES, add_generation_prompt=True)

    generate_path_ids = tokenizer.apply_chat_template(
        [HARBOR_TITO_MESSAGES],
        add_generation_prompt=True,
        add_special_tokens=False,
        return_dict=True,
        tokenize=True,
    )["input_ids"][0]
    assert body["tokens"] == list(generate_path_ids)


def test_add_generation_prompt_false_stops_before_the_generation_prompt(tokenizer, endpoint_client):
    client = endpoint_client(tokenizer=tokenizer)

    with_prompt = _tokenize(client, messages=HARBOR_TITO_MESSAGES, add_generation_prompt=True)["tokens"]
    without_prompt = _tokenize(client, messages=HARBOR_TITO_MESSAGES, add_generation_prompt=False)["tokens"]

    assert without_prompt == _local_chat_ids(tokenizer, HARBOR_TITO_MESSAGES, chat_template=None)[: len(without_prompt)]
    assert len(without_prompt) < len(with_prompt)
    assert with_prompt[: len(without_prompt)] == without_prompt


def test_chat_template_kwargs_reach_the_template(tokenizer, endpoint_client):
    """Qwen3's template branches on enable_thinking, so the ids must change with it."""
    client = endpoint_client(tokenizer=tokenizer)

    thinking = _tokenize(
        client,
        messages=HARBOR_TITO_MESSAGES,
        add_generation_prompt=True,
        chat_template_kwargs={"enable_thinking": True},
    )["tokens"]
    no_thinking = _tokenize(
        client,
        messages=HARBOR_TITO_MESSAGES,
        add_generation_prompt=True,
        chat_template_kwargs={"enable_thinking": False},
    )["tokens"]

    assert thinking != no_thinking
    assert thinking == _local_chat_ids(tokenizer, HARBOR_TITO_MESSAGES, enable_thinking=True)
    assert no_thinking == _local_chat_ids(tokenizer, HARBOR_TITO_MESSAGES, enable_thinking=False)


def test_backend_custom_chat_template_is_used(tokenizer, endpoint_client):
    """A run serving `custom_chat_template_chat_completion_path` must tokenize with it."""
    client = endpoint_client(tokenizer=tokenizer, custom_chat_template=CUSTOM_CHAT_TEMPLATE)

    body = _tokenize(client, messages=HARBOR_TITO_MESSAGES, add_generation_prompt=True)

    assert body["tokens"] == _local_chat_ids(tokenizer, HARBOR_TITO_MESSAGES, chat_template=CUSTOM_CHAT_TEMPLATE)
    assert body["tokens"] != _local_chat_ids(tokenizer, HARBOR_TITO_MESSAGES)


def test_request_chat_template_overrides_the_backend(tokenizer, endpoint_client):
    """vLLM resolves `request.chat_template or self.chat_template`; so do we."""
    client = endpoint_client(tokenizer=tokenizer, custom_chat_template="{{ 'ignored' }}")

    body = _tokenize(
        client,
        messages=HARBOR_TITO_MESSAGES,
        add_generation_prompt=True,
        chat_template=CUSTOM_CHAT_TEMPLATE,
    )

    assert body["tokens"] == _local_chat_ids(tokenizer, HARBOR_TITO_MESSAGES, chat_template=CUSTOM_CHAT_TEMPLATE)


def test_string_content_format_flattens_parts(tokenizer, endpoint_client):
    """With the format pinned to `string` (the Snowball arms), parts render as their text.

    Passing the raw list into the template instead would render its Python repr.
    """
    client = endpoint_client(tokenizer=tokenizer, chat_template_content_format="string")
    parts = [{"role": "user", "content": [{"type": "text", "text": "I am a user."}]}]
    plain = [{"role": "user", "content": "I am a user."}]

    assert _tokenize(client, messages=parts)["tokens"] == _tokenize(client, messages=plain)["tokens"]


def test_openai_content_format_wraps_strings_as_parts(tokenizer, endpoint_client):
    """With the format pinned to `openai`, a plain string is rendered as a text part."""
    client = endpoint_client(tokenizer=tokenizer, chat_template_content_format="openai")
    plain = [{"role": "user", "content": "I am a user."}]
    as_parts = [{"role": "user", "content": [{"type": "text", "text": "I am a user."}]}]

    assert _tokenize(client, messages=plain)["tokens"] == _local_chat_ids(tokenizer, as_parts)
    # Qwen3's template reads `content` as a string, so the two renders really do differ:
    # the assertion above would not hold if the wrapping had been skipped.
    assert _local_chat_ids(tokenizer, as_parts) != _local_chat_ids(tokenizer, plain)


# ---------------------------------------------------------------------------
# Harbor's own request shapes
# ---------------------------------------------------------------------------


def test_harbor_tito_probe_locates_the_assistant_content_end(tokenizer, endpoint_client):
    """Harbor's continuation algorithm, run verbatim against this route.

    It renders two conversations that differ only in assistant content, takes their longest
    common tail, and calls what precedes it the assistant-content end. The suffix after that
    point is what gets spliced onto the sampled tokens, so it must be the real between-turn
    text: `<|im_end|>` then the next user turn then the generation prompt.
    """
    client = endpoint_client(tokenizer=tokenizer)

    base_ids = _tokenize(client, messages=HARBOR_TITO_MESSAGES, add_generation_prompt=True)["tokens"]
    alt_ids = _tokenize(client, messages=HARBOR_TITO_MESSAGES_ALT, add_generation_prompt=True)["tokens"]

    shared = 0
    while shared < min(len(base_ids), len(alt_ids)) and base_ids[-1 - shared] == alt_ids[-1 - shared]:
        shared += 1
    assistant_content_end = len(base_ids) - shared

    assert 0 < assistant_content_end < len(base_ids)
    assert tokenizer.decode(base_ids[:assistant_content_end]).endswith("ok")
    assert tokenizer.decode(base_ids[assistant_content_end:]) == (
        "<|im_end|>\n<|im_start|>user\nI am another user.<|im_end|>\n<|im_start|>assistant\n"
    )


def test_terminus_token_count_body(tokenizer, endpoint_client):
    """Terminus-2's counter POSTs only {"model", "messages"} and reads `count`."""
    client = endpoint_client(tokenizer=tokenizer)

    response = client.post("/tokenize", json={"model": MODEL_NAME, "messages": HARBOR_TITO_MESSAGES})

    assert response.status_code == HTTPStatus.OK.value
    assert response.json()["count"] == len(_local_chat_ids(tokenizer, HARBOR_TITO_MESSAGES))


# ---------------------------------------------------------------------------
# Rejections
# ---------------------------------------------------------------------------


def test_returns_501_without_a_tokenizer(endpoint_client):
    """A backend that carries no tokenizer says so, rather than 500-ing or 404-ing."""
    client = endpoint_client(tokenizer=None)

    response = client.post("/tokenize", json={"model": MODEL_NAME, "messages": HARBOR_TITO_MESSAGES})

    assert response.status_code == HTTPStatus.NOT_IMPLEMENTED.value
    assert "without a tokenizer" in response.json()["error"]["message"]


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"prompt": "hi", "messages": HARBOR_TITO_MESSAGES},
        {"messages": []},
        {"messages": "not a list"},
        {"prompt": ["not", "a", "string"]},
        {"messages": HARBOR_TITO_MESSAGES, "add_generation_prompt": True, "continue_final_message": True},
    ],
)
def test_malformed_bodies_are_rejected(tokenizer, endpoint_client, body):
    client = endpoint_client(tokenizer=tokenizer)

    response = client.post("/tokenize", json={"model": MODEL_NAME, **body})

    assert response.status_code == HTTPStatus.BAD_REQUEST.value


def test_non_object_and_unparseable_bodies_are_rejected(tokenizer, endpoint_client):
    client = endpoint_client(tokenizer=tokenizer)

    assert client.post("/tokenize", json=["not", "an", "object"]).status_code == HTTPStatus.BAD_REQUEST.value
    assert client.post("/tokenize", content=b"{not json").status_code == HTTPStatus.BAD_REQUEST.value


def test_model_mismatch_is_rejected(tokenizer, endpoint_client):
    client = endpoint_client(tokenizer=tokenizer)

    response = client.post("/tokenize", json={"model": "some-other-model", "messages": HARBOR_TITO_MESSAGES})

    assert response.status_code == HTTPStatus.BAD_REQUEST.value
    assert "Model name mismatch" in response.json()["error"]["message"]


def test_a_broken_template_is_a_client_error(tokenizer, endpoint_client):
    """Deterministic failures answer 400 so harbor does not retry them forever."""
    client = endpoint_client(tokenizer=tokenizer)

    response = client.post(
        "/tokenize",
        json={"model": MODEL_NAME, "messages": HARBOR_TITO_MESSAGES, "chat_template": "{% for %}"},
    )

    assert response.status_code == HTTPStatus.BAD_REQUEST.value
    assert "Error when handling /tokenize request in SkyRL" in response.json()["error"]["message"]
