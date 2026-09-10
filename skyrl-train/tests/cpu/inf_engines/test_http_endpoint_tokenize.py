"""Contract tests for `POST /tokenize` as harbor's TITO transport uses it.

harbor (`harbor.llms.vllm_tokenization.tokenize_chat`) POSTs `{"model", "messages",
"add_generation_prompt", **template_options}` and reads `tokens`. Two things must hold for
exact-token continuation to splice a prompt the engines actually serve: the ids equal the
trainer-side chat render of the same messages, and harbor's assistant-content-end probe finds
the real between-turn text.
"""

from http import HTTPStatus

import pytest
from fastapi.testclient import TestClient
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.inference_engine_client_http_endpoint import create_app, set_global_state
from tests.cpu.inf_engines.conftest import MODEL_NAME

# The body harbor's TITO transport POSTs (lite_llm._TOKEN_IN_TOKEN_OUT_TEMPLATE_BASE + _FOLLOWUP).
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


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


@pytest.fixture
def client(tokenizer):
    config = OmegaConf.create(
        {
            "trainer": {"policy": {"model": {"path": MODEL_NAME}}},
            "generator": {
                "backend": "vllm",
                "enable_http_endpoint": False,
                "http_endpoint_host": "127.0.0.1",
                "http_endpoint_port": 0,
                "engine_init_kwargs": {},
            },
        }
    )
    set_global_state(InferenceEngineClient(engines=[], tokenizer=tokenizer, full_config=config), None)
    return TestClient(create_app())


def _tokenize(client: TestClient, **body):
    response = client.post("/tokenize", json=body)
    assert response.status_code == HTTPStatus.OK.value, response.text
    return response.json()["tokens"]


def test_messages_match_the_engine_clients_own_chat_render(tokenizer, client):
    """The ids equal what InferenceEngineClient.generate() renders for the same messages."""
    ids = _tokenize(client, messages=HARBOR_TITO_MESSAGES, add_generation_prompt=True)

    generate_path_ids = tokenizer.apply_chat_template(
        [HARBOR_TITO_MESSAGES],
        add_generation_prompt=True,
        add_special_tokens=False,
        return_dict=True,
        tokenize=True,
    )["input_ids"][0]
    assert ids == list(generate_path_ids)


def test_harbor_tito_probe_locates_the_assistant_content_end(tokenizer, client):
    """Harbor's continuation algorithm, run verbatim against this route: two renders that
    differ only in assistant content, longest common tail, what precedes it is the
    assistant-content end, and the suffix after it must be the real between-turn text."""
    base_ids = _tokenize(client, messages=HARBOR_TITO_MESSAGES, add_generation_prompt=True)
    alt_ids = _tokenize(client, messages=HARBOR_TITO_MESSAGES_ALT, add_generation_prompt=True)

    shared = 0
    while shared < min(len(base_ids), len(alt_ids)) and base_ids[-1 - shared] == alt_ids[-1 - shared]:
        shared += 1
    assistant_content_end = len(base_ids) - shared

    assert 0 < assistant_content_end < len(base_ids)
    assert tokenizer.decode(base_ids[:assistant_content_end]).endswith("ok")
    assert tokenizer.decode(base_ids[assistant_content_end:]) == (
        "<|im_end|>\n<|im_start|>user\nI am another user.<|im_end|>\n<|im_start|>assistant\n"
    )
