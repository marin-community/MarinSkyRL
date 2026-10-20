"""Backend-reported prompt evidence survives routing and interrupted generation."""

import asyncio
from copy import deepcopy
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf
from ray.exceptions import ActorDiedError

from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient


class _ScriptedEngine:
    def __init__(self, steps):
        self.steps = iter(steps)

    async def generate(self, request):
        expected_prompt_ids, response = next(self.steps)
        assert request["prompt_token_ids"] == expected_prompt_ids
        if isinstance(response, Exception):
            raise response
        return deepcopy(response)


class _Tokenizer:
    def decode(self, tokens, *, skip_special_tokens):
        assert tokens == [20, 21]
        return "{}" if skip_special_tokens else "{}<|eot_id|>"


def _output(prompts, tokens, stop="stop"):
    result = {
        "responses": [f"text-{row[0]}" for row in tokens],
        "response_ids": tokens,
        "stop_reasons": [stop] * len(tokens),
        "response_logprobs": None,
    }
    if prompts is not None:
        result["prompt_ids"] = prompts
    return result


def _client(engines):
    config = OmegaConf.create(
        {
            "trainer": {"policy": {"model": {"path": "fixture"}}},
            "generator": {
                "backend": "vllm",
                "enable_http_endpoint": False,
                "http_endpoint_host": "127.0.0.1",
                "http_endpoint_port": 0,
                "weight_sync_pause_timeout_seconds": 30,
            },
        }
    )
    return InferenceEngineClient(engines, _Tokenizer(), config)


@pytest.mark.parametrize("failover", [False, True])
def test_reported_prompt_ids_follow_original_batch_order_and_failover(failover):
    # Session hashes route c/e to engine 0 and a/b to engine 1, interleaving rows.
    zero = _output([[10], [12]], [[20], [22]])
    one = _output([[11], [13]], [[21], [23]])
    engines = [
        _ScriptedEngine([([[10], [12]], ActorDiedError() if failover else zero)]),
        _ScriptedEngine([([[11], [13]], one), *([([[10], [12]], zero)] if failover else [])]),
    ]
    result = asyncio.run(
        _client(engines).generate({"prompt_token_ids": [[10], [11], [12], [13]], "session_ids": ["c", "a", "e", "b"]})
    )
    assert result["prompt_ids"] == [[10], [11], [12], [13]]
    assert result["response_ids"] == [[20], [21], [22], [23]]
    assert result["responses"] == ["text-20", "text-21", "text-22", "text-23"]


@pytest.mark.parametrize("skip_special_tokens", [False, True])
def test_retry_reports_original_backend_prefix_and_preserves_decode_options(skip_special_tokens):
    engine = _ScriptedEngine(
        [
            ([[10, 11]], _output([[10, 11]], [[20]], "abort")),
            ([[10, 11, 20]], _output([[10, 11, 20]], [[21]])),
        ]
    )
    result = asyncio.run(
        _client([engine]).generate(
            {
                "prompt_token_ids": [[10, 11]],
                "sampling_params": {"max_tokens": 2, "skip_special_tokens": skip_special_tokens},
            }
        )
    )
    assert result["prompt_ids"] == [[10, 11]]
    assert result["response_ids"] == [[20, 21]]
    assert result["responses"] == ["{}" if skip_special_tokens else "{}<|eot_id|>"]


@pytest.mark.parametrize("second_reported_prefix", [None, [[10, 11, 99]]])
def test_retry_rejects_missing_or_altered_continuation_evidence(second_reported_prefix):
    engine = _ScriptedEngine(
        [
            ([[10, 11]], _output([[10, 11]], [[20]], "abort")),
            ([[10, 11, 20]], _output(second_reported_prefix, [[21]])),
        ]
    )
    with pytest.raises(ValueError, match="prompt ID"):
        asyncio.run(_client([engine]).generate({"prompt_token_ids": [[10, 11]]}))


def test_single_failover_discards_interrupted_trajectory_and_its_prompt_evidence():
    engines = [
        _ScriptedEngine(
            [
                ([[10, 11]], _output([[10, 11]], [[20]], "abort")),
                ([[10, 11, 20]], ActorDiedError()),
            ]
        ),
        _ScriptedEngine([([[10, 11]], _output([[10, 11]], [[30]]))]),
    ]
    result = asyncio.run(_client(engines).generate({"prompt_token_ids": [[10, 11]], "session_ids": ["c"]}))
    assert result["prompt_ids"] == [[10, 11]]
    assert result["response_ids"] == [[30]]


@pytest.mark.parametrize("batch_size", [1, 2])
def test_absent_backend_evidence_is_never_replaced_with_input_echo(batch_size):
    prompts = [[10 + index] for index in range(batch_size)]
    engine = _ScriptedEngine([(prompts, _output(None, [[20]] * batch_size))])
    result = asyncio.run(_client([engine]).generate({"prompt_token_ids": prompts}))
    assert "prompt_ids" not in result


def test_batch_rejects_partial_backend_evidence():
    engines = [
        _ScriptedEngine([([[10]], _output([[10]], [[20]]))]),
        _ScriptedEngine([([[11]], _output(None, [[21]]))]),
    ]
    with pytest.raises(ValueError, match="prompt ID evidence"):
        asyncio.run(_client(engines).generate({"prompt_token_ids": [[10], [11]]}))


def test_vllm_postprocess_reports_request_output_prompt_tokens():
    pytest.importorskip("vllm")
    from skyrl_train.inference_engines.vllm.vllm_engine import AsyncVLLMInferenceEngine

    request_output = SimpleNamespace(
        prompt_token_ids=[10, 11],
        prompt_logprobs=None,
        outputs=[
            SimpleNamespace(text="answer", finish_reason="stop", token_ids=[20], routed_experts=None, logprobs=None)
        ],
    )
    result = AsyncVLLMInferenceEngine._postprocess_outputs(object(), [request_output])
    assert result["prompt_ids"] == [[10, 11]]
    assert result["response_ids"] == [[20]]
    request_output.prompt_token_ids = None
    result = AsyncVLLMInferenceEngine._postprocess_outputs(object(), [request_output])
    assert "prompt_ids" not in result
