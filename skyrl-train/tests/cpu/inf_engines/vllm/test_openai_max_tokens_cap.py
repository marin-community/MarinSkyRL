"""The OpenAI request path bounds completions by ``max_generate_length`` like the native path does.

Terminus-2 sends no ``max_tokens`` on an ordinary turn; without the cap vLLM generated into the whole
remaining context and one runaway turn set the wave time of every Snowball training step (2026-09-05).

Run with:
  uv run --isolated --group dev --extra cpu pytest tests/cpu/inf_engines/vllm/test_openai_max_tokens_cap.py
"""

from skyrl_train.inference_engines.vllm.utils import (
    apply_openai_max_tokens_cap,
    is_openai_output_budget_overflow,
    openai_error_message,
    pop_vllm_wrapper_kwargs,
)


def test_cap_fills_in_missing_max_tokens():
    body = {"model": "m", "messages": [], "temperature": 1.0}
    assert apply_openai_max_tokens_cap(body, 16384) is True
    assert body["max_tokens"] == 16384
    assert "max_completion_tokens" not in body


def test_cap_lowers_a_larger_client_limit_on_every_key_the_client_used():
    body = {"max_tokens": 60000}
    assert apply_openai_max_tokens_cap(body, 16384) is True
    assert body == {"max_tokens": 16384}

    body = {"max_completion_tokens": 60000}
    assert apply_openai_max_tokens_cap(body, 16384) is True
    assert body == {"max_tokens": 16384, "max_completion_tokens": 16384}


def test_smaller_client_limit_is_left_alone():
    """Terminus-2's summarization calls pass their own reserve (4096); the cap must not raise it."""
    body = {"max_tokens": 4096}
    assert apply_openai_max_tokens_cap(body, 16384) is False
    assert body == {"max_tokens": 4096}

    body = {"max_tokens": 16384}
    assert apply_openai_max_tokens_cap(body, 16384) is False


def test_no_cap_when_max_generate_length_is_unset_or_zero():
    for cap in (None, 0, -1):
        body = {"max_tokens": 60000}
        assert apply_openai_max_tokens_cap(body, cap) is False
        assert body == {"max_tokens": 60000}


def test_explicit_null_client_value_counts_as_unset():
    body = {"max_tokens": None}
    assert apply_openai_max_tokens_cap(body, 8192) is True
    assert body["max_tokens"] == 8192


def test_error_message_is_read_from_both_error_shapes_and_not_from_success():
    nested = {"error": {"message": "boom", "type": "BadRequest", "code": 400}}
    flat = {"object": "error", "message": "boom", "type": "BadRequest", "code": 400}
    success = {"id": "c1", "object": "chat.completion", "choices": [{"message": {"content": "hi"}}]}
    assert openai_error_message(nested) == "boom"
    assert openai_error_message(flat) == "boom"
    assert openai_error_message(success) is None
    assert openai_error_message("not a dict") is None


def test_output_budget_overflow_matches_vllm_validation_text_only():
    over_budget = (
        "This model's maximum context length is 65536 tokens. However, you requested 16384 output "
        "tokens and your prompt contains 49200 input tokens, for a total of 65584 tokens. "
        "Please reduce the length of the input prompt or the number of requested output tokens."
    )
    assert is_openai_output_budget_overflow(over_budget) is True
    # A prompt that does not fit even by itself is a real context overflow, not a cap artefact.
    assert (
        is_openai_output_budget_overflow("You passed 70000 input tokens but the context length is only 65536") is False
    )
    assert is_openai_output_budget_overflow("CUDA error: an illegal memory access") is False
    assert is_openai_output_budget_overflow(None) is False
    assert is_openai_output_budget_overflow("") is False


def test_cap_switch_is_popped_from_engine_kwargs_as_a_bool():
    """``generator.openai_max_tokens_cap`` rides engine_init_kwargs and must never reach vLLM's engine args."""
    engine_kwargs = {
        "openai_max_tokens_cap": 1,
        "openai_sampling_params": {"max_generate_length": 16384},
        "other": "keep",
    }
    wrapper_kwargs = pop_vllm_wrapper_kwargs(engine_kwargs)
    assert wrapper_kwargs["openai_max_tokens_cap"] is True
    assert wrapper_kwargs["openai_sampling_params"] == {"max_generate_length": 16384}
    assert engine_kwargs == {"other": "keep"}

    assert "openai_max_tokens_cap" not in pop_vllm_wrapper_kwargs({"other": "keep"})
