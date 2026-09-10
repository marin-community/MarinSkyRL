import json
from typing import Any, Dict, Optional, Protocol

from skyrl_train.config.behavior_logprobs import (
    ROLLOUT_LOGPROB_VALIDATION_KEY,
    validate_behavior_logprob_sampling,
)

# Values vLLM's chat renderer accepts for ``chat_template_content_format``.
CHAT_TEMPLATE_CONTENT_FORMATS = ("auto", "string", "openai")


class PrefixCacheStatsLike(Protocol):
    """The token counters vLLM's `PrefixCacheStats` carries for one scheduler iteration."""

    queries: int
    hits: int


def prefix_cache_hit_rate_percent(stats: PrefixCacheStatsLike) -> float | None:
    """Share of queried prefix tokens that were already cached, as a percentage.

    `PrefixCacheStats` counts tokens, not requests: `queries` is how many were looked up and
    `hits` how many were served from cache.

    Returns None for an iteration that queried nothing, which has no rate to report rather
    than a rate of zero.
    """
    if stats.queries == 0:
        return None
    return stats.hits / stats.queries * 100.0


class PrefixCacheHitRateAccumulator:
    """Peak and per-iteration samples of the prefix cache hit rate for one engine.

    Owns which scheduler iterations are sampled at all. Only iterations that queried prefix
    tokens are: `prefix_cache_stats` is a delta that admission writes, decode-only iterations
    outnumber admissions heavily, and scoring those as zeroes would pin the median near 0.0.
    vLLM's `CachingMetrics.observe` skips the same iterations, keyed on `requests`, which
    `PrefixCacheStats.record` bumps alongside `queries`.
    """

    def __init__(self) -> None:
        self.peak: float = 0.0
        self.samples: list[float] = []

    def observe(self, stats: PrefixCacheStatsLike | None, is_active: bool) -> None:
        """Fold one scheduler iteration in, where `is_active` means it had queued or running work."""
        if stats is None:
            return
        rate = prefix_cache_hit_rate_percent(stats)
        if rate is None:
            return
        self.peak = max(self.peak, rate)
        if is_active:
            self.samples.append(rate)


def pop_vllm_wrapper_kwargs(engine_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove SkyRL serving options before passing engine_kwargs to vLLM.
    """
    wrapper_kwargs: Dict[str, Any] = {}

    enable_auto_tools = engine_kwargs.pop("enable_auto_tools", engine_kwargs.pop("enable_auto_tool_choice", None))
    if enable_auto_tools is not None:
        wrapper_kwargs["enable_auto_tools"] = bool(enable_auto_tools)

    tool_parser = engine_kwargs.pop("tool_parser", engine_kwargs.pop("tool_call_parser", None))
    if tool_parser is not None:
        wrapper_kwargs["tool_parser"] = tool_parser

    # Bound OpenAI-path completions at sampling_params.max_generate_length (see apply_openai_max_tokens_cap)
    cap_flag = engine_kwargs.pop("openai_max_tokens_cap", None)
    if cap_flag is not None:
        wrapper_kwargs["openai_max_tokens_cap"] = bool(cap_flag)

    # Sampling params for OpenAI-style requests (Harbor terminal-bench rollouts)
    openai_sampling = engine_kwargs.pop("openai_sampling_params", None)
    if openai_sampling is not None:
        wrapper_kwargs["openai_sampling_params"] = openai_sampling

    # How vLLM's renderer hands message content to the chat template. Unset leaves vLLM's
    # ``auto``, which sniffs the template and may pick the OpenAI parts format; a template
    # can render parts differently from a plain string (Snowball's emits an extra newline
    # after each assistant EOT), so ``string`` / ``openai`` pin the format via
    # ``++generator.engine_init_kwargs.chat_template_content_format=...``.
    content_format = engine_kwargs.pop("chat_template_content_format", None)
    if content_format is not None:
        if content_format not in CHAT_TEMPLATE_CONTENT_FORMATS:
            raise ValueError(
                f"chat_template_content_format must be one of {CHAT_TEMPLATE_CONTENT_FORMATS}, got {content_format!r}"
            )
        wrapper_kwargs["chat_template_content_format"] = content_format

    if ROLLOUT_LOGPROB_VALIDATION_KEY in engine_kwargs:
        wrapper_kwargs[ROLLOUT_LOGPROB_VALIDATION_KEY] = engine_kwargs.pop(ROLLOUT_LOGPROB_VALIDATION_KEY)

    return wrapper_kwargs


def ensure_token_ids_in_sse_chunk(sse_chunk: str) -> str:
    """Copy per-chunk token IDs to harbor's canonical streaming location.

    vLLM 0.20.2 puts delta token IDs flat on the choice at
    ``choices[0].token_ids`` (sibling to ``delta``, gated by
    ``return_token_ids=True``).  Harbor's ``_chunk_completion_token_ids``
    checks ``choices[0].delta.provider_specific_fields.token_ids`` first
    (canonical streaming path), then falls back to ``choices[0].token_ids``.

    This function copies the flat field into the delta's
    ``provider_specific_fields`` so harbor finds it on the FIRST path —
    avoiding edge cases where the fallback misses control-token chunks
    (empty-delta chunks that still carry token IDs).
    """
    if not sse_chunk.startswith("data: "):
        return sse_chunk
    payload = sse_chunk[len("data: ") :]
    if payload.strip() == "[DONE]":
        return sse_chunk
    try:
        data = json.loads(payload)
        choices = data.get("choices")
        if not choices:
            return sse_chunk
        choice = choices[0]
        token_ids = choice.get("token_ids")
        if not isinstance(token_ids, list):
            return sse_chunk
        delta = choice.setdefault("delta", {})
        psf = delta.setdefault("provider_specific_fields", {})
        if "token_ids" not in psf:
            psf["token_ids"] = token_ids
            return f"data: {json.dumps(data)}\n\n"
    except (json.JSONDecodeError, IndexError, KeyError):
        pass
    return sse_chunk


def apply_openai_sampling(
    body: Dict[str, Any], sampling_params: Dict[str, Any], validate_rollout_logprob_sampling: bool
) -> None:
    """Apply generator sampling overrides and validate training probabilities."""
    body.update(
        {
            "temperature": sampling_params.get("temperature", 1.0),
            "top_p": sampling_params.get("top_p", 1.0),
            "top_k": sampling_params.get("top_k", -1),
            "min_p": sampling_params.get("min_p", 0.0),
        }
    )
    # Completion logprobs=0 requests sampled-token probabilities; False disables chat logprobs.
    logprobs = body.get("logprobs")
    if validate_rollout_logprob_sampling and logprobs is not None and logprobs is not False:
        validate_behavior_logprob_sampling(body)


_OPENAI_MAX_TOKENS_KEYS = ("max_tokens", "max_completion_tokens")


def apply_openai_max_tokens_cap(body: Dict[str, Any], max_generate_length: Optional[int]) -> bool:
    """Bound an OpenAI-style request's completion length by the generator's ``max_generate_length``.

    The native generation path already turns ``sampling_params.max_generate_length`` into vLLM's
    ``max_tokens`` (``get_sampling_params_for_backend``); the OpenAI path used by Harbor rollouts
    forwarded whatever the agent sent. Terminus-2 sends no ``max_tokens`` on an ordinary turn, so
    vLLM defaulted to "fill the remaining context": a runaway completion decoded 50k+ tokens for up
    to 17 minutes, died on the context limit, and that one tail set the wave time of every training
    step (7.5 % of Snowball R2E-Gym trajectories, 2026-09-05). Under the cap such a turn ends with
    ``finish_reason="length"`` after ``max_generate_length`` tokens and Terminus-2 recovers it
    (``_recover_output_overflow``) instead of the trial dying on context.

    Writes the smaller of the cap and the client's own limit back to the keys the client used
    (``max_tokens`` when it sent neither; vLLM prefers ``max_completion_tokens`` when both are set).
    A client limit already at or below the cap is left untouched. Returns True when the body changed.
    Opt-in per run via ``generator.openai_max_tokens_cap=true`` (the engine passes ``None`` as the cap
    when it is off), so a mid-campaign code pull leaves running recipes and eval probes unchanged.
    """
    if not max_generate_length or int(max_generate_length) <= 0:
        return False
    cap = int(max_generate_length)
    requested = [int(body[k]) for k in _OPENAI_MAX_TOKENS_KEYS if body.get(k) is not None]
    if requested and min(requested) <= cap:
        return False
    body["max_tokens"] = cap
    if "max_completion_tokens" in body:
        body["max_completion_tokens"] = cap
    return True


def openai_error_message(response: Any) -> Optional[str]:
    """The message of an OpenAI-style error response dict, or None for a non-error response.

    Accepts both shapes the engine returns: vLLM >= 0.10 nests the fields under ``error``
    (``ErrorInfo``); older vLLM and the flat fallback put ``message`` at the top level.
    """
    if not isinstance(response, dict):
        return None
    err = response.get("error")
    if isinstance(err, dict):
        return str(err.get("message", ""))
    if "choices" not in response and (response.get("object") == "error" or "message" in response):
        return str(response.get("message", ""))
    return None


def is_openai_output_budget_overflow(message: Optional[str]) -> bool:
    """True for vLLM's validation error that the prompt plus the requested output does not fit.

    vLLM derives ``max_input_tokens = max_model_len - max_output_tokens`` and rejects a longer
    prompt with "This model's maximum context length is N tokens. However, you requested M output
    tokens and your prompt contains ..." (``vllm/renderers/params.py``). A prompt that fits by
    itself but leaves fewer than ``max_generate_length`` tokens is exactly the case the uncapped
    request used to serve (vLLM then generates into the remaining room), so the caller retries it
    without the cap. Uses the message text: the exception is raised inside vLLM's serving layer
    and only its rendered form survives the engine's error handler.
    """
    if not message:
        return False
    return "maximum context length" in message and "output tokens" in message
