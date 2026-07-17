import json
from typing import Dict, Any


def pop_openai_kwargs(engine_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize & remove OpenAI-serving-only kwargs from engine_kwargs.
    """
    openai_kwargs: Dict[str, Any] = {}

    enable_auto_tools = engine_kwargs.pop("enable_auto_tools", engine_kwargs.pop("enable_auto_tool_choice", None))
    if enable_auto_tools is not None:
        openai_kwargs["enable_auto_tools"] = bool(enable_auto_tools)

    tool_parser = engine_kwargs.pop("tool_parser", engine_kwargs.pop("tool_call_parser", None))
    if tool_parser is not None:
        openai_kwargs["tool_parser"] = tool_parser

    # Sampling params for OpenAI-style requests (Harbor terminal-bench rollouts)
    openai_sampling = engine_kwargs.pop("openai_sampling_params", None)
    if openai_sampling is not None:
        openai_kwargs["openai_sampling_params"] = openai_sampling

    return openai_kwargs


def ensure_token_ids_in_sse_chunk(sse_chunk: str) -> str:
    """Remap ``new_token_ids`` → ``token_ids`` in ``provider_specific_fields``.

    vLLM streaming chunks carry ``delta.provider_specific_fields.new_token_ids``;
    harbor's ``accumulate_stream_chunk`` reads ``provider_specific_fields.token_ids``
    (or ``delta.token_ids``). Without this remap the literal record is empty and RL
    training silently corrupts.
    """
    if not sse_chunk.startswith("data: "):
        return sse_chunk
    payload = sse_chunk[len("data: ") :].strip()
    if payload == "[DONE]":
        return sse_chunk
    try:
        data = json.loads(payload)
        choices = data.get("choices", [])
        if not choices:
            return sse_chunk
        delta = choices[0].get("delta", {})
        psf = delta.get("provider_specific_fields")
        if psf and "new_token_ids" in psf and "token_ids" not in psf:
            psf["token_ids"] = psf["new_token_ids"]
            return f"data: {json.dumps(data)}\n\n"
    except (json.JSONDecodeError, IndexError, KeyError):
        pass
    return sse_chunk
