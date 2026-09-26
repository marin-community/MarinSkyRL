"""Model transports used by trajectory runners."""

import asyncio
from copy import deepcopy
from dataclasses import dataclass
from collections.abc import Awaitable, Callable
from typing import Any, Protocol


from skyrl_train.inference_engines.base import ChatContinuation, InferenceEngineInput, InferenceEngineOutput
from skyrl_train.inference_engines.chat_continuation import EXACT_PROMPT_TOKEN_IDS_KEY, render_exact_chat_continuation
from skyrl_train.inference_engines.chat_template import (
    SINGLE_TOOL_CALL_TEMPLATE_ERROR,
    sequentialize_multi_tool_call_turns,
    template_error_from_exception,
)
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.response_topk import select_chat_response_topk
from skyrl_train.trajectory_runners.types import TokenProvenance
from skyrl_train.trajectory_runners.routed_experts import normalize_routed_experts


_CHAT_SAMPLING_EXCLUSIONS = frozenset({"max_generate_length", "logprobs", "stop"})


class ModelClientOutput(InferenceEngineOutput):
    """Normalized model output with explicit token provenance."""

    token_provenance: TokenProvenance


class ModelClient(Protocol):
    """Transport-neutral model request boundary for trajectory runners."""

    async def generate(self, request: InferenceEngineInput) -> ModelClientOutput: ...


@dataclass(frozen=True)
class _ChatResult:
    prompt_ids: list[int]
    response_ids: list[int]
    response_logprobs: list[float] | None
    student_topk_indices: list[list[int]] | None
    behavior_topk_logprobs: list[list[float]] | None
    text: str
    stop_reason: str
    assistant_message: dict[str, Any]
    routed_experts: list[list[list[int]]] | None = None


def _choice_routed_experts(
    choice: dict[str, Any], prompt_ids: list[int], response_ids: list[int]
) -> list[list[list[int]]] | None:
    provider_fields = choice.get("provider_specific_fields") or {}
    routes = choice.get("routed_experts", provider_fields.get("routed_experts"))
    if routes is None:
        return None
    return normalize_routed_experts(routes, prompt_ids, response_ids)


def _assemble_chat_results(results: list[_ChatResult]) -> ModelClientOutput:
    logprobs = [result.response_logprobs for result in results]
    selected_indices = [result.student_topk_indices for result in results]
    selected_scores = [result.behavior_topk_logprobs for result in results]
    output = ModelClientOutput(
        prompt_ids=[result.prompt_ids for result in results],
        response_ids=[result.response_ids for result in results],
        response_logprobs=logprobs if all(value is not None for value in logprobs) else None,
        responses=[result.text for result in results],
        stop_reasons=[result.stop_reason for result in results],
        prompt_logprobs=None,
        assistant_messages=[result.assistant_message for result in results],
        token_provenance=TokenProvenance.ENGINE,
    )
    if all(rows is not None for rows in selected_indices):
        output["student_topk_indices"] = selected_indices
        output["behavior_topk_logprobs"] = selected_scores
    if any(result.routed_experts is not None for result in results):
        output["routed_experts"] = [result.routed_experts for result in results]
    return output


async def _render_chat_prompt(
    tokenize: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]],
    render_request: dict[str, Any],
    continuation: ChatContinuation | None,
) -> tuple[list[dict[str, Any]], list[int]]:
    async def render(request: dict[str, Any], current_continuation: ChatContinuation | None) -> list[int] | None:
        if current_continuation is None:
            return (await tokenize(request)).get("tokens")
        return await render_exact_chat_continuation(
            tokenize,
            request,
            assistant_message_index=current_continuation["assistant_message_index"],
            served_prefix_token_ids=current_continuation["served_prefix_token_ids"],
        )

    messages = render_request["json"]["messages"]
    try:
        prompt_ids = await render(render_request, continuation)
    except Exception as error:
        template_error = template_error_from_exception(error)
        if template_error is None:
            raise
        if SINGLE_TOOL_CALL_TEMPLATE_ERROR not in str(template_error):
            raise template_error from error
        assistant_index = continuation["assistant_message_index"] if continuation is not None else None
        messages, assistant_index = sequentialize_multi_tool_call_turns(messages, assistant_index)
        if messages == render_request["json"]["messages"]:
            raise template_error from error
        render_request = deepcopy(render_request)
        render_request["json"]["messages"] = messages
        if continuation is not None:
            assert assistant_index is not None
            continuation = ChatContinuation(
                served_prefix_token_ids=continuation["served_prefix_token_ids"],
                assistant_message_index=assistant_index,
            )
        try:
            prompt_ids = await render(render_request, continuation)
        except Exception as retry_error:
            retry_template_error = template_error_from_exception(retry_error)
            if retry_template_error is not None:
                raise retry_template_error from retry_error
            raise
    if prompt_ids is None:
        raise RuntimeError("Cannot preserve the exact served token prefix across this chat turn")
    if not isinstance(prompt_ids, list) or not all(isinstance(token, int) for token in prompt_ids):
        raise RuntimeError("vLLM tokenization did not return prompt token IDs")
    return messages, prompt_ids


class DirectModelClient:
    """Delegate model requests to the colocated inference-engine client."""

    def __init__(self, inference_engine_client: InferenceEngineClient):
        self._client = inference_engine_client

    async def generate(self, request: InferenceEngineInput) -> ModelClientOutput:
        if request.get("chat_completion_params") is not None:
            return await self._generate_chat(request)
        output = await self._client.generate(request)
        return ModelClientOutput(**output, token_provenance=TokenProvenance.ENGINE)

    @staticmethod
    def _chat_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        converted = []
        for tool in tools:
            if tool.get("type") != "function":
                raise ValueError(f"Only function tools are supported, got {tool.get('type')!r}")
            function = {key: value for key, value in tool.items() if key != "type"}
            converted.append({"type": "function", "function": function})
        return converted

    @classmethod
    def _chat_options(cls, options: dict[str, Any], sampling_params: dict[str, Any]) -> dict[str, Any]:
        result = dict(options)
        result.pop("input", None)
        if result.get("tools"):
            result["tools"] = cls._chat_tools(result["tools"])
        else:
            result.pop("tools", None)
        if "max_output_tokens" in result:
            result["max_completion_tokens"] = result.pop("max_output_tokens")
        if "max_generate_length" in sampling_params:
            configured_max = int(sampling_params["max_generate_length"])
            requested_max = result.get("max_completion_tokens")
            result["max_completion_tokens"] = (
                configured_max if requested_max is None else min(configured_max, int(requested_max))
            )
        return result

    async def _generate_chat(self, request: InferenceEngineInput) -> ModelClientOutput:
        prompts = request.get("prompts")
        options = request.get("chat_completion_params")
        if prompts is None or options is None or len(prompts) != len(options):
            raise ValueError("Chat generation requires aligned prompts and chat_completion_params")
        session_ids = request.get("session_ids") or [None] * len(prompts)
        if len(session_ids) != len(prompts):
            raise ValueError("session_ids and prompts must have the same batch size")
        continuations = request.get("chat_continuations") or [None] * len(prompts)
        if len(continuations) != len(prompts):
            raise ValueError("chat_continuations and prompts must have the same batch size")
        sampling_params = dict(request.get("sampling_params") or {})
        requested_top_k = sampling_params.get("logprobs")
        requested_top_k = requested_top_k if isinstance(requested_top_k, int) and requested_top_k > 0 else None

        async def generate_one(messages, row_options, session_id, continuation):
            chat_options = self._chat_options(row_options, sampling_params)
            tokenize_options = {key: chat_options[key] for key in ("tools", "tool_choice") if key in chat_options}
            render_request = {
                "json": {
                    "model": self._client.model_name,
                    "messages": messages,
                    **tokenize_options,
                    "add_generation_prompt": True,
                },
                "headers": {},
            }
            messages, prompt_ids = await _render_chat_prompt(self._client.tokenize, render_request, continuation)

            body = {
                "model": self._client.model_name,
                "messages": messages,
                "session_id": session_id,
                **{key: value for key, value in sampling_params.items() if key not in _CHAT_SAMPLING_EXCLUSIONS},
                **chat_options,
                "return_token_ids": True,
            }
            if continuation is not None:
                body[EXACT_PROMPT_TOKEN_IDS_KEY] = prompt_ids
            if sampling_params.get("stop") is not None:
                body["stop"] = sampling_params["stop"]
            if sampling_params.get("logprobs") is not None:
                body["logprobs"] = True
            if requested_top_k is not None:
                # vLLM may include the sampled token outside the natural top K.
                body["top_logprobs"] = requested_top_k + 1
                body["return_tokens_as_token_ids"] = True
            response = await self._client.chat_completion({"json": body, "headers": {}})
            if "choices" not in response:
                raise RuntimeError(f"vLLM chat completion failed: {response}")
            choice = response["choices"][0]
            response_ids = choice.get("token_ids")
            if not isinstance(response_ids, list) or not all(isinstance(token, int) for token in response_ids):
                raise RuntimeError("vLLM chat completion did not return exact token IDs")
            message = choice["message"]
            text = self._client.tokenizer.decode(response_ids, skip_special_tokens=True)
            logprob_items = (choice.get("logprobs") or {}).get("content")
            response_logprobs = (
                [float(item["logprob"]) for item in logprob_items] if logprob_items is not None else None
            )
            selected = None
            if requested_top_k is not None and logprob_items is not None:
                if len(logprob_items) != len(response_ids):
                    raise ValueError("chat response top-K rows must align with exact token IDs")
                selected = [
                    select_chat_response_topk(item.get("top_logprobs") or [], requested_top_k) for item in logprob_items
                ]
            return _ChatResult(
                prompt_ids,
                response_ids,
                response_logprobs,
                None if selected is None else [ids for ids, _ in selected],
                None if selected is None else [scores for _, scores in selected],
                text,
                choice["finish_reason"],
                message,
                _choice_routed_experts(choice, prompt_ids, response_ids),
            )

        results = await asyncio.gather(
            *(
                generate_one(messages, row_options, session_id, continuation)
                for messages, row_options, session_id, continuation in zip(prompts, options, session_ids, continuations)
            )
        )
        return _assemble_chat_results(results)
