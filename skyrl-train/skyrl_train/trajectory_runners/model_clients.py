"""Model transports used by trajectory runners."""

import asyncio
from copy import deepcopy
from dataclasses import dataclass
from collections.abc import Awaitable, Callable
from functools import partial
from typing import Any, Protocol, TypeVar

import aiohttp
from transformers import PreTrainedTokenizerBase

from skyrl_train.inference_engines.base import ChatContinuation, InferenceEngineInput, InferenceEngineOutput
from skyrl_train.inference_engines.chat_continuation import EXACT_PROMPT_TOKEN_IDS_KEY, render_exact_chat_continuation
from skyrl_train.inference_engines.chat_template import (
    SINGLE_TOOL_CALL_TEMPLATE_ERROR,
    sequentialize_multi_tool_call_turns,
    template_error_from_exception,
)
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.response_topk import select_chat_response_topk
from skyrl_train.policy_version import (
    RESPONSE_POLICY_VERSION_SEGMENTS_KEY,
    PolicyVersionSegment,
    validate_policy_version_segments,
)
from skyrl_train.trajectory_runners.types import TokenProvenance
from skyrl_train.trajectory_runners.routed_experts import normalize_routed_experts


_CHAT_SAMPLING_EXCLUSIONS = frozenset({"max_generate_length", "logprobs", "stop"})
_T = TypeVar("_T")


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
    policy_version_segments: list[PolicyVersionSegment] | None = None


def _choice_routed_experts(
    choice: dict[str, Any], prompt_ids: list[int], response_ids: list[int]
) -> list[list[list[int]]] | None:
    provider_fields = choice.get("provider_specific_fields") or {}
    routes = choice.get("routed_experts", provider_fields.get("routed_experts"))
    if routes is None:
        return None
    return normalize_routed_experts(routes, prompt_ids, response_ids)


@dataclass(frozen=True)
class _ChatChoice:
    """One chat completion choice with the served prompt, the engine's own tokens, logprobs and version spans."""

    message: dict[str, Any]
    finish_reason: str
    prompt_ids: list[int]
    response_ids: list[int]
    response_logprobs: list[float] | None
    logprob_items: list[dict[str, Any]] | None
    policy_version_segments: list[PolicyVersionSegment] | None
    routed_experts: list[list[list[int]]] | None


def _is_token_id_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(token, int) for token in value)


def _parse_chat_choice(choice: dict[str, Any], *, prompt_ids: Any, logprobs_requested: bool) -> _ChatChoice:
    """Read one OpenAI chat completion choice, rejecting one without exact tokens or requested logprobs.

    ``prompt_ids`` is the prompt the engine served for this choice: vLLM's response-level
    ``prompt_token_ids`` under ``return_token_ids``, or the tokenize call that rendered it.
    """
    response_ids = choice.get("token_ids")
    if not _is_token_id_list(response_ids):
        raise RuntimeError("OpenAI chat completion did not return exact response token IDs")
    if not _is_token_id_list(prompt_ids):
        raise RuntimeError("OpenAI chat completion did not return the served prompt token IDs")
    logprob_items = (choice.get("logprobs") or {}).get("content")
    if logprobs_requested and logprob_items is None:
        raise RuntimeError("OpenAI chat completion did not return the requested logprobs")
    response_logprobs = [float(item["logprob"]) for item in logprob_items] if logprob_items is not None else None
    if response_logprobs is not None and len(response_logprobs) != len(response_ids):
        raise RuntimeError("OpenAI chat completion logprobs do not align with exact response token IDs")
    segments = choice.get(RESPONSE_POLICY_VERSION_SEGMENTS_KEY)
    if segments is not None:
        validate_policy_version_segments(segments, response_length=len(response_ids), require_known=False)
    return _ChatChoice(
        message=choice["message"],
        finish_reason=choice["finish_reason"],
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        response_logprobs=response_logprobs,
        logprob_items=logprob_items,
        policy_version_segments=segments,
        routed_experts=_choice_routed_experts(choice, prompt_ids, response_ids),
    )


def _assemble_plain_results(results: list[_ChatChoice]) -> ModelClientOutput:
    logprobs = [result.response_logprobs for result in results]
    segments = [result.policy_version_segments for result in results]
    output = ModelClientOutput(
        responses=[result.message["content"] for result in results],
        prompt_ids=[result.prompt_ids for result in results],
        response_ids=[result.response_ids for result in results],
        stop_reasons=[result.finish_reason for result in results],
        response_logprobs=logprobs if all(value is not None for value in logprobs) else None,
        prompt_logprobs=None,
        token_provenance=TokenProvenance.ENGINE,
    )
    if all(rows is not None for rows in segments):
        output[RESPONSE_POLICY_VERSION_SEGMENTS_KEY] = segments
    if any(result.routed_experts is not None for result in results):
        output["routed_experts"] = [result.routed_experts for result in results]
    return output


def _structured_chat_result(choice: _ChatChoice, *, requested_top_k: int | None, text: str) -> _ChatResult:
    selected = None
    if requested_top_k is not None and choice.logprob_items is not None:
        selected = [
            select_chat_response_topk(item.get("top_logprobs") or [], requested_top_k) for item in choice.logprob_items
        ]
    return _ChatResult(
        prompt_ids=choice.prompt_ids,
        response_ids=choice.response_ids,
        response_logprobs=choice.response_logprobs,
        student_topk_indices=None if selected is None else [ids for ids, _ in selected],
        behavior_topk_logprobs=None if selected is None else [scores for _, scores in selected],
        text=text,
        stop_reason=choice.finish_reason,
        assistant_message=choice.message,
        routed_experts=choice.routed_experts,
        policy_version_segments=choice.policy_version_segments,
    )


def _assemble_chat_results(results: list[_ChatResult]) -> ModelClientOutput:
    logprobs = [result.response_logprobs for result in results]
    selected_indices = [result.student_topk_indices for result in results]
    selected_scores = [result.behavior_topk_logprobs for result in results]
    segments = [result.policy_version_segments for result in results]
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
    if all(rows is not None for rows in segments):
        output[RESPONSE_POLICY_VERSION_SEGMENTS_KEY] = segments
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
            choice = _parse_chat_choice(
                response["choices"][0],
                prompt_ids=prompt_ids,
                logprobs_requested=sampling_params.get("logprobs") is not None,
            )
            return _structured_chat_result(
                choice,
                requested_top_k=requested_top_k,
                text=self._client.tokenizer.decode(choice.response_ids, skip_special_tokens=True),
            )

        results = await asyncio.gather(
            *(
                generate_one(messages, row_options, session_id, continuation)
                for messages, row_options, session_id, continuation in zip(prompts, options, session_ids, continuations)
            )
        )
        return _assemble_chat_results(results)


class OpenAIHTTPModelClient:
    """Call an OpenAI-compatible chat endpoint and normalize its response."""

    def __init__(
        self,
        *,
        base_url: str,
        model_name: str,
        tokenizer: PreTrainedTokenizerBase,
        max_concurrent_requests: int,
    ):
        self._base_url = base_url.rstrip("/")
        self._model_name = model_name
        self._tokenizer = tokenizer
        self._request_slots = asyncio.Semaphore(max_concurrent_requests)

    async def _with_request_slot(self, operation: Callable[[], Awaitable[_T]]) -> _T:
        async with self._request_slots:
            return await operation()

    async def generate(self, request: InferenceEngineInput) -> ModelClientOutput:
        prompts = request.get("prompts")
        if prompts is None:
            raise ValueError("OpenAIHTTPModelClient requires message prompts; token-only requests are unsupported")

        session_ids = request.get("session_ids") or [None] * len(prompts)
        if len(session_ids) != len(prompts):
            raise ValueError("session_ids and prompts must have the same batch size")
        chat_options = request.get("chat_completion_params")
        if chat_options is not None and len(chat_options) != len(prompts):
            raise ValueError("chat_completion_params and prompts must have the same batch size")
        continuations = request.get("chat_continuations") or [None] * len(prompts)
        if len(continuations) != len(prompts):
            raise ValueError("chat_continuations and prompts must have the same batch size")

        timeout = aiohttp.ClientTimeout(total=None)
        connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            if chat_options is not None:
                results = await asyncio.gather(
                    *(
                        self._with_request_slot(
                            partial(
                                self._generate_structured_chat,
                                session,
                                messages=messages,
                                session_id=session_id,
                                sampling_params=request.get("sampling_params") or {},
                                chat_options=options,
                                continuation=continuation,
                            )
                        )
                        for messages, session_id, options, continuation in zip(
                            prompts, session_ids, chat_options, continuations, strict=True
                        )
                    )
                )
                return _assemble_chat_results(results)
            responses = await asyncio.gather(
                *(
                    self._with_request_slot(
                        partial(
                            self._generate_one,
                            session,
                            messages=messages,
                            session_id=session_id,
                            sampling_params=request.get("sampling_params") or {},
                        )
                    )
                    for messages, session_id in zip(prompts, session_ids)
                )
            )

        return _assemble_plain_results(responses)

    async def _generate_structured_chat(
        self,
        session: aiohttp.ClientSession,
        *,
        messages: list[dict[str, Any]],
        session_id: Any,
        sampling_params: dict[str, Any],
        chat_options: dict[str, Any],
        continuation: ChatContinuation | None,
    ) -> _ChatResult:
        options = DirectModelClient._chat_options(chat_options, sampling_params)
        tokenize_options = {key: options[key] for key in ("tools", "tool_choice") if key in options}
        render_request = {
            "json": {
                "model": self._model_name,
                "messages": messages,
                **tokenize_options,
                "add_generation_prompt": True,
            },
            "headers": {},
        }

        async def tokenize(request: dict[str, Any]) -> dict[str, Any]:
            async with session.post(f"{self._base_url}/tokenize", json=request["json"]) as response:
                body = await response.json()
                if response.status >= 400:
                    raise RuntimeError(f"OpenAI chat tokenization returned HTTP {response.status}: {body}")
                return body

        messages, prompt_ids = await _render_chat_prompt(tokenize, render_request, continuation)

        payload = {
            "model": self._model_name,
            "messages": messages,
            "session_id": session_id,
            **{key: value for key, value in sampling_params.items() if key not in _CHAT_SAMPLING_EXCLUSIONS},
            **options,
            "return_token_ids": True,
        }
        if continuation is not None:
            payload[EXACT_PROMPT_TOKEN_IDS_KEY] = prompt_ids
        if sampling_params.get("stop") is not None:
            payload["stop"] = sampling_params["stop"]
        if sampling_params.get("logprobs") is not None:
            payload["logprobs"] = True
        requested_top_k = sampling_params.get("logprobs")
        requested_top_k = requested_top_k if isinstance(requested_top_k, int) and requested_top_k > 0 else None
        if requested_top_k is not None:
            payload["top_logprobs"] = requested_top_k + 1
            payload["return_tokens_as_token_ids"] = True
        async with session.post(f"{self._base_url}/v1/chat/completions", json=payload) as response:
            body = await response.json()
            if response.status >= 400:
                raise RuntimeError(f"OpenAI chat completion returned HTTP {response.status}: {body}")
        choice = _parse_chat_choice(
            body["choices"][0], prompt_ids=prompt_ids, logprobs_requested=sampling_params.get("logprobs") is not None
        )
        return _structured_chat_result(
            choice,
            requested_top_k=requested_top_k,
            text=self._tokenizer.decode(choice.response_ids, skip_special_tokens=True),
        )

    async def _generate_one(
        self,
        session: aiohttp.ClientSession,
        *,
        messages: list[dict[str, str]],
        session_id,
        sampling_params: dict,
    ) -> _ChatChoice:
        request_sampling_params = {
            key: value for key, value in sampling_params.items() if key not in _CHAT_SAMPLING_EXCLUSIONS
        }
        if "max_generate_length" in sampling_params:
            request_sampling_params["max_completion_tokens"] = sampling_params["max_generate_length"]
        if sampling_params.get("stop") is not None:
            request_sampling_params["stop"] = sampling_params["stop"]
        payload = {
            "model": self._model_name,
            "messages": [{"role": message["role"], "content": message["content"]} for message in messages],
            "session_id": session_id,
            **request_sampling_params,
            # The engine's own prompt and response tokens, so the trainer trains on what was served
            # and sampled.
            "return_token_ids": True,
        }
        if sampling_params.get("logprobs") is not None:
            payload["logprobs"] = True
        async with session.post(
            f"{self._base_url}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
        ) as response:
            body = await response.json()
            if response.status >= 400:
                raise RuntimeError(f"OpenAI chat completion returned HTTP {response.status}: {body}")
        return _parse_chat_choice(
            body["choices"][0],
            prompt_ids=body.get("prompt_token_ids"),
            logprobs_requested=sampling_params.get("logprobs") is not None,
        )
