"""Model transports used by trajectory runners."""

import asyncio
from dataclasses import dataclass
from typing import Any, Protocol

import aiohttp
from transformers import PreTrainedTokenizerBase

from skyrl_train.inference_engines.base import ChatContinuation, InferenceEngineInput, InferenceEngineOutput
from skyrl_train.inference_engines.chat_continuation import EXACT_PROMPT_TOKEN_IDS_KEY, render_exact_chat_continuation
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.response_topk import select_chat_response_topk
from skyrl_train.trajectory_runners.types import TokenProvenance


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


def _choice_routed_experts(choice: dict[str, Any], response_ids: list[int]) -> list[list[list[int]]] | None:
    provider_fields = choice.get("provider_specific_fields") or {}
    routes = choice.get("routed_experts", provider_fields.get("routed_experts"))
    if routes is None:
        return None
    if not isinstance(routes, list) or len(routes) != len(response_ids):
        raise ValueError("chat response routed_experts must align with exact response token IDs")
    if not all(
        isinstance(token, list)
        and all(isinstance(layer, list) and all(isinstance(expert, int) for expert in layer) for layer in token)
        for token in routes
    ):
        raise ValueError("chat response routed_experts must have [token, layer, expert] integer shape")
    return routes


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
            if continuation is None:
                tokenize_response = await self._client.tokenize(render_request)
                prompt_ids = tokenize_response.get("tokens")
            else:
                prompt_ids = await render_exact_chat_continuation(
                    self._client.tokenize,
                    render_request,
                    assistant_message_index=continuation["assistant_message_index"],
                    served_prefix_token_ids=continuation["served_prefix_token_ids"],
                )
                if prompt_ids is None:
                    raise RuntimeError("Cannot preserve the exact served token prefix across this chat turn")
            if not isinstance(prompt_ids, list) or not all(isinstance(token, int) for token in prompt_ids):
                raise RuntimeError("vLLM tokenization did not return prompt token IDs")

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
                _choice_routed_experts(choice, response_ids),
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

    def __init__(self, *, base_url: str, model_name: str, tokenizer: PreTrainedTokenizerBase):
        self._base_url = base_url.rstrip("/")
        self._model_name = model_name
        self._tokenizer = tokenizer

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
                        self._generate_structured_chat(
                            session,
                            messages=messages,
                            session_id=session_id,
                            sampling_params=request.get("sampling_params") or {},
                            chat_options=options,
                            continuation=continuation,
                        )
                        for messages, session_id, options, continuation in zip(
                            prompts, session_ids, chat_options, continuations, strict=True
                        )
                    )
                )
                return _assemble_chat_results(results)
            responses = await asyncio.gather(
                *(
                    self._generate_one(
                        session,
                        messages=messages,
                        session_id=session_id,
                        sampling_params=request.get("sampling_params") or {},
                    )
                    for messages, session_id in zip(prompts, session_ids)
                )
            )

        texts = [response[0] for response in responses]
        return ModelClientOutput(
            responses=texts,
            response_ids=[self._tokenizer.encode(text, add_special_tokens=False) for text in texts],
            stop_reasons=[response[1] for response in responses],
            response_logprobs=None,
            prompt_logprobs=None,
            token_provenance=TokenProvenance.RECONSTRUCTED,
        )

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

        if continuation is None:
            prompt_ids = (await tokenize(render_request)).get("tokens")
        else:
            prompt_ids = await render_exact_chat_continuation(
                tokenize,
                render_request,
                assistant_message_index=continuation["assistant_message_index"],
                served_prefix_token_ids=continuation["served_prefix_token_ids"],
            )
            if prompt_ids is None:
                raise RuntimeError("Cannot preserve the exact served token prefix across this chat turn")
        if not isinstance(prompt_ids, list) or not all(isinstance(token, int) for token in prompt_ids):
            raise RuntimeError("OpenAI chat tokenization did not return prompt token IDs")

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
        choice = body["choices"][0]
        response_ids = choice.get("token_ids")
        if not isinstance(response_ids, list) or not all(isinstance(token, int) for token in response_ids):
            raise RuntimeError("OpenAI chat completion did not return exact response token IDs")
        logprob_items = (choice.get("logprobs") or {}).get("content")
        response_logprobs = [float(item["logprob"]) for item in logprob_items] if logprob_items is not None else None
        if response_logprobs is not None and len(response_logprobs) != len(response_ids):
            raise RuntimeError("OpenAI chat completion logprobs do not align with exact response token IDs")
        selected = None
        if requested_top_k is not None and logprob_items is not None:
            selected = [
                select_chat_response_topk(item.get("top_logprobs") or [], requested_top_k) for item in logprob_items
            ]
        return _ChatResult(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_logprobs=response_logprobs,
            student_topk_indices=None if selected is None else [ids for ids, _ in selected],
            behavior_topk_logprobs=None if selected is None else [scores for _, scores in selected],
            text=self._tokenizer.decode(response_ids, skip_special_tokens=True),
            stop_reason=choice["finish_reason"],
            assistant_message=choice["message"],
            routed_experts=_choice_routed_experts(choice, response_ids),
        )

    async def _generate_one(
        self,
        session: aiohttp.ClientSession,
        *,
        messages: list[dict[str, str]],
        session_id,
        sampling_params: dict,
    ) -> tuple[str, str]:
        request_sampling_params = dict(sampling_params)
        if "max_generate_length" in request_sampling_params:
            request_sampling_params["max_completion_tokens"] = request_sampling_params.pop("max_generate_length")
        payload = {
            "model": self._model_name,
            "messages": [{"role": message["role"], "content": message["content"]} for message in messages],
            "session_id": session_id,
            **request_sampling_params,
        }
        async with session.post(
            f"{self._base_url}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
        ) as response:
            body = await response.json()
            if response.status >= 400:
                raise RuntimeError(f"OpenAI chat completion returned HTTP {response.status}: {body}")
        choice = body["choices"][0]
        return choice["message"]["content"], choice["finish_reason"]
