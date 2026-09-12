"""Model transports used by trajectory runners."""

import asyncio
from dataclasses import dataclass
from typing import Any, Protocol

import aiohttp
from transformers import PreTrainedTokenizerBase

from skyrl_train.inference_engines.base import InferenceEngineInput, InferenceEngineOutput
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.trajectory_runners.types import TokenProvenance


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
    text: str
    stop_reason: str
    assistant_message: dict[str, Any]


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
        sampling_params = dict(request.get("sampling_params") or {})

        async def generate_one(messages, row_options, session_id):
            chat_options = self._chat_options(row_options, sampling_params)
            tokenize_options = {key: chat_options[key] for key in ("tools", "tool_choice") if key in chat_options}
            tokenize_response = await self._client.tokenize(
                {
                    "json": {
                        "model": self._client.model_name,
                        "messages": messages,
                        **tokenize_options,
                        "add_generation_prompt": True,
                    },
                    "headers": {},
                }
            )
            prompt_ids = tokenize_response.get("tokens")
            if not isinstance(prompt_ids, list) or not all(isinstance(token, int) for token in prompt_ids):
                raise RuntimeError(f"vLLM tokenization failed: {tokenize_response}")

            body = {
                "model": self._client.model_name,
                "messages": messages,
                "session_id": session_id,
                **{
                    key: value
                    for key, value in sampling_params.items()
                    if key not in {"max_generate_length", "logprobs", "stop"}
                },
                **chat_options,
                "return_token_ids": True,
            }
            if sampling_params.get("stop") is not None:
                body["stop"] = sampling_params["stop"]
            if sampling_params.get("logprobs") is not None:
                body["logprobs"] = True
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
            return _ChatResult(prompt_ids, response_ids, response_logprobs, text, choice["finish_reason"], message)

        results = await asyncio.gather(
            *(
                generate_one(messages, row_options, session_id)
                for messages, row_options, session_id in zip(prompts, options, session_ids)
            )
        )
        logprobs = [result.response_logprobs for result in results]
        return ModelClientOutput(
            prompt_ids=[result.prompt_ids for result in results],
            response_ids=[result.response_ids for result in results],
            response_logprobs=logprobs if all(value is not None for value in logprobs) else None,
            responses=[result.text for result in results],
            stop_reasons=[result.stop_reason for result in results],
            prompt_logprobs=None,
            assistant_messages=[result.assistant_message for result in results],
            token_provenance=TokenProvenance.ENGINE,
        )


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

        timeout = aiohttp.ClientTimeout(total=None)
        connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
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

    async def _generate_one(
        self,
        session: aiohttp.ClientSession,
        *,
        messages: list[dict[str, str]],
        session_id,
        sampling_params: dict,
    ) -> tuple[str, str]:
        payload = {
            "model": self._model_name,
            "messages": [{"role": message["role"], "content": message["content"]} for message in messages],
            "session_id": session_id,
            **sampling_params,
        }
        async with session.post(
            f"{self._base_url}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
        ) as response:
            response.raise_for_status()
            body = await response.json()
        choice = body["choices"][0]
        return choice["message"]["content"], choice["finish_reason"]
