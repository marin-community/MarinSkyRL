"""Model transports used by trajectory runners."""

import asyncio
from typing import Protocol

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


class DirectModelClient:
    """Delegate model requests to the colocated inference-engine client."""

    def __init__(self, inference_engine_client: InferenceEngineClient):
        self._client = inference_engine_client

    async def generate(self, request: InferenceEngineInput) -> ModelClientOutput:
        output = await self._client.generate(request)
        return ModelClientOutput(**output, token_provenance=TokenProvenance.ENGINE)


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
