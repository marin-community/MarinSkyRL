"""Exact token continuation for OpenCode requests at the terminal-bench bridge."""

import asyncio
import json
import logging
from collections import OrderedDict
from collections.abc import AsyncIterator
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from skyrl_train.inference_engines.inference_engine_client_http_endpoint import InferenceHTTPBackend


logger = logging.getLogger(__name__)

TRIAL_ID_HEADER = "x-ot-trial-id"
EXACT_PROMPT_TOKEN_IDS_KEY = "_skyrl_exact_prompt_token_ids"
_TOKENIZE_FIELDS = {
    "model",
    "messages",
    "add_generation_prompt",
    "continue_final_message",
    "add_special_tokens",
    "chat_template",
    "chat_template_kwargs",
    "media_io_kwargs",
    "mm_processor_kwargs",
    "tools",
}
_RENDER_SIGNATURE_FIELDS = _TOKENIZE_FIELDS - {"messages", "add_generation_prompt", "continue_final_message"}


@dataclass(frozen=True)
class _ContinuationState:
    messages: list[dict[str, Any]]
    render_signature: dict[str, Any]
    prompt_token_ids: list[int]
    completion_token_ids: list[int]


def _extract_chunk_token_ids(chunk: dict[str, Any]) -> tuple[list[int] | None, list[int]]:
    prompt_ids = chunk.get("prompt_token_ids")
    prompt = list(prompt_ids) if isinstance(prompt_ids, list) else None
    completion: list[int] = []
    for choice in chunk.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        for candidate in (choice.get("delta"), choice):
            if not isinstance(candidate, dict):
                continue
            provider_fields = candidate.get("provider_specific_fields")
            ids = provider_fields.get("token_ids") if isinstance(provider_fields, dict) else None
            if not isinstance(ids, list):
                ids = candidate.get("token_ids")
            if isinstance(ids, list):
                completion.extend(ids)
                break
    return prompt, completion


def _tokenize_body(body: dict[str, Any], messages: list[dict[str, Any]]) -> dict[str, Any]:
    request = {key: deepcopy(value) for key, value in body.items() if key in _TOKENIZE_FIELDS}
    request["messages"] = deepcopy(messages)
    return request


def _render_signature(body: dict[str, Any]) -> dict[str, Any]:
    return {key: deepcopy(value) for key, value in body.items() if key in _RENDER_SIGNATURE_FIELDS}


class OpenCodeContinuationLease:
    """One serialized OpenCode request whose stream will establish the next state."""

    def __init__(
        self,
        manager: "OpenCodeContinuationManager",
        trial_id: str,
        request_body: dict[str, Any],
        lock: asyncio.Lock,
    ) -> None:
        self._manager = manager
        self._trial_id = trial_id
        self._request_body = request_body
        self._lock = lock

    async def capture(self, stream: AsyncIterator[str]) -> AsyncIterator[str]:
        prompt_ids: list[int] | None = None
        completion_ids: list[int] = []
        buffer = b""
        completed = False
        valid = True
        reached_output_limit = False
        try:
            async for chunk in stream:
                yield chunk
                buffer += chunk.encode()
                while b"\n" in buffer:
                    raw_line, buffer = buffer.split(b"\n", 1)
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line.removeprefix("data:").strip()
                    if payload == "[DONE]":
                        completed = True
                        continue
                    if not payload:
                        continue
                    try:
                        parsed = json.loads(payload)
                    except json.JSONDecodeError:
                        logger.warning("Ignoring malformed SSE JSON while tracking OpenCode continuation")
                        valid = False
                        continue
                    chunk_prompt, chunk_completion = _extract_chunk_token_ids(parsed)
                    if prompt_ids is None and chunk_prompt is not None:
                        prompt_ids = chunk_prompt
                    completion_ids.extend(chunk_completion)
                    reached_output_limit = reached_output_limit or any(
                        isinstance(choice, dict) and choice.get("finish_reason") == "length"
                        for choice in parsed.get("choices") or []
                    )
        finally:
            try:
                if completed and reached_output_limit:
                    logger.info("OpenCode task-agent response reached output limit: trial_id=%s", self._trial_id)
                if completed and valid and prompt_ids and completion_ids:
                    expected_prompt = self._request_body.get(EXACT_PROMPT_TOKEN_IDS_KEY)
                    if expected_prompt is not None and prompt_ids != expected_prompt:
                        logger.error("vLLM did not serve the exact OpenCode continuation prompt")
                        self._manager.discard(self._trial_id)
                    else:
                        self._manager.commit(
                            self._trial_id,
                            self._request_body,
                            prompt_token_ids=prompt_ids,
                            completion_token_ids=completion_ids,
                        )
            finally:
                self._lock.release()
                self._manager._forget_unused_lock(self._trial_id, self._lock)


class OpenCodeContinuationManager:
    """Maintain exact served-token prefixes independently for concurrent trials."""

    def __init__(self, backend: "InferenceHTTPBackend", *, max_trials: int = 4096) -> None:
        self._backend = backend
        self._max_trials = max_trials
        self._states: OrderedDict[str, _ContinuationState] = OrderedDict()
        self._locks: dict[str, asyncio.Lock] = {}

    async def begin(self, request_payload: dict[str, Any]) -> OpenCodeContinuationLease | None:
        """Acquire a trial lease, or return ``None`` for an unmarked request."""
        headers = request_payload.get("headers", {})
        trial_id = headers.get(TRIAL_ID_HEADER)
        body = request_payload.get("json", {})
        if (
            not isinstance(trial_id, str)
            or not trial_id
            or not isinstance(body.get("messages"), list)
            # OpenCode's title and compaction agents share the trial header but call
            # the model with no tools. They are auxiliary generations, not turns in
            # the task agent's causal action chain, and must not replace its state.
            or not body.get("tools")
        ):
            return None

        lock = self._locks.setdefault(trial_id, asyncio.Lock())
        await lock.acquire()
        try:
            body.setdefault("session_id", trial_id)
            state = self._states.get(trial_id)
            if state is not None:
                exact_prompt = await self._continue_prompt(state, request_payload)
                if exact_prompt is None:
                    self._states.pop(trial_id, None)
                else:
                    body[EXACT_PROMPT_TOKEN_IDS_KEY] = exact_prompt
            return OpenCodeContinuationLease(self, trial_id, deepcopy(body), lock)
        except BaseException:
            lock.release()
            raise

    async def _continue_prompt(
        self,
        state: _ContinuationState,
        request_payload: dict[str, Any],
    ) -> list[int] | None:
        body = request_payload["json"]
        messages = body["messages"]
        prior_count = len(state.messages)
        reset_reason = None
        if _render_signature(body) != state.render_signature:
            reset_reason = "render signature changed"
        elif len(messages) <= prior_count:
            reset_reason = "history did not grow"
        elif messages[:prior_count] != state.messages:
            reset_reason = "history prefix was rewritten"
        elif not isinstance(messages[prior_count], dict) or messages[prior_count].get("role") != "assistant":
            reset_reason = "next message was not the served assistant turn"
        if reset_reason is not None:
            logger.info(
                "OpenCode continuation reset: %s (prior_messages=%d, current_messages=%d)",
                reset_reason,
                prior_count,
                len(messages),
            )
            return None

        history_messages = messages[:prior_count]
        prefix_messages = messages[: prior_count + 1]
        full_request = _tokenize_body(body, messages)
        prefix_request = _tokenize_body(body, prefix_messages)
        prefix_request["add_generation_prompt"] = False
        prefix_request["continue_final_message"] = False

        open_assistant_request = _tokenize_body(body, history_messages)
        open_assistant_request["add_generation_prompt"] = True
        open_assistant_request["continue_final_message"] = False
        empty_assistant_request = _tokenize_body(
            body,
            [*history_messages, {"role": "assistant", "content": ""}],
        )
        empty_assistant_request["add_generation_prompt"] = False
        empty_assistant_request["continue_final_message"] = False

        headers = request_payload.get("headers", {})
        full = await self._backend.tokenize({"json": full_request, "headers": headers})
        prefix = await self._backend.tokenize({"json": prefix_request, "headers": headers})
        open_assistant = await self._backend.tokenize({"json": open_assistant_request, "headers": headers})
        empty_assistant = await self._backend.tokenize({"json": empty_assistant_request, "headers": headers})
        full_ids = full.get("tokens") if isinstance(full, dict) else None
        prefix_ids = prefix.get("tokens") if isinstance(prefix, dict) else None
        open_assistant_ids = open_assistant.get("tokens") if isinstance(open_assistant, dict) else None
        empty_assistant_ids = empty_assistant.get("tokens") if isinstance(empty_assistant, dict) else None
        if (
            not isinstance(full_ids, list)
            or not isinstance(prefix_ids, list)
            or not isinstance(open_assistant_ids, list)
            or not isinstance(empty_assistant_ids, list)
            or full_ids[: len(prefix_ids)] != prefix_ids
            or empty_assistant_ids[: len(open_assistant_ids)] != open_assistant_ids
        ):
            logger.warning("OpenCode continuation reset because the rendered assistant boundary is not prefix-stable")
            return None

        # vLLM excludes the EOS/stop token from streamed completion token IDs.
        # Recover only the renderer's structural assistant-closing boundary from
        # an empty message; never reuse its re-tokenized assistant content.
        assistant_boundary = empty_assistant_ids[len(open_assistant_ids) :]
        suffix = full_ids[len(prefix_ids) :]
        return [*state.prompt_token_ids, *state.completion_token_ids, *assistant_boundary, *suffix]

    def commit(
        self,
        trial_id: str,
        request_body: dict[str, Any],
        *,
        prompt_token_ids: list[int],
        completion_token_ids: list[int],
    ) -> None:
        self._states[trial_id] = _ContinuationState(
            messages=deepcopy(request_body["messages"]),
            render_signature=_render_signature(request_body),
            prompt_token_ids=list(prompt_token_ids),
            completion_token_ids=list(completion_token_ids),
        )
        self._states.move_to_end(trial_id)
        while len(self._states) > self._max_trials:
            expired_trial, _ = self._states.popitem(last=False)
            expired_lock = self._locks.get(expired_trial)
            if expired_lock is not None and not expired_lock.locked():
                self._locks.pop(expired_trial, None)

    def discard(self, trial_id: str) -> None:
        """Forget a trial whose backend response disproved its requested prefix."""
        self._states.pop(trial_id, None)

    def _forget_unused_lock(self, trial_id: str, lock: asyncio.Lock) -> None:
        if trial_id not in self._states and self._locks.get(trial_id) is lock:
            self._locks.pop(trial_id, None)
