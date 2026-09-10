"""
OpenAI-compatible HTTP endpoint using InferenceEngineClient as backend.

This module provides a FastAPI-based HTTP endpoint that exposes OpenAI's chat completion API
while routing requests to our internal InferenceEngineClient system.

Main functions:
- serve(): Start the HTTP endpoint.
- wait_for_server_ready(): Wait for server to be ready.
- shutdown_server(): Shutdown the server.
"""

import asyncio
import json
import logging
import time
import requests
import traceback
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import Any, Coroutine, Dict, List, Optional, Protocol, Sequence, TypeVar

import fastapi
import uvicorn
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from skyrl_train.inference_engines.vllm.stats import HTTPBridgeStatsAccumulator


logger = logging.getLogger(__name__)

_ResponseT = TypeVar("_ResponseT")


class CompletionBackend(Protocol):
    """What this endpoint needs of the engine client it serves.

    InferenceEngineClient satisfies it. It is a protocol rather than that concrete type
    because the dependency runs the other way: inference_engine_client imports this module
    for the error types and the server entrypoints below.
    """

    model_name: str

    async def chat_completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]: ...

    async def chat_completion_stream(self, request_payload: Dict[str, Any]): ...

    async def completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]: ...


# Global state to hold the inference engine client and backend
_global_inference_engine_client: Optional[CompletionBackend] = None
_global_uvicorn_server: Optional[uvicorn.Server] = None


# Adapted from vllm.entrypoints.openai.protocol.ErrorResponse
class ErrorInfo(BaseModel):
    message: str
    type: str
    param: Optional[str] = None
    code: int


class ErrorResponse(BaseModel):
    error: ErrorInfo


def _error_response(message: str, status: HTTPStatus) -> ErrorResponse:
    """One error body for every route: the message, and the status in both its forms."""
    return ErrorResponse(error=ErrorInfo(message=message, type=status.phrase, code=status.value))


def _model_name_error(request_json: Dict[str, Any], endpoint: str) -> Optional[ErrorResponse]:
    """Reject a request that does not name the model this endpoint serves.

    One policy for every route. ``served_model_name`` (supported in
    ``generator.engine_init_kwargs``, and used by both vllm_engine.py and
    InferenceEngineClient for Harbor/LiteLLM compatibility) is the name callers address
    this endpoint with, so a request naming anything else is talking to the wrong server
    whichever route it arrived on.
    See https://github.com/NovaSky-AI/SkyRL/pull/238#discussion_r2326561295
    """
    if "model" not in request_json:
        return _error_response(f"The field `model` is required in your `{endpoint}` request.", HTTPStatus.BAD_REQUEST)
    served = getattr(_global_inference_engine_client, "model_name", None)
    if served != request_json["model"]:
        return _error_response(
            f"Model name mismatch: loaded model name {served} != model name in request {request_json['model']}",
            HTTPStatus.BAD_REQUEST,
        )
    return None


def set_global_state(inference_engine_client: CompletionBackend, uvicorn_server: uvicorn.Server):
    """Set the global inference engine client."""
    global _global_inference_engine_client
    global _global_uvicorn_server
    _global_inference_engine_client = inference_engine_client
    _global_uvicorn_server = uvicorn_server


def _validate_openai_request(request_json: Dict[str, Any], endpoint: str) -> Optional[ErrorResponse]:
    """Common validation for /chat/completions and /completions endpoints."""
    assert endpoint in ["/completions", "/chat/completions"]

    if _global_inference_engine_client is None:
        return _error_response("Inference engine client not initialized", HTTPStatus.INTERNAL_SERVER_ERROR)
    model_error = _model_name_error(request_json, endpoint)
    if model_error is not None:
        return model_error
    if endpoint == "/completions" and "n" in request_json and request_json["n"] > 1:
        # TODO(Charlie): this constraint can be removed when we leave DP routing to
        # inference frameworks. Or we could try to resolve it when needed.
        return _error_response(
            "n is not supported in SkyRL for /completions request yet, please set n to 1.", HTTPStatus.BAD_REQUEST
        )
    if endpoint == "/chat/completions" and "messages" not in request_json:
        return _error_response(
            "The field `messages` is required in your `/chat/completions` request.", HTTPStatus.BAD_REQUEST
        )
    if endpoint == "/chat/completions" and request_json["messages"] == []:
        return _error_response(
            "The field `messages` in `/chat/completions` cannot be an empty list.", HTTPStatus.BAD_REQUEST
        )
    return None


def _status_class(status_code: int) -> str:
    return f"{status_code // 100}xx"


def _json_response(
    content: Any,
    *,
    endpoint: str,
    bridge_stats: HTTPBridgeStatsAccumulator,
    status_code: int = HTTPStatus.OK.value,
) -> JSONResponse:
    started = time.perf_counter()
    response = JSONResponse(content=content, status_code=status_code)
    serialization_seconds = time.perf_counter() - started
    attributes = {"endpoint": endpoint, "transport": "json", "status": _status_class(status_code)}
    bridge_stats.observe("json_serialization_seconds", serialization_seconds, attributes=attributes)
    bridge_stats.observe("response_bytes", float(len(response.body)), attributes=attributes)
    return response


async def _safe_sse_stream(generator, *, endpoint: str, bridge_stats: HTTPBridgeStatsAccumulator):
    """Wrap an async generator so the SSE body always completes.

    Guarantees ``data: [DONE]\\n\\n`` is the last chunk even when the
    underlying generator raises mid-stream (Ray actor death, vLLM error,
    serialization failure, etc.).  Without this, ``StreamingResponse``
    closes the connection on any unhandled exception, producing
    ``RemoteProtocolError: incomplete chunked read`` on the client side.
    """
    done_sent = False
    response_bytes = 0
    status = HTTPStatus.OK.value
    try:
        async for chunk in generator:
            if isinstance(chunk, str) and "[DONE]" in chunk:
                done_sent = True
            response_bytes += len(chunk.encode()) if isinstance(chunk, str) else len(chunk)
            yield chunk
    except asyncio.CancelledError:
        # Client disconnected — re-raise so Starlette stops sending.
        raise
    except Exception as e:
        status = HTTPStatus.INTERNAL_SERVER_ERROR.value
        tb = traceback.format_exc()
        logger.error(f"Mid-stream SSE error:\n{tb}")
        if not done_sent:
            err = ErrorResponse(
                error=ErrorInfo(
                    message=f"Streaming error: {str(e)}",
                    type=HTTPStatus.INTERNAL_SERVER_ERROR.phrase,
                    code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                ),
            ).model_dump()
            error_chunk = f"data: {json.dumps(err)}\n\n"
            done_chunk = "data: [DONE]\n\n"
            response_bytes += len(error_chunk.encode()) + len(done_chunk.encode())
            yield error_chunk
            yield done_chunk
        return
    else:
        # Normal completion — ensure [DONE] if the generator didn't send it.
        if not done_sent:
            done_chunk = "data: [DONE]\n\n"
            response_bytes += len(done_chunk.encode())
            yield done_chunk
    finally:
        bridge_stats.observe(
            "response_bytes",
            float(response_bytes),
            attributes={"endpoint": endpoint, "transport": "sse", "status": _status_class(status)},
        )


async def _wait_for_disconnect(raw_request: Request) -> None:
    """Wait for the ASGI server to report that the HTTP client disconnected."""
    while True:
        message = await raw_request.receive()
        if message["type"] == "http.disconnect":
            return


async def _await_with_disconnect(raw_request: Request, backend_request: Coroutine[Any, Any, _ResponseT]) -> _ResponseT:
    """Cancel backend work when its HTTP client abandons the request."""
    backend_task = asyncio.create_task(backend_request)
    disconnect_task = asyncio.create_task(_wait_for_disconnect(raw_request))
    try:
        done, _ = await asyncio.wait(
            {backend_task, disconnect_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if backend_task in done:
            return await backend_task
        raise asyncio.CancelledError
    finally:
        backend_task.cancel()
        disconnect_task.cancel()
        await asyncio.gather(backend_task, disconnect_task, return_exceptions=True)


async def handle_openai_request(raw_request: Request, endpoint: str, bridge_stats: HTTPBridgeStatsAccumulator):
    """Handle /completions or /chat/completions request.

    Returns ``StreamingResponse`` for ``stream:true`` chat-completions and
    ``JSONResponse`` for everything else (byte-identical to pre-streaming behavior).
    """
    assert endpoint in ["/completions", "/chat/completions"]
    try:
        request_json = await raw_request.json()

        # SkyRL-side validation
        error_response = _validate_openai_request(request_json, endpoint=endpoint)
        if error_response is not None:
            return _json_response(
                error_response.model_dump(),
                endpoint=endpoint,
                bridge_stats=bridge_stats,
                status_code=error_response.error.code,
            )

        # Serialize fastapi.Request because it is not pickable, which causes ray methods to fail.
        payload = {
            "json": request_json,
            "headers": dict(raw_request.headers) if hasattr(raw_request, "headers") else {},
        }

        # ── Streaming branch ──────────────────────────────────────────────
        if request_json.get("stream", False) and endpoint == "/chat/completions":
            raw_gen = _global_inference_engine_client.chat_completion_stream(payload)
            return StreamingResponse(
                content=_safe_sse_stream(raw_gen, endpoint=endpoint, bridge_stats=bridge_stats),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        # Non-streaming requests stay attached to the client until completion.
        if endpoint == "/chat/completions":
            backend_request = _global_inference_engine_client.chat_completion(payload)
        else:
            backend_request = _global_inference_engine_client.completion(payload)
        response = await _await_with_disconnect(raw_request, backend_request)

        if "error" in response or response.get("object", "") == "error":
            # former is vllm format, latter is sglang format
            error_code = response["error"]["code"] if "error" in response else response["code"]
            return _json_response(response, endpoint=endpoint, bridge_stats=bridge_stats, status_code=error_code)
        else:
            return _json_response(response, endpoint=endpoint, bridge_stats=bridge_stats)

    except json.JSONDecodeError as e:
        # To catch possible raw_request.json() errors
        error_response = ErrorResponse(
            error=ErrorInfo(
                message=f"Invalid JSON error: {str(e)}",
                type=HTTPStatus.BAD_REQUEST.phrase,
                code=HTTPStatus.BAD_REQUEST.value,
            ),
        )
        return _json_response(
            error_response.model_dump(),
            endpoint=endpoint,
            bridge_stats=bridge_stats,
            status_code=HTTPStatus.BAD_REQUEST.value,
        )
    except Exception as e:
        # Include full traceback for debugging
        tb = traceback.format_exc()
        logger.error(f"Error when handling {endpoint} request in SkyRL:\n{tb}")
        error_response = ErrorResponse(
            error=ErrorInfo(
                message=f"Error when handling {endpoint} request in SkyRL: {str(e)}\n\nTraceback:\n{tb}",
                type=HTTPStatus.INTERNAL_SERVER_ERROR.phrase,
                code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            ),
        )
        return _json_response(
            error_response.model_dump(),
            endpoint=endpoint,
            bridge_stats=bridge_stats,
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
        )


# ─────────────────────────── POST /tokenize (vLLM-compatible) ────────────────────────────
#
# vLLM serves `/tokenize` at the SERVER ROOT (not under `/v1`), and harbor talks to it in
# two places:
#
#   * `harbor.llms.vllm_tokenization.tokenize_chat` — the TITO (token-in-token-out)
#     transport (harbor #111 / #117). It POSTs
#     ``{"model", "messages", "add_generation_prompt", **template_options}`` (where
#     template_options carries any of ``chat_template`` / ``chat_template_kwargs`` /
#     ``tools``) and reads ``tokens``. It has NO fallback: without this route every
#     rollout dies on ``HTTPStatusError: 404 Not Found for url .../tokenize`` (job
#     1738272 on Jupiter, 877 occurrences, every trial failed).
#   * `Terminus2._count_total_tokens` — the per-turn context-budget probe. It POSTs
#     ``{"model", "messages"}`` and reads ``count``; on a non-200 it falls back to
#     litellm's local tiktoken counter (27 % of the RolloutCoordinator's CPU at 45k
#     contexts), so serving this route also takes that cost off the coordinator.
#
# The ids handed back become the PROMPT the engines are then asked to continue, so they
# must equal what the serving path itself would render. The render therefore mirrors the
# OpenAI serving path exactly: the engine client's own tokenizer (loaded from
# ``trainer.policy.model.path``), the engine's custom chat template
# (``generator.engine_init_kwargs.custom_chat_template_chat_completion_path``) and its
# pinned ``chat_template_content_format`` — a template can render list-of-parts content
# differently from a plain string, which is why that knob exists.

# HF sets `model_max_length` to a sentinel (VERY_LARGE_INTEGER, 1e30) for tokenizers that
# declare no limit. Report `null` rather than that number.
_MAX_MODEL_LEN_SENTINEL = int(1e12)

_NO_TOKENIZER_MESSAGE = (
    "This SkyRL inference endpoint was started without a tokenizer, so `/tokenize` cannot "
    "render prompts. Serve it from an InferenceEngineClient, which carries the policy tokenizer."
)


def _part_text(part: Any) -> str:
    """Text of one OpenAI content part, as vLLM's ``string`` content format renders it."""
    if isinstance(part, str):
        return part
    if isinstance(part, dict) and part.get("type") in (None, "text"):
        return part.get("text") or ""
    return ""


def _messages_in_content_format(messages: Sequence[Any], content_format: str) -> List[Any]:
    """Coerce message content the way vLLM's renderer does for ``content_format``.

    ``string`` flattens list-of-parts content to its concatenated text; ``openai`` wraps a
    plain string as a single text part. Anything else is passed through untouched.
    """
    if content_format not in ("string", "openai"):
        return list(messages)

    converted: List[Any] = []
    for message in messages:
        if not isinstance(message, dict):
            converted.append(message)
            continue
        content = message.get("content")
        if content_format == "string" and isinstance(content, list):
            message = {**message, "content": "".join(_part_text(part) for part in content)}
        elif content_format == "openai" and isinstance(content, str):
            message = {**message, "content": [{"type": "text", "text": content}]}
        converted.append(message)
    return converted


def _resolve_content_format(configured: Optional[str], chat_template: Optional[str], tools: Any, tokenizer: Any) -> str:
    """Resolve the engine's configured content format to a concrete one.

    ``string`` / ``openai`` are already concrete. ``auto`` (vLLM's default) is resolved by
    vLLM's own template sniffing when vLLM is importable in this process — it is, in a real
    training run — and otherwise falls back to ``string``, which is what a plain-string
    conversation (everything harbor sends) renders as either way.
    """
    if configured in ("string", "openai"):
        return configured
    try:
        from vllm.entrypoints.chat_utils import resolve_chat_template_content_format

        return resolve_chat_template_content_format(chat_template, tools, "auto", tokenizer)
    except Exception:
        return "string"


def _resolve_max_model_len(tokenizer: Any) -> Optional[int]:
    """The context length to report, or ``None`` when nothing trustworthy is available."""
    for source in (
        getattr(_global_inference_engine_client, "max_model_len", None),
        getattr(tokenizer, "model_max_length", None),
    ):
        if isinstance(source, int) and 0 < source < _MAX_MODEL_LEN_SENTINEL:
            return source
    return None


def _render_tokenize_ids(request_json: Dict[str, Any], tokenizer: Any) -> List[int]:
    """Render one `/tokenize` request body to token ids (blocking; call off the event loop)."""
    if "prompt" in request_json:
        # Completion form: vLLM defaults add_special_tokens to True here.
        return list(
            tokenizer.encode(
                request_json["prompt"], add_special_tokens=bool(request_json.get("add_special_tokens", True))
            )
        )

    # Chat form. Request-level chat_template wins over the engine's, as in vLLM
    # (`request.chat_template or self.chat_template`).
    chat_template = request_json.get("chat_template") or getattr(
        _global_inference_engine_client, "custom_chat_template", None
    )
    tools = request_json.get("tools")
    content_format = _resolve_content_format(
        getattr(_global_inference_engine_client, "chat_template_content_format", None),
        chat_template,
        tools,
        tokenizer,
    )
    messages = _messages_in_content_format(request_json["messages"], content_format)

    rendered = tokenizer.apply_chat_template(
        messages,
        chat_template=chat_template,
        tools=tools,
        add_generation_prompt=bool(request_json.get("add_generation_prompt", True)),
        continue_final_message=bool(request_json.get("continue_final_message", False)),
        tokenize=False,
        **(request_json.get("chat_template_kwargs") or {}),
    )
    # vLLM renders to text and then encodes, with add_special_tokens defaulting to False for
    # the chat form (the template already emits them). Encoding the rendered string — rather
    # than `tokenize=True` — is what keeps that flag meaningful.
    return list(tokenizer.encode(rendered, add_special_tokens=bool(request_json.get("add_special_tokens", False))))


def _validate_tokenize_request(request_json: Dict[str, Any], tokenizer: Any) -> Optional[ErrorResponse]:
    """Reject bodies vLLM's TokenizeRequest union would reject, before any rendering."""
    if _global_inference_engine_client is None:
        return _error_response("Inference engine client not initialized", HTTPStatus.INTERNAL_SERVER_ERROR)
    if tokenizer is None:
        return _error_response(_NO_TOKENIZER_MESSAGE, HTTPStatus.NOT_IMPLEMENTED)
    if not isinstance(request_json, dict):
        return _error_response("The body of a `/tokenize` request must be a JSON object.", HTTPStatus.BAD_REQUEST)

    model_error = _model_name_error(request_json, "/tokenize")
    if model_error is not None:
        return model_error

    has_prompt = "prompt" in request_json
    has_messages = "messages" in request_json
    if has_prompt == has_messages:
        return _error_response(
            "Exactly one of `prompt` or `messages` is required in your `/tokenize` request.",
            HTTPStatus.BAD_REQUEST,
        )
    if has_prompt and not isinstance(request_json["prompt"], str):
        return _error_response("The field `prompt` must be a string.", HTTPStatus.BAD_REQUEST)
    if has_messages:
        messages = request_json["messages"]
        if not isinstance(messages, list):
            return _error_response("The field `messages` must be a list.", HTTPStatus.BAD_REQUEST)
        if not messages:
            # HF's apply_chat_template indexes conversation[0] before its own empty guard.
            return _error_response(
                "The field `messages` in `/tokenize` cannot be an empty list.", HTTPStatus.BAD_REQUEST
            )
        if request_json.get("add_generation_prompt") and request_json.get("continue_final_message"):
            return _error_response(
                "Cannot set both `add_generation_prompt` and `continue_final_message`.", HTTPStatus.BAD_REQUEST
            )
    return None


async def handle_tokenize_request(
    raw_request: Request,
    *,
    bridge_stats: HTTPBridgeStatsAccumulator,
) -> JSONResponse:
    """Serve vLLM's ``POST /tokenize`` contract against this endpoint's tokenizer.

    Accepts either the completion form (``prompt``) or the chat form (``messages``) and
    answers ``{"count", "max_model_len", "tokens", "token_strs"}``.
    """
    endpoint = "/tokenize"
    tokenizer = getattr(_global_inference_engine_client, "tokenizer", None)
    try:
        request_json = await raw_request.json()
    except Exception as e:
        error_response = _error_response(f"Invalid JSON error: {str(e)}", HTTPStatus.BAD_REQUEST)
        return _json_response(
            error_response.model_dump(),
            endpoint=endpoint,
            bridge_stats=bridge_stats,
            status_code=error_response.error.code,
        )

    error_response = _validate_tokenize_request(request_json, tokenizer)
    if error_response is not None:
        return _json_response(
            error_response.model_dump(),
            endpoint=endpoint,
            bridge_stats=bridge_stats,
            status_code=error_response.error.code,
        )

    try:
        # Tokenization is CPU-bound and this loop also serves every rollout's completions,
        # so it renders on a worker thread (HF's fast tokenizers drop the GIL).
        token_ids = await asyncio.to_thread(_render_tokenize_ids, request_json, tokenizer)
    except Exception as e:
        # A template/encoding failure is deterministic: the same body can never succeed, so
        # answer 400 (non-retryable for harbor/litellm) rather than 500.
        logger.warning(f"Error when handling /tokenize request in SkyRL: {traceback.format_exc()}")
        error_response = _error_response(
            f"Error when handling /tokenize request in SkyRL: {str(e)}", HTTPStatus.BAD_REQUEST
        )
        return _json_response(
            error_response.model_dump(),
            endpoint=endpoint,
            bridge_stats=bridge_stats,
            status_code=error_response.error.code,
        )

    token_strs = None
    if request_json.get("return_token_strs"):
        token_strs = tokenizer.convert_ids_to_tokens(token_ids)

    return _json_response(
        {
            "count": len(token_ids),
            "max_model_len": _resolve_max_model_len(tokenizer),
            "tokens": token_ids,
            "token_strs": token_strs,
        },
        endpoint=endpoint,
        bridge_stats=bridge_stats,
    )


def shutdown_server(host: str = "127.0.0.1", port: int = 8000, max_wait_seconds: int = 30) -> None:
    """Shutdown the server.

    Args:
        host: Server host.
        port: Server port.
        max_wait_seconds: How long to wait until the server stops listening.

    Raises:
        Exception: If the server is still responding after *max_wait_seconds*.
    """
    if _global_uvicorn_server is not None:
        _global_uvicorn_server.should_exit = True

    health_url = f"http://{host}:{port}/health"

    for i in range(max_wait_seconds):
        try:
            # If this succeeds, server is still alive
            requests.get(health_url, timeout=1)
        except requests.exceptions.RequestException:
            # A network error / connection refused means server is down.
            logger.info(f"Server shut down after {i + 1} seconds")
            return
        time.sleep(1)

    raise Exception(f"Server failed to shut down within {max_wait_seconds} seconds")


def wait_for_server_ready(host: str = "127.0.0.1", port: int = 8000, max_wait_seconds: int = 30) -> None:
    """
    Wait for the HTTP endpoint to be ready by polling the health endpoint.

    Args:
        host: Host where the server is running
        port: Port where the server is running
        max_wait_seconds: Maximum time to wait in seconds

    Raises:
        Exception: If server doesn't become ready within max_wait_seconds
    """
    max_retries = max_wait_seconds
    health_url = f"http://{host}:{port}/health"

    for i in range(max_retries):
        try:
            response = requests.get(health_url, timeout=1)
            if response.status_code == 200:
                logger.info(f"Server ready after {i + 1} attempts ({i + 1} seconds)")
                return
        except (requests.exceptions.RequestException, requests.exceptions.ConnectionError):
            if i == max_retries - 1:
                raise Exception(f"Server failed to start within {max_wait_seconds} seconds")
            time.sleep(1)  # Wait 1 second between retries


async def _monitor_event_loop_lag(
    bridge_stats: HTTPBridgeStatsAccumulator,
    interval_seconds: float,
) -> None:
    """Measure delay between a scheduled wake-up and the event loop running it."""
    try:
        loop = asyncio.get_running_loop()
        expected = loop.time() + interval_seconds
        while True:
            await asyncio.sleep(interval_seconds)
            now = loop.time()
            bridge_stats.observe("event_loop_lag_seconds", max(0.0, now - expected))
            expected = now + interval_seconds
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("Inference HTTP event-loop lag monitor failed")
        raise


def create_app(
    bridge_stats: HTTPBridgeStatsAccumulator | None = None,
    *,
    event_loop_lag_interval_seconds: float = 0.5,
) -> fastapi.FastAPI:
    """Create the FastAPI application.

    `/tokenize` renders with the backend's own tokenizer (``InferenceEngineClient.
    tokenizer``) and answers 501 if the backend has none.
    """
    bridge_stats = bridge_stats or HTTPBridgeStatsAccumulator()

    @asynccontextmanager
    async def lifespan(app: fastapi.FastAPI):
        logger.info("Starting inference HTTP endpoint...")
        monitor = asyncio.create_task(
            _monitor_event_loop_lag(bridge_stats, event_loop_lag_interval_seconds),
            name="inference-http-event-loop-lag",
        )
        try:
            yield
        finally:
            monitor.cancel()
            try:
                await monitor
            except asyncio.CancelledError:
                pass

    app = fastapi.FastAPI(
        title="InferenceEngine OpenAI-Compatible API",
        description="OpenAI-compatible chat completion API using InferenceEngineClient",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/v1/chat/completions")
    async def chat_completion(raw_request: Request):
        """
        Takes in OpenAI's `ChatCompletionRequest` and returns OpenAI's `ChatCompletionResponse`.

        Note that the specific fields inside the request and response depend on the backend you use.
        If `config.generator.backend` is `vllm`, then the request and response will be vLLM's.
        Same for SGLang. SkyRL does not perform field validation beyond `model` and `session_id`,
        and otherwise depends on the underlying engines' validation.

        Make sure you add in `session_id` (a string or an integer) to ensure load balancing and
        sticky routing. The same agentic rollout / session should share the same `session_id` so
        they get routed to the same engine for better prefix caching. If unprovided, we will route
        to a random engine which is not performant.

        API reference:
        - https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
        - https://docs.sglang.ai/basic_usage/openai_api_completions.html
        """
        return await handle_openai_request(raw_request, endpoint="/chat/completions", bridge_stats=bridge_stats)

    @app.post("/v1/completions")
    async def completions(raw_request: Request):
        """
        Takes in OpenAI's `CompletionRequest` and returns OpenAI's `CompletionResponse`.

        Note that the specific fields inside the request and response depend on the backend you use.
        If `config.generator.backend` is `vllm`, then the request and response will be vLLM's.
        SkyRL only validates the fields `model` and `session_id`, and otherwise offloads
        field validation to the underlying engines.

        Make sure you add in `session_id` to ensure load balancing and sticky routing. Since
        `request["prompt"]` can be `Union[list[int], list[list[int]], str, list[str]]`, i.e.
        {batched, single} x {string, token IDs}, we follow the following logic for request routing:
        - For batched request: `session_id`, if provided, must have the same length as `request["prompt"]`
          so that each `request["prompt"][i]` is routed based on `session_id[i]`.
        - For single request: `session_id`, if provided, must be a single integer or a singleton
          list, where each `session_id` is a string or an integer.

        API reference:
        - https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
        - https://docs.sglang.ai/basic_usage/openai_api_completions.html
        """
        return await handle_openai_request(raw_request, endpoint="/completions", bridge_stats=bridge_stats)

    @app.post("/tokenize")
    async def tokenize(raw_request: Request):
        """
        Takes in vLLM's `TokenizeRequest` and returns vLLM's `TokenizeResponse`.

        Served at the server root, not under `/v1`, exactly where vLLM serves it — that is
        where harbor's TITO transport and terminus-2's token counter look for it. The body
        is either `{"model", "prompt", "add_special_tokens"?}` or `{"model", "messages",
        "add_generation_prompt"?, "continue_final_message"?, "add_special_tokens"?,
        "chat_template"?, "chat_template_kwargs"?, "tools"?}`, and the response is
        `{"count", "max_model_len", "tokens", "token_strs"}`.

        API reference:
        - https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
        """
        return await handle_tokenize_request(raw_request, bridge_stats=bridge_stats)

    # Health check endpoint
    # All inference engine replicas are initialized before creating `InferenceEngineClient`, and thus
    # we can start receiving requests as soon as the FastAPI server starts
    @app.get("/health")
    async def health_check():
        return {"status": "healthy"}

    # This handler only catches unexpected server-side exceptions
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {str(exc)}\n{traceback.format_exc()}")
        error_response = ErrorResponse(
            error=ErrorInfo(
                message=f"Unhandled exception: {str(exc)}",
                type=HTTPStatus.INTERNAL_SERVER_ERROR.phrase,
                code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            ),
        )
        return _json_response(
            error_response.model_dump(),
            endpoint=request.url.path,
            bridge_stats=bridge_stats,
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
        )

    return app


def serve(
    inference_engine_client: CompletionBackend,
    host: str = "0.0.0.0",
    port: int = 8000,
    log_level: str = "info",
    bridge_stats: HTTPBridgeStatsAccumulator | None = None,
):
    """
    Start the HTTP endpoint.

    Args:
        inference_engine_client: The InferenceEngineClient to use as backend
        host: Host to bind to (default: "0.0.0.0")
        port: Port to bind to (default: 8000)
        log_level: Logging level (default: "info")
        bridge_stats: Shared accumulator for HTTP bridge metrics
    """
    # Create app
    app = create_app(bridge_stats)

    # Configure logging
    logging.basicConfig(level=getattr(logging, log_level.upper()))

    logger.info(f"Starting server on {host}:{port}")

    # Run server
    config = uvicorn.Config(app, host=host, port=port, log_level=log_level, access_log=True)
    server = uvicorn.Server(config)

    # Expose server for external shutdown control (tests)
    set_global_state(inference_engine_client, server)

    try:
        # Run until shutdown
        asyncio.run(server.serve())
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise
