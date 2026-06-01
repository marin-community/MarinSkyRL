from skyrl_train.inference_engines.base import (
    InferenceEngineInterface,
    InferenceEngineInput,
    InferenceEngineOutput,
    NamedWeightsUpdateRequest,
)
from skyrl_train.inference_engines.inference_engine_client_http_endpoint import ErrorResponse, ErrorInfo
from transformers import PreTrainedTokenizerBase
import asyncio
from typing import List, Any, Optional, Dict, Union
from skyrl_train.inference_engines.utils import (
    route_prompts_to_engines,
    hash_with_sha256,
    postprocess_completion_request,
    aggregate_completion_usage_info,
)
from omegaconf import DictConfig
import threading
from loguru import logger
import random
import ray.exceptions
from dataclasses import dataclass, field

ABORT_GENERATION_GRACE_PERIOD_SECONDS = 5


class InferenceEngineClient(InferenceEngineInterface):
    """
    Client to talk to a set of InferenceEngines.

    Note that InferenceEngineClient sub-classes InferenceEngineInterface so it can be used as if talking to a single engine.
    """

    def __init__(
        self, engines: List[InferenceEngineInterface], tokenizer: PreTrainedTokenizerBase, full_config: DictConfig
    ):
        """
        Args:
            engines: List[InferenceEngineInterface] - The inference engines, remote or local.
            tokenizer: PreTrainedTokenizerBase - The tokenizer to use.
            full_config: DictConfig - See ppo_base_config.yaml
        """
        self.engines = engines
        self.tokenizer = tokenizer
        # Use served_model_name if configured (for Harbor/LiteLLM compatibility),
        # otherwise fall back to the full model path.
        # See https://github.com/NovaSky-AI/SkyRL/pull/238#discussion_r2326561295
        served_model_name = None
        if hasattr(full_config.generator, "engine_init_kwargs"):
            served_model_name = getattr(full_config.generator.engine_init_kwargs, "served_model_name", None)
        self.model_name = served_model_name if served_model_name else full_config.trainer.policy.model.path
        self.backend = full_config.generator.backend
        self.enable_http_endpoint = full_config.generator.enable_http_endpoint
        self.http_endpoint_host = full_config.generator.http_endpoint_host
        self.http_endpoint_port = full_config.generator.http_endpoint_port
        self.generation_paused_event = threading.Event()
        self._dead_engines: set[int] = set()
        if self.enable_http_endpoint:
            self._spin_up_http_endpoint()

        logger.info(f"InferenceEngineClient initialized with {len(engines)} engines.")

    def _mark_engine_dead(self, engine_idx: int, error: Exception) -> None:
        """Mark an engine as dead and log a warning."""
        if engine_idx not in self._dead_engines:
            self._dead_engines.add(engine_idx)
            remaining = len(self.engines) - len(self._dead_engines)
            logger.warning(
                f"Inference engine {engine_idx} died ({type(error).__name__}). "
                f"{remaining}/{len(self.engines)} engines remaining."
            )
            if remaining == 0:
                logger.error("All inference engines have died!")

    def _pick_fallback_engine(self, exclude_idx: int) -> int | None:
        """Pick a random live engine, excluding the given index."""
        live = [
            i for i in range(len(self.engines))
            if i not in self._dead_engines and i != exclude_idx
        ]
        return random.choice(live) if live else None

    def _resolve_engine_idx(self, engine_idx: int) -> int:
        """If the chosen engine is dead, pick a fallback. Raises if all are dead."""
        if engine_idx not in self._dead_engines:
            return engine_idx
        fallback = self._pick_fallback_engine(engine_idx)
        if fallback is None:
            raise RuntimeError("All inference engines have died")
        return fallback

    async def _run_on_all_engines(self, method_name: str, *args, **kwargs):
        """
        Call a method on all live engines concurrently and gather the results.
        """
        live_engines = [
            engine for i, engine in enumerate(self.engines)
            if i not in self._dead_engines
        ]
        if not live_engines:
            raise RuntimeError("All inference engines have died")

        awaitables = [getattr(engine, method_name)(*args, **kwargs) for engine in live_engines]
        return await asyncio.gather(*awaitables)

    async def generate(self, input_batch: InferenceEngineInput) -> InferenceEngineOutput:
        if self.generation_paused_event.is_set():
            raise RuntimeError("pause_generation is unsupported for InferenceEngineClient.generate().")
        # 0. Extract input
        prompts = input_batch.get("prompts")
        prompt_token_ids = input_batch.get("prompt_token_ids")
        session_ids = input_batch.get("session_ids")
        sampling_params = input_batch.get("sampling_params")

        if (prompts is None and prompt_token_ids is None) or (prompts is not None and prompt_token_ids is not None):
            raise ValueError("Either `prompts` or `prompt_token_ids` must be provided, but not both.")
        if prompt_token_ids is None:
            prompt_token_ids = self.tokenizer.apply_chat_template(
                prompts,
                add_generation_prompt=True,
                add_special_tokens=False,
                return_dict=True,
                tokenize=True,
            )["input_ids"]

        num_prompts = len(prompt_token_ids)
        num_inference_engines = len(self.engines)

        # 1. Route prompts to engines
        engine_idx_to_prompt_ids: dict[int, list[int]] = route_prompts_to_engines(
            num_prompts=num_prompts,
            num_inference_engines=num_inference_engines,
            session_ids=session_ids,
        )

        # We do a shortcut for non-batched requests, which can support pause/continue generation for
        # in-flight weight updates.
        if num_prompts == 1:
            # Route to a single engine for this single prompt and use retry flow.
            assert len(engine_idx_to_prompt_ids) == 1
            ((engine_idx, prompt_ids_list),) = engine_idx_to_prompt_ids.items()
            assert prompt_ids_list == [0], "Single prompt should map to index [0]"
            engine_idx = self._resolve_engine_idx(engine_idx)
            original_prompt_ids = prompt_token_ids[0]
            return await self._generate_single_with_retry(
                engine_idx=engine_idx,
                original_prompt_ids=original_prompt_ids,
                sampling_params=sampling_params,
            )

        # For batched generate(), pause/continue cannot be supported.
        if self.generation_paused_event.is_set():
            raise RuntimeError("pause_generation is unsupported for batched InferenceEngineClient.generate().")

        # 2. Generate responses concurrently (with failover for dead engines)
        tasks: list[asyncio.Task] = []
        indices_list: list[list[int]] = []
        task_engine_idxs: list[int] = []

        # Reroute prompts away from known-dead engines
        rerouted_mapping: dict[int, list[int]] = {}
        for engine_idx, prompt_ids in engine_idx_to_prompt_ids.items():
            resolved = self._resolve_engine_idx(engine_idx)
            rerouted_mapping.setdefault(resolved, []).extend(prompt_ids)

        for engine_idx, prompt_ids in rerouted_mapping.items():
            cur_prompt_token_ids = [prompt_token_ids[i] for i in prompt_ids]
            engine_input = InferenceEngineInput(
                prompt_token_ids=cur_prompt_token_ids,
                sampling_params=sampling_params,
            )
            tasks.append(asyncio.create_task(self.engines[engine_idx].generate(engine_input)))
            indices_list.append(prompt_ids)
            task_engine_idxs.append(engine_idx)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 2.1. Handle engine deaths: retry failed tasks on fallback engines
        for i, result in enumerate(results):
            if isinstance(result, (ray.exceptions.ActorDiedError, ray.exceptions.RayActorError)):
                self._mark_engine_dead(task_engine_idxs[i], result)
                fallback = self._pick_fallback_engine(task_engine_idxs[i])
                if fallback is None:
                    raise RuntimeError("All inference engines have died") from result
                cur_prompt_token_ids = [prompt_token_ids[j] for j in indices_list[i]]
                engine_input = InferenceEngineInput(
                    prompt_token_ids=cur_prompt_token_ids,
                    sampling_params=sampling_params,
                )
                results[i] = await self.engines[fallback].generate(engine_input)
            elif isinstance(result, BaseException):
                raise result

        # 3. Reconstruct output in original order
        n = len(prompt_token_ids)
        responses: list[str] = [""] * n
        stop_reasons: list[str] = [""] * n
        response_logprobs: List[Optional[List[float]]] = [None for _ in range(n)]
        response_ids: List[List[int]] = [[] for _ in range(n)]
        prompt_logprobs: List[Optional[Any]] = [None for _ in range(n)]
        # a bit hacky for now
        add_resp_logprobs = False
        add_prompt_logprobs = False

        for indices, result in zip(indices_list, results):
            for local_idx, original_idx in enumerate(indices):
                responses[original_idx] = result["responses"][local_idx]
                stop_reasons[original_idx] = result["stop_reasons"][local_idx]
                response_ids[original_idx] = result["response_ids"][local_idx]
                if result.get("response_logprobs", None):
                    add_resp_logprobs = True
                    response_logprobs[original_idx] = result["response_logprobs"][local_idx]
                if result.get("prompt_logprobs") is not None:
                    add_prompt_logprobs = True
                    prompt_logprobs[original_idx] = result["prompt_logprobs"][local_idx]

        return InferenceEngineOutput(
            responses=responses,
            stop_reasons=stop_reasons,
            response_ids=response_ids,
            response_logprobs=response_logprobs if add_resp_logprobs else None,
            prompt_logprobs=prompt_logprobs if add_prompt_logprobs else None,
        )

    async def _generate_single_with_retry(
        self, engine_idx: int, original_prompt_ids: List[int], sampling_params: Optional[Dict[str, Any]]
    ) -> InferenceEngineOutput:
        """
        Generate a single response with retry mechanism.

        This method is equivalent to `_chat_completion_with_retry()` but for the `generate()` codepath.
        We keep sending `generate` requests (with previous responses accumulated) until the finish_reason
        is not "abort". It is intended to be used in combination with `pause_generation()` and `resume_generation()` for
        in-flight weight updates and partial rollouts.

        This method is equivalent to a single `generate()` call if we do not use `pause_generation()`.

        Since we operate purely in the token space, it is token-in-token-out, unlike `_chat_completion_with_retry()`
        which re-encodes in each new request.

        For subsequent retry requests (`InferenceEngineInput`), we:
        - Update the `InferenceEngineInput.prompt_token_ids` with the accumulated output tokens.
        - Skip accumulating `InferenceEngineOutput.responses` since we decode the final output.
        - Adjust remaining max tokens if `max_tokens` or `max_completion_tokens` is present.

        For the final response, we return `InferenceEngineOutput` with:
        - `responses`: decoded at the end from `response_ids` if generation is completed in > 1 turns, otherwise the text response of the first turn.
        - `response_ids`: the accumulated output tokens
        - `stop_reasons`: the stop reason of the final response
        - `response_logprobs`: the accumulated logprobs
        """
        if sampling_params is None:
            sampling_params = {}

        # 1. First determine original max tokens key and value (if any)
        max_key = None
        if "max_tokens" in sampling_params:
            max_key = "max_tokens"
        elif "max_completion_tokens" in sampling_params:
            max_key = "max_completion_tokens"
        original_max_tokens: Optional[int] = sampling_params.get(max_key) if max_key else None

        # 2. Initialize fields we want to accumulate or update in each loop iteration
        accum_response_ids: List[int] = []
        accum_response_logprobs: List[float] = []
        stop_reason: str = "abort"

        # We only use it if generation is completed in one turn to maintain original behavior with no retry.
        text_response: Optional[str] = None
        num_turns = 0

        # 3. Loop until geneartion is completed.
        while stop_reason == "abort":
            await self._wait_for_generation_to_resume()

            # 3.1. Prepare the request payload.
            cur_sampling_params = sampling_params.copy()
            if original_max_tokens is not None:
                new_max_tokens = original_max_tokens - len(accum_response_ids)
                assert new_max_tokens >= 0, f"Expect new_max_tokens to be non-negative, but got {new_max_tokens}"
                cur_sampling_params[max_key] = new_max_tokens
            new_prompt_ids = original_prompt_ids + accum_response_ids
            engine_input = InferenceEngineInput(
                prompt_token_ids=[new_prompt_ids],
                sampling_params=cur_sampling_params,
            )

            # 3.2. Send the request.
            logger.debug(f"generate() request sent (including potential retries): {engine_input}")
            try:
                partial_response: InferenceEngineOutput = await self.engines[engine_idx].generate(engine_input)
            except (ray.exceptions.ActorDiedError, ray.exceptions.RayActorError) as e:
                self._mark_engine_dead(engine_idx, e)
                fallback = self._pick_fallback_engine(engine_idx)
                if fallback is None:
                    raise RuntimeError("All inference engines have died") from e
                engine_idx = fallback
                # Reset accumulation — new engine has no prior context
                accum_response_ids = []
                accum_response_logprobs = []
                num_turns = 0
                stop_reason = "abort"
                continue

            # 3.3. Parse the partial response.
            assert len(partial_response["response_ids"]) == 1, "Expected exactly one response."
            new_response_ids: List[int] = partial_response["response_ids"][0]
            text_response = partial_response["responses"][0]
            stop_reason = partial_response["stop_reasons"][0]
            new_response_logprobs: Optional[List[float]] = None
            new_response_logprobs_list: Optional[List[List[float]]] = partial_response.get("response_logprobs", None)
            if new_response_logprobs_list is not None and len(new_response_logprobs_list) > 0:
                new_response_logprobs = new_response_logprobs_list[0]

            # 3.4 Aborted without generating tokens, so partial_response is useless.
            if stop_reason == "abort" and len(new_response_ids) == 0:
                continue

            # 3.5 Accumulate outputs
            accum_response_ids.extend(new_response_ids)
            if new_response_logprobs is not None:
                accum_response_logprobs.extend(new_response_logprobs)
            num_turns += 1

        # 4. Build the final response and return.
        if num_turns == 1:
            final_text_response = text_response
        else:
            final_text_response = self.tokenizer.decode(accum_response_ids, skip_special_tokens=True)

        # Propagate prompt_logprobs from the last partial response (only meaningful
        # for teacher scoring where max_tokens=1 and num_turns=1).
        final_prompt_logprobs = partial_response.get("prompt_logprobs") if partial_response else None

        return InferenceEngineOutput(
            responses=[final_text_response],
            stop_reasons=[stop_reason],
            response_ids=[accum_response_ids],
            response_logprobs=[accum_response_logprobs] if len(accum_response_logprobs) > 0 else None,
            prompt_logprobs=final_prompt_logprobs,
        )

    async def _chat_completion_with_retry(
        self, engine_idx: int, original_request_payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Keep sending `chat_completion` requests (with previous responses accumulated) until the finish_reason is not "abort".

        The retry mechanism is intended to be used in combination with `pause_generation()` and `resume_generation()` for
        in-flight weight updates and partial rollouts.

        This method is equivalent to a single `chat_completion()` call if we do not use `pause_generation()`.

        For subsequent retry requests, we can reuse the original request with the following exceptions:
        - Update the last assistant message content to accumulated content, where the role uses the first non-empty response's role.
        - Set continue_final_message=True and add_generation_prompt=False.
        - Adjust remaining max tokens if `max_tokens` or `max_completion_tokens` is present.
        - If no tokens have been generated yet, resend the original request unchanged.

        For the final response, we maintain all the first non-empty response's fields (i.e. prefilled already),
        with the following exceptions:
        - Accumulate the following across retry requests:
          - `choices[0]["logprobs"]["content"]`
          - `choices[0]["token_ids"]`
          - `choices[0]["message"]["content"]`
        - Use the last response's finish_reason and stop_reason
        """
        original_request_json: Dict[str, Any] = original_request_payload.get("json", {}).copy()
        headers: Dict[str, str] = original_request_payload.get("headers", {}).copy()

        assert not original_request_json.get(
            "continue_final_message", False
        ), "continue_final_message must be False for /chat/completions requests"

        # Accumulated fields for building subsequent requests and final response. It is inplace-updated
        # in `_parse_partial_response_and_inplace_update_accum()`.
        accum = AccumulatedResponse()

        # First non-empty response (i.e. the response that prefilled the prompt) to copy meta from.
        base_response: Optional[Dict[str, Any]] = None

        # Determine original max tokens key and value (if any)
        max_key = None
        if "max_tokens" in original_request_json:
            max_key = "max_tokens"
        elif "max_completion_tokens" in original_request_json:
            max_key = "max_completion_tokens"
        orig_max_tokens: Optional[int] = original_request_json.get(max_key) if max_key else None

        # Fields to be updated in each loop iteration
        finish_reason: str = "abort"
        stop_reason: Optional[str] = None
        response_role: Optional[str] = None

        # 1. Loop until the generation is completed.
        while finish_reason == "abort":
            await self._wait_for_generation_to_resume()

            # 1.1. Prepare the request payload.
            cur_request_json = _prepare_retry_request(
                original_request_json=original_request_json,
                accum=accum,
                response_role=response_role,
                orig_max_tokens=orig_max_tokens,
                max_key=max_key,
            )

            # 1.2. Send the request.
            logger.debug(f"/chat/completions request sent (including potential retries): {cur_request_json}")
            try:
                partial_response = await self.engines[engine_idx].chat_completion(
                    {"json": cur_request_json, "headers": headers}
                )
            except (ray.exceptions.ActorDiedError, ray.exceptions.RayActorError) as e:
                self._mark_engine_dead(engine_idx, e)
                fallback = self._pick_fallback_engine(engine_idx)
                if fallback is None:
                    raise RuntimeError("All inference engines have died") from e
                engine_idx = fallback
                # Reset accumulator — new engine has no prior context
                accum = AccumulatedResponse()
                base_response = None
                response_role = None
                finish_reason = "abort"
                continue

            # 1.2.1. Check for error response from vLLM/sglang.
            # Error responses have "error" key (vLLM) or "object"="error" (sglang), not "choices".
            if "error" in partial_response or partial_response.get("object", "") == "error":
                error_info = partial_response.get("error", partial_response)
                error_msg = error_info.get("message", str(error_info)) if isinstance(error_info, dict) else str(error_info)

                # Handle continue_final_message errors by falling back to fresh request.
                # This can happen when chat templates (e.g., Qwen3 thinking) modify assistant
                # content in ways that make vLLM unable to find the continuation point.
                if "continue_final_message" in error_msg and accum.completion_tokens > 0:
                    logger.warning(
                        f"continue_final_message failed after {accum.completion_tokens} tokens, retrying fresh"
                    )
                    # Reset accumulator and retry with original request
                    accum = AccumulatedResponse()
                    response_role = None
                    continue

                # Return the error response dict instead of raising, so that
                # the HTTP endpoint can forward it with the correct status code
                # (e.g., 400 for context length errors). Raising RuntimeError here
                # would cause the endpoint to wrap it as a generic HTTP 500, which
                # breaks LiteLLM's error classification in downstream consumers
                # like Harbor.
                logger.warning(f"Inference engine error: {error_msg}")
                return partial_response

            # 1.3. Parse partial response and in-place update accumulators.
            finish_reason, stop_reason, response_role, aborted_without_generating = (
                _parse_partial_response_and_inplace_update_accum(
                    partial_response=partial_response,
                    accum=accum,
                    response_role=response_role,
                )
            )

            # 1.4. Aborted without generating tokens, so partial_response is useless.
            if aborted_without_generating:
                continue

            # At this point, either some tokens were generated and/or request completed with a non-"abort" finish_reason

            # 1.5. Update base response if it is the first non-empty response
            if base_response is None:
                if finish_reason != "abort":
                    # If we only made one request and it is not aborted, return the partial result directly.
                    # This is the codepath that will hit when we do not use `pause_generation()` or `resume_generation()`.
                    return partial_response
                # NOTE(Charlie): not doing deepcopy here to avoid copying large logprobs, so be careful when modifying this.
                base_response = partial_response.copy()

        # 2. Build final response by combining fields
        assert base_response is not None, "Expected at least one non-empty response to build final response"
        return _build_final_response(
            base_response=base_response,
            accum=accum,
            finish_reason=finish_reason,
            stop_reason=stop_reason,
        )

    async def chat_completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        session_id = request_payload["json"].pop("session_id", None)
        if session_id is None:
            engine_idx = random.randint(0, len(self.engines) - 1)
        else:
            assert isinstance(session_id, (str, int)), "Session ID must be an integer or string for `/chat/completions`"
            engine_idx = hash_with_sha256(str(session_id)) % len(self.engines)
        engine_idx = self._resolve_engine_idx(engine_idx)

        # Always use the retry loop which also issues the first request inside
        return await self._chat_completion_with_retry(engine_idx, request_payload)

    async def completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handles an OpenAI /completions request.

        Since `request["prompt"]` can be `Union[list[int], list[list[int]], str, list[str]]`,
        (i.e. {batched, single} x {string, token IDs}), we need to route the request to engines
        differently, based on whether it's a single or batched request, and whether `request["session_id"]`
        is provided. This is similar to `generate()` method.

        For single, we do the same routing logic as `chat_completion()`. For batched, we route by
        `request["session_id"]` if present, and if not we split evenly across engines.

        Regardless, the order will be maintained, i.e. `output["choices"][i]` corresponds to `request["prompt"][i]`.
        """
        if self.generation_paused_event.is_set():
            raise RuntimeError("pause_generation is unsupported for /completions requests.")
        body = request_payload.get("json", {})

        # NOTE(Charlie): do not reuse headers here as the single request may become various new requests
        headers = {"Content-Type": "application/json"}

        # 1. Postprocess prompt, session_id, and validate request.
        prompt = body.get("prompt")
        session_id_value = body.pop("session_id", None)
        ret = postprocess_completion_request(prompt, session_id_value)
        session_id_list: Optional[Union[List[int], List[str], ErrorResponse]] = ret[0]
        prompt: Union[List[List[int]], List[str]] = ret[1]
        if isinstance(session_id_list, ErrorResponse):
            return session_id_list.model_dump()

        num_prompts = len(prompt)
        num_inference_engines = len(self.engines)
        assert num_prompts > 0, "Number of prompts must be greater than 0"

        # 1. Route prompts to engines
        engine_idx_to_prompt_ids: dict[int, list[int]] = route_prompts_to_engines(
            num_prompts=num_prompts,
            num_inference_engines=num_inference_engines,
            session_ids=session_id_list,
        )

        # 2. Generate responses concurrently (with failover for dead engines)
        tasks: list[asyncio.Task] = []
        indices_list: list[list[int]] = []  # the original prompt indices that each task works on
        task_engine_idxs: list[int] = []  # engine idx for each task (for failover)

        # Reroute prompts away from known-dead engines before dispatching
        rerouted_mapping: dict[int, list[int]] = {}
        for engine_idx, prompt_ids in engine_idx_to_prompt_ids.items():
            resolved = self._resolve_engine_idx(engine_idx)
            rerouted_mapping.setdefault(resolved, []).extend(prompt_ids)

        for engine_idx, prompt_ids in rerouted_mapping.items():
            cur_prompt = [prompt[i] for i in prompt_ids]
            cur_json = dict(body)
            cur_json["prompt"] = cur_prompt
            coro = self.engines[engine_idx].completion({"json": cur_json, "headers": headers})
            tasks.append(asyncio.create_task(coro))
            indices_list.append(prompt_ids)
            task_engine_idxs.append(engine_idx)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 2.1. Handle engine deaths: retry failed tasks on fallback engines
        for i, result in enumerate(results):
            if isinstance(result, (ray.exceptions.ActorDiedError, ray.exceptions.RayActorError)):
                self._mark_engine_dead(task_engine_idxs[i], result)
                fallback = self._pick_fallback_engine(task_engine_idxs[i])
                if fallback is None:
                    raise RuntimeError("All inference engines have died") from result
                cur_prompt = [prompt[j] for j in indices_list[i]]
                cur_json = dict(body)
                cur_json["prompt"] = cur_prompt
                results[i] = await self.engines[fallback].completion(
                    {"json": cur_json, "headers": headers}
                )
            elif isinstance(result, BaseException):
                raise result

        # 3. Check for errors.
        # results can be ErrorResponse or CompletionResponse. If one of the sub-requests fails, we
        # return an error response. That is, there is no partial success, following vLLM and SGLang's behavior.
        for result in results:
            if "error" in result or result.get("object", "") == "error":
                # former is vllm format, latter is sglang format
                error_details = result.get("error", result)  # resolves vllm/sglang format difference
                error_code = error_details["code"]
                error_type = error_details["type"]
                return ErrorResponse(
                    error=ErrorInfo(
                        message=f"In one of the engines that SkyRL manages, an error occurred: {error_details['message']}",
                        type=error_type,
                        code=error_code,
                    ),
                ).model_dump()

        # 4. Combine choices and preserve original order.
        # If there is only one result, we return it directly.
        if len(results) == 1:
            return results[0]

        # Use the first result as base response. There are some fields that cannot be shared
        # across sub-requests. For now it is just the usage field.
        final_response = dict(results[0])
        final_response["usage"] = aggregate_completion_usage_info(results, self.backend)

        # Aggregate choices. TODO(Charlie): improve logic when we need to support n > 1
        # vLLM sets index positions per sub-batch, so we reset indices to be 0..n-1 for the combined response.
        combined_choices: list[Dict[str, Any]] = [None] * num_prompts
        for indices, result in zip(indices_list, results):
            # indices are the original prompt indices that the task's response corresponds to
            for local_idx, original_idx in enumerate(indices):
                choice = result["choices"][local_idx]
                choice["index"] = original_idx  # overwrite index with the global position
                combined_choices[original_idx] = choice

        # sanity check that the index is correct
        for new_idx in range(len(combined_choices)):
            assert combined_choices[new_idx]["index"] == new_idx

        final_response["choices"] = combined_choices
        return final_response

    async def wake_up(self, *args: Any, **kwargs: Any):
        return await self._run_on_all_engines("wake_up", *args, **kwargs)

    async def sleep(self, *args: Any, **kwargs: Any):
        return await self._run_on_all_engines("sleep", *args, **kwargs)

    async def init_weight_update_communicator(
        self,
        master_addr,
        master_port,
        rank_offset,
        world_size,
        group_name,
        backend,
        override_existing: bool = False,
    ):
        tasks = []
        rank_offset_count = rank_offset

        for i, engine in enumerate(self.engines):
            if i in self._dead_engines:
                continue
            tasks.append(
                engine.init_weight_update_communicator(
                    master_addr=master_addr,
                    master_port=master_port,
                    rank_offset=rank_offset_count,
                    world_size=world_size,
                    group_name=group_name,
                    backend=backend,
                    override_existing=override_existing,
                )
            )
            rank_offset_count += engine.tp_size() * engine.pp_size()
        await asyncio.gather(*tasks)

    async def update_named_weights(self, request: NamedWeightsUpdateRequest):
        return await self._run_on_all_engines("update_named_weights", request=request)

    async def reset_prefix_cache(self):
        return await self._run_on_all_engines("reset_prefix_cache")

    async def teardown(self):
        return await self._run_on_all_engines("teardown")

    async def get_stats(self) -> Dict[str, Any]:
        """Get aggregated statistics from all vLLM inference engines.

        Queries each engine for its accumulated stats (peak and median values since last read)
        and returns aggregated metrics. Stats are reset on each engine after reading,
        so each training step gets fresh stats.

        Returns:
            Dict with keys:
            - engines: List of per-engine stats dicts
            - num_engines: Total number of engines
            - Peak metrics (sum/avg across engines):
              - total_peak_running_reqs, total_peak_waiting_reqs
              - avg_peak_prompt_throughput, avg_peak_generation_throughput
              - avg_peak_gpu_cache_usage_perc, avg_peak_prefix_cache_hit_rate
            - Median metrics (avg across engines):
              - avg_median_prompt_throughput, avg_median_generation_throughput
              - avg_median_running_reqs, avg_median_waiting_reqs
            - total_samples, total_active_samples

        Used by VLLMStatsCallback to log engine stats without relying on
        Ray's log-to-driver functionality.
        """
        engine_stats_list = await self._run_on_all_engines("get_stats")
        num_engines = len(engine_stats_list)

        if num_engines == 0:
            return {
                "engines": [],
                "num_engines": 0,
                # Peak metrics
                "total_peak_running_reqs": 0,
                "total_peak_waiting_reqs": 0,
                "avg_peak_prompt_throughput": 0.0,
                "avg_peak_generation_throughput": 0.0,
                "avg_peak_gpu_cache_usage_perc": 0.0,
                "avg_peak_prefix_cache_hit_rate": 0.0,
                # Median metrics
                "avg_median_prompt_throughput": 0.0,
                "avg_median_generation_throughput": 0.0,
                "avg_median_running_reqs": 0.0,
                "avg_median_waiting_reqs": 0.0,
                "avg_median_gpu_cache_usage_perc": 0.0,
                "avg_median_prefix_cache_hit_rate": 0.0,
                # Per-request latency stats
                "avg_latency_prefill_mean": 0.0, "max_latency_prefill_p90": 0.0, "avg_latency_prefill_median": 0.0,
                "avg_latency_decode_mean": 0.0, "max_latency_decode_p90": 0.0, "avg_latency_decode_median": 0.0,
                "avg_latency_e2e_mean": 0.0, "max_latency_e2e_p90": 0.0, "avg_latency_e2e_median": 0.0,
                "avg_latency_queued_mean": 0.0, "max_latency_queued_p90": 0.0, "avg_latency_queued_median": 0.0,
                "avg_latency_ttft_mean": 0.0, "max_latency_ttft_p90": 0.0, "avg_latency_ttft_median": 0.0,
                "total_finished_requests": 0,
                "total_preempted_reqs": 0,
                # Legacy field names for backwards compatibility
                "total_running_reqs": 0,
                "total_waiting_reqs": 0,
                "avg_prompt_throughput": 0.0,
                "avg_generation_throughput": 0.0,
                "avg_gpu_cache_usage_perc": 0.0,
                "avg_prefix_cache_hit_rate": 0.0,
                # Metadata
                "total_samples": 0,
                "total_active_samples": 0,
            }

        # Aggregate PEAK stats
        total_peak_running = sum(
            s.get("peak_running_reqs", s.get("num_running_reqs", 0))
            for s in engine_stats_list
        )
        total_peak_waiting = sum(
            s.get("peak_waiting_reqs", s.get("num_waiting_reqs", 0))
            for s in engine_stats_list
        )
        avg_peak_prompt_tp = sum(
            s.get("peak_prompt_throughput", s.get("avg_prompt_throughput", 0.0))
            for s in engine_stats_list
        ) / num_engines
        avg_peak_gen_tp = sum(
            s.get("peak_generation_throughput", s.get("avg_generation_throughput", 0.0))
            for s in engine_stats_list
        ) / num_engines
        avg_peak_gpu_cache = sum(
            s.get("peak_gpu_cache_usage_perc", s.get("gpu_cache_usage_perc", 0.0))
            for s in engine_stats_list
        ) / num_engines
        avg_peak_prefix_hit = sum(
            s.get("peak_prefix_cache_hit_rate", s.get("prefix_cache_hit_rate", 0.0))
            for s in engine_stats_list
        ) / num_engines

        # Aggregate MEDIAN stats (average of medians across engines)
        avg_median_prompt_tp = sum(
            s.get("median_prompt_throughput", 0.0) for s in engine_stats_list
        ) / num_engines
        avg_median_gen_tp = sum(
            s.get("median_generation_throughput", 0.0) for s in engine_stats_list
        ) / num_engines
        avg_median_running = sum(
            s.get("median_running_reqs", 0.0) for s in engine_stats_list
        ) / num_engines
        avg_median_waiting = sum(
            s.get("median_waiting_reqs", 0.0) for s in engine_stats_list
        ) / num_engines
        avg_median_gpu_cache = sum(
            s.get("median_gpu_cache_usage_perc", 0.0) for s in engine_stats_list
        ) / num_engines
        avg_median_prefix_hit = sum(
            s.get("median_prefix_cache_hit_rate", 0.0) for s in engine_stats_list
        ) / num_engines

        # Total samples collected (useful for debugging stats collection)
        total_samples = sum(s.get("num_samples", 0) for s in engine_stats_list)
        total_active_samples = sum(s.get("num_active_samples", 0) for s in engine_stats_list)

        # Aggregate per-request latency stats (weighted average across engines by finished request count)
        latency_keys = ["prefill", "decode", "e2e", "queued", "ttft"]
        latency_agg = {}
        total_finished = sum(s.get("latency_num_finished_requests", 0) for s in engine_stats_list)
        total_preempted = sum(s.get("total_preempted_reqs", 0) for s in engine_stats_list)
        for key in latency_keys:
            if total_finished > 0:
                # Weighted mean across engines
                latency_agg[f"avg_latency_{key}_mean"] = sum(
                    s.get(f"latency_{key}_mean", 0.0) * s.get("latency_num_finished_requests", 0)
                    for s in engine_stats_list
                ) / total_finished
                # Max of p90s across engines (worst-case engine)
                latency_agg[f"max_latency_{key}_p90"] = max(
                    (s.get(f"latency_{key}_p90", 0.0) for s in engine_stats_list), default=0.0
                )
                # Avg of medians across engines
                latency_agg[f"avg_latency_{key}_median"] = sum(
                    s.get(f"latency_{key}_median", 0.0) for s in engine_stats_list
                ) / num_engines
            else:
                latency_agg[f"avg_latency_{key}_mean"] = 0.0
                latency_agg[f"max_latency_{key}_p90"] = 0.0
                latency_agg[f"avg_latency_{key}_median"] = 0.0

        return {
            "engines": engine_stats_list,
            "num_engines": num_engines,
            # Peak metrics
            "total_peak_running_reqs": total_peak_running,
            "total_peak_waiting_reqs": total_peak_waiting,
            "avg_peak_prompt_throughput": avg_peak_prompt_tp,
            "avg_peak_generation_throughput": avg_peak_gen_tp,
            "avg_peak_gpu_cache_usage_perc": avg_peak_gpu_cache,
            "avg_peak_prefix_cache_hit_rate": avg_peak_prefix_hit,
            # Median metrics
            "avg_median_prompt_throughput": avg_median_prompt_tp,
            "avg_median_generation_throughput": avg_median_gen_tp,
            "avg_median_running_reqs": avg_median_running,
            "avg_median_waiting_reqs": avg_median_waiting,
            "avg_median_gpu_cache_usage_perc": avg_median_gpu_cache,
            "avg_median_prefix_cache_hit_rate": avg_median_prefix_hit,
            # Aggregated per-request latency stats (seconds)
            **latency_agg,
            "total_finished_requests": total_finished,
            "total_preempted_reqs": total_preempted,
            # Legacy field names for backwards compatibility (use peak values)
            "total_running_reqs": total_peak_running,
            "total_waiting_reqs": total_peak_waiting,
            "avg_prompt_throughput": avg_peak_prompt_tp,
            "avg_generation_throughput": avg_peak_gen_tp,
            "avg_gpu_cache_usage_perc": avg_peak_gpu_cache,
            "avg_prefix_cache_hit_rate": avg_peak_prefix_hit,
            # Metadata
            "total_samples": total_samples,
            "total_active_samples": total_active_samples,
        }

    def tp_size(self) -> int:
        raise NotImplementedError("InferenceEngineClient does not implement tp_size()")

    def pp_size(self) -> int:
        raise NotImplementedError("InferenceEngineClient does not implement pp_size()")

    def dp_size(self) -> int:
        raise NotImplementedError("InferenceEngineClient does not implement dp_size()")

    # ----------------------------
    # Generation pause and resume
    # ----------------------------
    async def _wait_for_generation_to_resume(self) -> None:
        """Waits for generation to be resumed, intended for in-flight weight updates and partial rollouts."""
        while self.generation_paused_event.is_set():
            await asyncio.sleep(0.5)

    async def pause_generation(self) -> None:
        """
        Pauses generation for all engines, intended for in-flight weight updates and partial rollouts.

        Currently only supported for `/chat/completions` and not `/completions` or `generate()`.

        Both in-flight and incoming requests will be blocked until `resume_generation` is called.
        1. Set the paused event to avoid new requests from being submitted while aborting requests.
        2. Wait for a grace period to ensure all in-flight requests have entered the engine's
           scheduler and hence can be aborted. Otherwise, there can be requests already submitted
           but not yet entered the scheduler, which can miss the abort request.
        3. Finally, we abort requests on all engines. This will cause the requests sent from
           InferenceEngineClient to `InferenceEngineClient.engines` to return the already-generated tokens.
           The request to `InferenceEngineClient` will not yet return until requests are completed with
           stop reason that is not `abort`.
        """
        if self.generation_paused_event.is_set():
            raise RuntimeError("Generation is already paused, cannot pause again.")
        self.generation_paused_event.set()
        await asyncio.sleep(ABORT_GENERATION_GRACE_PERIOD_SECONDS)
        await self._run_on_all_engines("abort_generation")

    async def resume_generation(self) -> None:
        """
        Resumes generation for all engines, intended for in-flight weight updates and partial rollouts.

        Resume all in-flight requests with the previously-generated tokens, and unblock incoming requests
        that were blocked by `pause_generation()`.
        """
        if not self.generation_paused_event.is_set():
            raise RuntimeError("Generation is not paused, cannot resume.")
        self.generation_paused_event.clear()

    async def abort_generation(self) -> None:
        raise NotImplementedError(
            "InferenceEngineClient does not implement abort_generation(), but calls "
            "`abort_generation` on all engines in `pause_generation()`."
        )

    # ----------------------------
    # HTTP endpoint related methods
    # ----------------------------

    def __del__(self):
        """
        Destructor to shut down the HTTP endpoint if it was started.
        """
        # TODO(Charlie): __del__ is not guaranteed to be called in general. Add to `teardown` method
        # when the `_handle_termination` flow is implemented. See `skyrl_train/workers/worker.py`
        # comments on `_handle_termination` for more details.
        if (
            self.enable_http_endpoint
            and hasattr(
                self, "_server_thread"
            )  # don't want to shut down the server when it is pickled as a ray method argument.
            and self._server_thread is not None
        ):
            try:
                from skyrl_train.inference_engines.inference_engine_client_http_endpoint import shutdown_server

                shutdown_server(
                    host=self.http_endpoint_host,
                    port=self.http_endpoint_port,
                    max_wait_seconds=10,
                )
                if hasattr(self, "_server_thread") and self._server_thread.is_alive():
                    self._server_thread.join(timeout=10)
            except Exception as e:
                logger.error(f"Error shutting down HTTP endpoint: {e}")

    def __getstate__(self):
        """
        Override to avoid pickling the server thread and the threading.Event object, which are not picklable.
        Needed when passing InferenceEngineClient as an argument to async_run_ray_method(), mainly for
        invoking `init_weight_sync_state()` and `broadcast_to_inference_engines()`, which do
        not need these attributes.
        """
        state = self.__dict__.copy()
        state["_server_thread"] = None
        state["generation_paused_event"] = None
        return state

    def _spin_up_http_endpoint(self):
        from skyrl_train.inference_engines.inference_engine_client_http_endpoint import (
            serve,
            wait_for_server_ready,
        )

        # Bind the uvicorn server to 0.0.0.0 so that off-node clients (e.g. the
        # rollout-fanout RolloutCoordinator actors running on WORKER nodes) can
        # reach the endpoint over the internal compute network. The configured
        # `http_endpoint_host` (default 127.0.0.1) is the CLIENT-side host used
        # for the local readiness probe below; binding the SERVER to 0.0.0.0 is
        # a superset of binding to 127.0.0.1, so the loopback readiness probe and
        # the fan-out-OFF path (client uses 127.0.0.1 on the same node) are both
        # unaffected. This is safe: the endpoint only lives on the internal
        # 10.128.x.x compute net, never the public internet.
        self._server_thread = threading.Thread(
            target=serve,
            args=(self,),
            kwargs={
                "host": "0.0.0.0",
                "port": self.http_endpoint_port,
                "log_level": "warning",
            },
            daemon=True,
        )
        self._server_thread.start()
        wait_for_server_ready(
            host=self.http_endpoint_host,
            port=self.http_endpoint_port,
            max_wait_seconds=30,
        )
        logger.info(
            f"InferenceEngineClient HTTP endpoint started on {self.http_endpoint_host}:{self.http_endpoint_port}"
        )

    def shutdown_http_endpoint(self) -> None:
        """Shut down the HTTP endpoint server.

        Must be called during teardown BEFORE killing inference engines.
        Otherwise in-flight requests get forwarded to dead engines, receive
        ActorDiedError, and the retry logic keeps the process alive
        indefinitely.
        """
        if not self.enable_http_endpoint:
            return

        from skyrl_train.inference_engines.inference_engine_client_http_endpoint import (
            shutdown_server,
        )

        try:
            shutdown_server(
                host=self.http_endpoint_host,
                port=self.http_endpoint_port,
                max_wait_seconds=10,
            )
        except Exception as e:
            logger.warning(f"HTTP endpoint shutdown error (non-fatal): {e}")

        if self._server_thread is not None and self._server_thread.is_alive():
            self._server_thread.join(timeout=5)
            if self._server_thread.is_alive():
                logger.warning("HTTP endpoint thread did not exit within 5s")


# ----------------------------------------------
# Helper methods for _chat_completion_with_retry
# ----------------------------------------------


@dataclass
class AccumulatedResponse:
    content: str = ""
    logprobs_content: List[Any] = field(default_factory=list)
    token_ids: List[int] = field(default_factory=list)
    completion_tokens: int = 0


def _prepare_retry_request(
    original_request_json: Dict[str, Any],
    accum: AccumulatedResponse,
    response_role: Optional[str],
    orig_max_tokens: Optional[int],
    max_key: Optional[str],
) -> Dict[str, Any]:
    """Build the per-iteration request payload.

    If no tokens have been generated yet, resend the original request unchanged.
    Otherwise, build a continuation request that appends the accumulated content
    and adjusts remaining max tokens if present.
    """
    if accum.completion_tokens == 0:
        return original_request_json.copy()

    assert accum.content != "", "accum.content must be non-empty for a continuation request"
    assert response_role is not None, "response_role must be set for a continuation request"

    cur_request_json = original_request_json.copy()
    cur_request_json["messages"] = original_request_json["messages"] + [
        {"role": response_role, "content": accum.content}
    ]
    cur_request_json["continue_final_message"] = True
    cur_request_json["add_generation_prompt"] = False
    if orig_max_tokens is not None:
        assert (
            orig_max_tokens - accum.completion_tokens >= 0
        ), "orig_max_tokens - accum.completion_tokens must be non-negative"
        assert max_key is not None
        cur_request_json[max_key] = orig_max_tokens - accum.completion_tokens

    return cur_request_json


def _parse_partial_response_and_inplace_update_accum(
    partial_response: Dict[str, Any],
    accum: AccumulatedResponse,
    response_role: Optional[str],
) -> tuple[str, Optional[str], Optional[str], bool]:
    """Parse the partial response and in-place update accumulators.

    Returns (finish_reason, stop_reason, response_role, aborted_without_generating).
    """
    choice = partial_response["choices"][0]
    finish_reason: str = choice["finish_reason"]
    stop_reason: Optional[str] = choice.get("stop_reason", None)
    new_content: str = choice["message"]["content"]

    assert (
        partial_response["usage"] is not None and partial_response["usage"]["completion_tokens"] is not None
    ), "partial_response['usage']['completion_tokens'] must be present"
    new_completion_tokens: int = partial_response["usage"]["completion_tokens"]

    if response_role is None:
        response_role = choice["message"]["role"]
    else:
        assert response_role == choice["message"]["role"], "response_role must be the same across retries"

    # If aborted without generating tokens, ignore this partial response.
    aborted_without_generating = finish_reason == "abort" and new_completion_tokens == 0
    if not aborted_without_generating:
        accum.content += new_content
        logprobs = choice.get("logprobs")
        if logprobs is not None and logprobs.get("content") is not None:
            accum.logprobs_content.extend(logprobs["content"])
        if choice.get("token_ids") is not None:
            accum.token_ids.extend(choice["token_ids"])
        accum.completion_tokens += new_completion_tokens

    return finish_reason, stop_reason, response_role, aborted_without_generating


def _build_final_response(
    base_response: Dict[str, Any],
    accum: AccumulatedResponse,
    finish_reason: str,
    stop_reason: Optional[str],
) -> Dict[str, Any]:
    """Construct the final aggregated response from the base and accumulators."""
    # NOTE(Charlie): not doing deepcopy for performance. Be careful when re-using this method
    # as it mutates base_response.
    final_response = base_response

    # Combine usage: prompt_tokens from base, completion_tokens summed, total_tokens accordingly
    base_usage = final_response["usage"]
    prompt_tokens = base_usage["prompt_tokens"]
    final_usage = base_usage.copy()
    final_usage["completion_tokens"] = accum.completion_tokens
    final_usage["total_tokens"] = prompt_tokens + accum.completion_tokens
    final_response["usage"] = final_usage

    # Set accumulated content, logprobs, token_ids.
    final_choice = final_response["choices"][0]
    final_choice["message"]["content"] = accum.content
    if final_choice.get("logprobs", None) is not None:
        final_choice["logprobs"]["content"] = accum.logprobs_content
    if final_choice.get("token_ids", None) is not None:
        final_choice["token_ids"] = accum.token_ids

    # Set last response's finish_reason and stop_reason.
    final_choice["finish_reason"] = finish_reason
    if stop_reason is not None:
        final_choice["stop_reason"] = stop_reason

    return final_response
