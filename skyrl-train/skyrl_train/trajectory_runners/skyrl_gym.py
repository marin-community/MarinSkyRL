"""
This file implements ``SkyRLGymTrajectoryRunner``, an implementation of the `TrajectoryRunner` that
uses SkyRL-Gym as the environment.

For details, see https://skyrl.readthedocs.io/en/latest/tutorials/skyrl_gym_runner.html
"""

from __future__ import annotations

import asyncio
import copy
import json
from dataclasses import dataclass
from uuid import uuid4
import skyrl_gym
from typing import Callable, Generic, List, Dict, Any, Optional, Sequence, Tuple, TypeVar
from concurrent.futures import ThreadPoolExecutor
from loguru import logger

from skyrl_train.trajectory_runners.base import TrajectoryRunner, TrajectoryRequestBatch, TrajectoryBatch, TrajectoryID
from skyrl_train.trajectory_runners.types import AgentLoopOutput, TokenProvenance
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.base import InferenceEngineInput, ConversationType
from omegaconf import DictConfig
from skyrl_gym.envs.base_text_env import BaseTextEnvStepOutput
from skyrl_gym.envs.nemotron_ultra.genrm import grade_genrm_group, response_object
from skyrl_gym.envs.nemotron_ultra.judge import OpenAIJudge
from skyrl_gym.verification import RewardResult, RolloutEvidence, TrainingDisposition, VerificationResult
from skyrl_train.trajectory_runners.skyrl_gym_contracts import (
    environment_metrics_from_step,
    fold_verification_results,
    publish_rollout_evidence,
    reward_from_env_step,
    verification_from_env_step,
)
from skyrl_train.trajectory_runners.trajectory_processing import (
    get_custom_chat_template,
    get_generation_prompt_ids,
    apply_overlong_filtering,
    get_rollout_metrics,
    normalize_token_ids,
)
from skyrl_train.trajectory_runners.model_clients import DirectModelClient, ModelClient
from skyrl_train.trajectory_runners.collectors import RolloutCollector, collect_agent_loops
from skyrl_train.trajectory_runners.projections import (
    attach_unshaped_rewards,
    IdentityTrajectoryProjection,
    TrajectoryProjection,
    WholeTrajectoryProjection,
)


class WholeTrajectoryCollector:
    """Collect one complete interaction record per request."""

    def __init__(self, runner):
        self._runner = runner

    def validate(self) -> None:
        pass

    async def collect(self, request: TrajectoryRequestBatch, *, disable_tqdm: bool = False):
        return await collect_agent_loops(self._runner, request, self._runner.agent_loop, disable_tqdm=disable_tqdm)


class BatchedTrajectoryCollector:
    """Collect a batch from the supported single-turn batched environment path."""

    def __init__(self, runner):
        self._runner = runner

    def validate(self) -> None:
        pass

    async def collect(self, request: TrajectoryRequestBatch, *, disable_tqdm: bool = False):
        del disable_tqdm
        runner = self._runner
        batch = await runner.collect_batched(
            request["prompts"],
            request["env_classes"],
            request["env_extras"],
            runner.trajectory_runner_cfg.sampling_params.max_generate_length,
            request.get("sampling_params"),
        )
        return batch


PipelineOutputT = TypeVar("PipelineOutputT")


@dataclass(frozen=True)
class TrajectoryPipeline(Generic[PipelineOutputT]):
    """A type-coupled collector and projection pair."""

    collector_type: Callable[[SkyRLGymTrajectoryRunner], RolloutCollector[PipelineOutputT]]
    projection: TrajectoryProjection[PipelineOutputT]


SkyRLGymPipeline = (
    TrajectoryPipeline[Sequence[AgentLoopOutput]]
    | TrajectoryPipeline[Sequence[Sequence[AgentLoopOutput]]]
    | TrajectoryPipeline[TrajectoryBatch]
)


@dataclass
class _CanonicalizedChatPrefix:
    loss_mask: List[int]
    rollout_logprobs: Optional[List[float]]
    per_step_rewards: List[Tuple[float, Optional[int]]]


class SkyRLGymTrajectoryRunner(TrajectoryRunner):
    def __init__(
        self,
        trajectory_runner_cfg: DictConfig,
        skyrl_gym_cfg: DictConfig,
        inference_engine_client: InferenceEngineClient,
        tokenizer,
        model_client: ModelClient | None = None,
        pipeline: SkyRLGymPipeline | None = None,
    ):
        """
        Args:
            trajectory_runner_cfg: trajectory-runner configuration
            skyrl_gym_cfg: environment configuration keyed by SkyRL-Gym environment name
            inference_engine_client: InferenceEngineClient object for interacting with the inference engines
            tokenizer: tokenizer object for encoding and decoding text
            model_client: optional transport-neutral model client
            pipeline: optional type-coupled harness collector and projection
        """
        self.trajectory_runner_cfg = trajectory_runner_cfg
        self.skyrl_gym_cfg = skyrl_gym_cfg
        self.model_client = model_client or DirectModelClient(inference_engine_client)
        self.tokenizer = tokenizer
        if pipeline is None:
            pipeline = (
                TrajectoryPipeline(BatchedTrajectoryCollector, IdentityTrajectoryProjection())
                if trajectory_runner_cfg.batched
                else TrajectoryPipeline(
                    WholeTrajectoryCollector,
                    WholeTrajectoryProjection(trajectory_runner_cfg, tokenizer),
                )
            )
        self.collector = pipeline.collector_type(self)
        self.projection = pipeline.projection
        self.max_turns = trajectory_runner_cfg.max_turns
        self.batched = trajectory_runner_cfg.batched
        self.use_conversation_multi_turn = trajectory_runner_cfg.use_conversation_multi_turn
        # optionally use custom chat template to get loss masks (i.e. for Qwen3)
        self.custom_chat_template = get_custom_chat_template(trajectory_runner_cfg.chat_template)
        # get generation prompt ids for the tokenizer if needed
        self.generation_prompt_ids = get_generation_prompt_ids(tokenizer) if self.use_conversation_multi_turn else None
        if self.skyrl_gym_cfg.max_env_workers > 0:
            self.env_executor = ThreadPoolExecutor(
                max_workers=self.skyrl_gym_cfg.max_env_workers, thread_name_prefix="skyrl-gym-env-"
            )
        else:
            self.env_executor = None

        self._validate_cfg(trajectory_runner_cfg)
        self.collector.validate()

        # base_conversation is used when `use_conversation_multi_turn==True and custom_chat_template==None` to
        # correctly format and tokenize observations into `observation_ids`.
        # Follows https://jybsuper.github.io/posts/multiturn_tokenization/#the-breakthrough-fixed-base-approach
        self.base_conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "I am a user."},
        ]
        self.base_conversation_token_ids = normalize_token_ids(
            tokenizer.apply_chat_template(
                self.base_conversation,
                add_generation_prompt=False,
                tokenize=True,
                **self.trajectory_runner_cfg.chat_template_kwargs,
            )
        )
        # We remove tokens after the last EOS token so that it can be captured in `observation_ids`.
        # For details, see https://skyrl.readthedocs.io/en/latest/tutorials/skyrl_gym_runner.html#multi-turn-tokenization-and-ti-to
        if self.tokenizer.eos_token_id in self.base_conversation_token_ids:
            last_eos_token_index = (
                len(self.base_conversation_token_ids)
                - 1
                - self.base_conversation_token_ids[::-1].index(self.tokenizer.eos_token_id)
            )
            self.base_conversation_token_ids = self.base_conversation_token_ids[: last_eos_token_index + 1]

        # Optional callback to get trainer's current global_step (for accurate staleness tracking).
        # Set by the fully-async trainer before generation workers start.
        self.global_step_fn: Optional[Callable[[], int]] = None

        ultra_config = skyrl_gym_cfg.get("nemotron_ultra", {})
        self.genrm_config = dict(ultra_config.get("genrm", {}))
        genrm_judge = self.genrm_config.get("judge")
        self.genrm_judge = OpenAIJudge(**dict(genrm_judge)) if genrm_judge is not None else None

    def _validate_cfg(self, trajectory_runner_cfg: DictConfig):
        if len(trajectory_runner_cfg.chat_template_kwargs) and trajectory_runner_cfg.batched:
            raise ValueError(
                "`chat_template_kwargs` is not compatible with `batched=True` since the chat templating is handled by the inference engine"
            )

    async def _run_in_executor_if_available(self, func, *args, **kwargs):
        if (executor := self.env_executor) is not None:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(executor, func, *args, **kwargs)
        else:
            return func(*args, **kwargs)

    async def agent_loop(
        self,
        prompt: ConversationType,
        env_class: str,
        env_extras: Dict[str, Any],
        max_tokens: int,
        max_input_length: int,
        sampling_params: Optional[Dict[str, Any]] = None,
        trajectory_id: Optional[TrajectoryID] = None,
        global_step_fn: Optional[Callable[[], int]] = None,
    ) -> AgentLoopOutput:
        """
        Multi-turn generation loop that executes a single trajectory.

        Note:
            We ensure token-in-token-out generation. With three exceptions:
            - When calling Env.step() and BaseTextEnvStepOutput["postprocessed_action"] is not None.
              This will likely be deprecated soon.
            - When custom_chat_template = True and use_conversation_multi_turn = True. We always
              re-tokenize the entire chat history every turn and at the end. This is used for cases
              like removing Qwen3 thinking tokens in non-last-round assistant message.
            - When the inference backend canonicalizes a structured tool call while rendering the
              next turn. The re-rendered prior context is retained but excluded from optimization.

        Args:
            prompt: ConversationType
            env_extras: Dict[str, Any]
            max_tokens: int
            max_input_length: int
            sampling_params: Optional[Dict[str, Any]]
        Returns:
            response_ids: List[int]
            reward: Union[float, List[float]]
            stop_reason: str
            loss_mask: List[int]
            prompt_token_ids: List[int]
            rollout_logprobs: Optional[List[float]]
        """
        retokenize_chat_history = self.use_conversation_multi_turn and self.custom_chat_template

        # Create a new environment instance
        env_extras["max_turns"] = self.max_turns  # TODO(shu): move this to config
        env_config = self.skyrl_gym_cfg.get(env_class, DictConfig({}))
        env = skyrl_gym.make(env_class, env_config=env_config, extras=env_extras)

        session_id = (
            f"{trajectory_id.instance_id}_{trajectory_id.repetition_id}" if trajectory_id is not None else uuid4().hex
        )
        done = False

        # Instantiate chat_history and chat_end_index, which are only used if `retokenize_chat_history==True`.
        # Need copy here since the prompt is a list of messages and we are going to modify it.
        chat_history = copy.deepcopy(prompt)

        # init() returns the first prompt to be given to the model, and optional metadata dict
        chat_history, init_metadata = await self._run_in_executor_if_available(env.init, chat_history)
        chat_completion_params = init_metadata.get("chat_completion_params")
        if chat_completion_params is not None and not isinstance(chat_completion_params, dict):
            raise TypeError("environment chat_completion_params metadata must be a mapping")
        initial_chat_history_length = len(chat_history)
        chat_end_index = len(chat_history)
        input_ids = normalize_token_ids(
            self.tokenizer.apply_chat_template(
                chat_history,
                # If retokenize_chat_history==True, avoid including the generation prompt in both the
                # prompt_ids and response_ids due to how `response_encodings["input_ids"]` works.
                add_generation_prompt=not retokenize_chat_history,
                chat_template=self.custom_chat_template if retokenize_chat_history else None,
                tokenize=True,
                **self.trajectory_runner_cfg.chat_template_kwargs,
            )
        )

        initial_prompt_length = len(input_ids)
        if initial_prompt_length > max_input_length:
            logger.warning(
                "Skipping generation because the templated initial prompt has {} tokens, exceeding max_input_length={}",
                initial_prompt_length,
                max_input_length,
            )
            env_metrics = env.get_metrics()
            await self._run_in_executor_if_available(env.close)
            return AgentLoopOutput(
                evidence=RolloutEvidence(
                    messages=tuple(chat_history),
                    stop_reason="length",
                    generated_token_count=0,
                    prompt_token_ids=tuple(input_ids),
                    response_token_ids=(),
                    behavior_logprobs=(),
                ),
                verification=VerificationResult.unavailable("initial prompt exceeds the model input limit"),
                reward=RewardResult(
                    unshaped_reward=None,
                    optimization_reward=0.0,
                    token_rewards=None if retokenize_chat_history else (),
                ),
                disposition=TrainingDisposition(
                    loss_eligible=False,
                    baseline_eligible=True,
                    reason="initial prompt exceeds the model input limit",
                ),
                loss_mask=[],
                env_metrics=env_metrics,
            )

        loss_mask = []  # this excludes the prompt
        current_sampling_params = (
            sampling_params if sampling_params is not None else self.trajectory_runner_cfg.sampling_params
        )
        collect_logprobs = current_sampling_params.get("logprobs", None) is not None
        rollout_logprobs: Optional[List[float]] = [] if collect_logprobs else None
        # Accumulate per-step rewards. Format: (reward, response_end_token_idx)
        per_step_rewards: List[Tuple[float, Optional[int]]] = []
        verification_results: List[VerificationResult] = []
        # Capture global_step at first inference for accurate staleness tracking
        captured_global_step: Optional[int] = None
        token_provenance = TokenProvenance.ENGINE

        while not done:
            if len(input_ids) > max_input_length:
                stop_reason = "length"
                break

            # 1. Generate output
            if retokenize_chat_history or chat_completion_params is not None:
                engine_input = InferenceEngineInput(
                    prompts=[copy.deepcopy(chat_history)],
                    session_ids=[session_id],
                    sampling_params=current_sampling_params,
                    **(
                        {"chat_completion_params": [chat_completion_params]}
                        if chat_completion_params is not None
                        else {}
                    ),
                )
            else:
                # Token-in-token-out.
                engine_input = InferenceEngineInput(
                    prompt_token_ids=[input_ids], session_ids=[session_id], sampling_params=sampling_params
                )
            engine_output = await self.model_client.generate(engine_input)
            if engine_output["token_provenance"] == TokenProvenance.RECONSTRUCTED:
                token_provenance = TokenProvenance.RECONSTRUCTED
            # Capture global_step after first inference returns — at this point the vLLM
            # engine has definitively served the request with its current weights.
            if captured_global_step is None and global_step_fn is not None:
                captured_global_step = global_step_fn()
            output = engine_output["responses"][0]
            output_ids = engine_output["response_ids"][0]
            if chat_completion_params is not None:
                rendered_prompt_ids = engine_output.get("prompt_ids")
                if rendered_prompt_ids is None:
                    raise RuntimeError("chat generation must return the backend-rendered prompt token IDs")
                if not per_step_rewards:
                    input_ids = rendered_prompt_ids[0]
                    initial_prompt_length = len(input_ids)
            stop_reason = engine_output["stop_reasons"][0]
            assistant_messages = engine_output.get("assistant_messages")
            assistant_message = assistant_messages[0] if assistant_messages is not None else None
            response_logprobs_batch = engine_output.get("response_logprobs")
            response_logprobs = response_logprobs_batch[0] if response_logprobs_batch is not None else None
            if response_logprobs is not None and len(response_logprobs) != len(output_ids):
                raise ValueError(
                    "Inference engine returned response logprobs that do not align with response token IDs: "
                    f"{len(response_logprobs)=}, {len(output_ids)=}"
                )
            if collect_logprobs and response_logprobs is None:
                rollout_logprobs = None

            if chat_completion_params is not None:
                if assistant_message is None:
                    raise RuntimeError("chat generation must return the structured assistant message")
                if per_step_rewards:
                    if input_ids != rendered_prompt_ids[0][: len(input_ids)]:
                        canonical_prefix = self._mask_canonicalized_chat_prefix(
                            input_ids,
                            rendered_prompt_ids[0],
                            initial_prompt_length,
                            rollout_logprobs,
                            per_step_rewards,
                        )
                        loss_mask = canonical_prefix.loss_mask
                        rollout_logprobs = canonical_prefix.rollout_logprobs
                        per_step_rewards = canonical_prefix.per_step_rewards
                        token_provenance = TokenProvenance.RECONSTRUCTED
                    else:
                        observation_token_count = len(rendered_prompt_ids[0]) - len(input_ids)
                        loss_mask += [0] * observation_token_count
                        if rollout_logprobs is not None:
                            rollout_logprobs += [0.0] * observation_token_count
                input_ids = rendered_prompt_ids[0]

            # Append eos when sampling_params.stop is not None. Does not affect 3.a as chat templates add eos_token.
            # sampling_params is not None for eval, but None for training (which uses engine.sampling_params which are from cfg)
            stop_strs = current_sampling_params.get("stop", None)
            added_eos = False
            if (
                stop_strs is not None
                and self.trajectory_runner_cfg.append_eos_token_after_stop_str_in_multi_turn
                and self.use_conversation_multi_turn
            ):
                if output.endswith(tuple(stop_strs)) and output_ids[-1] != self.tokenizer.eos_token_id:
                    output_ids.append(self.tokenizer.eos_token_id)
                    if response_logprobs is not None:
                        response_logprobs.append(0.0)
                    added_eos = True

            # 2. Environment step
            publish_rollout_evidence(
                env,
                response=output,
                stop_reason=stop_reason,
                prompt_token_ids=input_ids,
                response_token_ids=output_ids,
                behavior_logprobs=response_logprobs,
                metadata={
                    "generation_token_budget": max_tokens,
                    **({"assistant_message": assistant_message} if assistant_message is not None else {}),
                },
            )
            env_step_output: BaseTextEnvStepOutput = await self._run_in_executor_if_available(env.step, output)
            new_obs = env_step_output["observations"]
            step_reward: float = env_step_output["reward"]
            verification_results.append(verification_from_env_step(env_step_output))
            done = env_step_output["done"]

            reset_conversation = env_step_output.get("reset_conversation")
            if reset_conversation is not None:
                if done or chat_completion_params is None:
                    raise ValueError("reset_conversation requires a continuing structured-chat trajectory")
                # NVIDIA's proof-refinement agent issues every correction as a
                # fresh single-turn request and returns only the final attempt to
                # NeMo RL. Discard this failed attempt before the next render.
                chat_history = copy.deepcopy(reset_conversation)
                initial_chat_history_length = len(chat_history)
                input_ids = []
                initial_prompt_length = 0
                loss_mask = []
                rollout_logprobs = [] if collect_logprobs else None
                per_step_rewards = []
                verification_results = []
                continue

            if env_step_output.get("postprocessed_action", None) is not None:
                # TODO(Charlie): come back to this, we should deprecate postprocessed action
                logger.warning(
                    "WARNING: postprocessed action may violate token-in-token-out. Ideally you "
                    "post-process it in the token space rather than string space. "
                    "A better solution coming soon."
                )
                output = env_step_output["postprocessed_action"]
                postprocessed_output_ids = self.tokenizer.encode(output, add_special_tokens=False)
                if collect_logprobs and response_logprobs is not None and postprocessed_output_ids != output_ids:
                    logger.warning(
                        "Discarding rollout logprobs because postprocessed_action changed the generated token IDs"
                    )
                    response_logprobs = None
                    rollout_logprobs = None
                output_ids = postprocessed_output_ids

            # 3. Update states: input ids, loss_mask, chat_history, etc.
            # Three ways of managing input
            if chat_completion_params is not None:
                input_ids += output_ids
                loss_mask += [1] * len(output_ids)
                response_end_idx = len(input_ids) - 1
                if rollout_logprobs is not None:
                    if response_logprobs is None:
                        rollout_logprobs = None
                    else:
                        rollout_logprobs += response_logprobs
                per_step_rewards.append((step_reward, response_end_idx))
                chat_history.append(dict(assistant_message))
                if not done:
                    chat_history.extend(new_obs)
            elif retokenize_chat_history:
                # a. We always re-tokenize the entire chat history every turn and at the end.
                chat_history, chat_end_index, input_ids = self._get_next_input_ids_by_retokenizing_chat_history(
                    chat_history, chat_end_index, output, new_obs
                )
                # Re-tokenizing text can change token boundaries, so engine logprobs no
                # longer have an exact position in the returned response.
                rollout_logprobs = None
                # TODO(tgriggs): Support turn-level rewards for multi-turn chat template
                per_step_rewards.append((step_reward, None))
            elif self.use_conversation_multi_turn:
                # b. Token-in-token-out. Follow multi-turn chat history format.
                input_ids, loss_mask, rollout_logprobs, response_end_idx = (
                    self._get_next_input_ids_with_multiturn_chat_template(
                        input_ids,
                        loss_mask,
                        rollout_logprobs,
                        output_ids,
                        response_logprobs,
                        new_obs,
                        done,
                        added_eos,
                    )
                )
                per_step_rewards.append((step_reward, response_end_idx))
            else:
                # c. Token-in-token-out. All steps/observations are appended to a single assistant message.
                loss_mask, input_ids, rollout_logprobs, response_end_idx = (
                    self._get_next_input_ids_with_single_turn_chat_template(
                        output_ids,
                        response_logprobs,
                        new_obs,
                        loss_mask,
                        input_ids,
                        rollout_logprobs,
                        done,
                    )
                )
                per_step_rewards.append((step_reward, response_end_idx))

        # Get environment-specific metrics after the episode is done
        env_metrics = environment_metrics_from_step(env_step_output, env.get_metrics())
        # Close the environment
        await self._run_in_executor_if_available(env.close)

        prompt_ids = input_ids[:initial_prompt_length]
        if retokenize_chat_history:
            response_encodings = self.tokenizer.apply_chat_template(
                chat_history[initial_chat_history_length : len(chat_history) - len(new_obs)],
                chat_template=self.custom_chat_template,
                add_generation_prompt=False,
                return_dict=True,
                return_assistant_tokens_mask=True,
                tokenize=True,
                **self.trajectory_runner_cfg.chat_template_kwargs,
            )
            loss_mask = response_encodings["assistant_masks"]
            response_ids = response_encodings["input_ids"]
        else:
            assert not any(loss_mask[response_end_idx - initial_prompt_length + 1 :]), (
                "loss_mask at index after response end should be all 0"
            )
            loss_mask = loss_mask[: response_end_idx - initial_prompt_length + 1]
            response_ids = input_ids[initial_prompt_length : response_end_idx + 1]
            if rollout_logprobs is not None:
                rollout_logprobs = rollout_logprobs[: len(response_ids)]
            per_step_rewards = [
                (reward, None if idx is None else idx - initial_prompt_length) for reward, idx in per_step_rewards
            ]
        assert len(loss_mask) == len(response_ids), "loss_mask and response_ids should have the same length"

        if not self.use_conversation_multi_turn:
            if stop_reason != "length" and response_ids and response_ids[-1] != self.tokenizer.eos_token_id:
                response_ids.append(self.tokenizer.eos_token_id)
                # This EOS is an assembly delimiter, not a sampled action. Keep it out of
                # behavior-referenced objectives and place the final reward on the last
                # genuinely generated token below.
                loss_mask.append(0)
                if rollout_logprobs is not None:
                    rollout_logprobs.append(0.0)

        assert rollout_logprobs is None or len(rollout_logprobs) == len(response_ids), (
            "rollout_logprobs and response_ids should have the same length"
        )

        # Build reward output
        if retokenize_chat_history:
            # TODO(Charlie): Currently, the possible response truncation will not affect the reward
            # in the if branch, but some final rewards may be lost in the else branch. Fix this
            # when we support turn-level rewards for the `retokenize_chat_history` codepath.
            optimization_reward = float(per_step_rewards[-1][0])
            token_rewards = None
        else:
            token_level_rewards = self._place_step_rewards(response_ids, loss_mask, per_step_rewards)
            optimization_reward = float(sum(token_level_rewards))
            token_rewards = tuple(token_level_rewards)

        verification, unshaped_reward = fold_verification_results(verification_results)

        evidence = RolloutEvidence(
            messages=tuple(chat_history) if retokenize_chat_history or chat_completion_params is not None else (),
            response=output,
            stop_reason=stop_reason,
            generated_token_count=sum(bool(value) for value in loss_mask),
            prompt_token_ids=tuple(prompt_ids),
            response_token_ids=tuple(response_ids),
            behavior_logprobs=None if rollout_logprobs is None else tuple(rollout_logprobs),
        )
        reward_result = RewardResult(
            unshaped_reward=unshaped_reward,
            optimization_reward=optimization_reward,
            token_rewards=token_rewards,
        )
        reward_result.validate_for(evidence)
        return AgentLoopOutput(
            evidence=evidence,
            verification=verification,
            reward=reward_result,
            disposition=TrainingDisposition.train(),
            loss_mask=loss_mask,
            env_metrics=env_metrics,
            captured_global_step=captured_global_step,
            token_provenance=token_provenance,
        )

    async def collect_batched(
        self,
        prompts: List[ConversationType],
        env_classes: List[str],
        env_extras: List[Dict[str, Any]],
        max_tokens: int,
        sampling_params: Optional[Dict[str, Any]] = None,
    ) -> TrajectoryBatch:
        """
        Single-turn batched generation (can use the synchronous offline engine)

        Args:
            prompts: List[ConversationType]
            env_classes: List[str]
            env_extras: List[Dict[str, Any]]
            max_tokens: int
            sampling_params: Optional[Dict[str, Any]]
        Returns:
            TrajectoryBatch
        """
        envs = []
        init_prompts = []
        for env_class, env_extra, prompt in zip(env_classes, env_extras, prompts):
            env_extra["max_turns"] = self.max_turns
            env_config = self.skyrl_gym_cfg.get(env_class, DictConfig({}))
            env = skyrl_gym.make(env_class, env_config=env_config, extras=env_extra)
            init_prompt, _ = await self._run_in_executor_if_available(env.init, prompt)
            init_prompts.append(init_prompt)
            envs.append(env)

        # For single-turn generation, we can use text-in-token-out, since we do not need to re-tokenize.
        engine_input = InferenceEngineInput(prompts=init_prompts, sampling_params=sampling_params)
        engine_output = await self.model_client.generate(engine_input)
        outputs = engine_output["responses"]
        responses = engine_output["response_ids"]
        stop_reasons = engine_output["stop_reasons"]
        logprobs = engine_output.get("response_logprobs", None)

        truncated_responses = []
        rewards = []
        unshaped_rewards = []
        successes = []
        exclude_from_baseline = []
        loss_masks = []
        env_metrics = []
        truncated_logprobs: Optional[List[List[float]]] = [] if logprobs is not None else None

        for i, (output, response, env, env_class) in enumerate(zip(outputs, responses, envs, env_classes)):
            publish_rollout_evidence(
                env,
                messages=init_prompts[i],
                response=output,
                stop_reason=stop_reasons[i],
                response_token_ids=response,
                behavior_logprobs=None if logprobs is None else logprobs[i],
                metadata={"generation_token_budget": max_tokens},
            )
            # step on environment and compute reward
            env_step_output: BaseTextEnvStepOutput = await self._run_in_executor_if_available(env.step, output)
            verification = verification_from_env_step(env_step_output)

            if len(response) > max_tokens:
                response = response[:max_tokens]
            loss_masks.append([1] * len(response))
            truncated_responses.append(response)
            if logprobs is not None:
                sample_logprobs = logprobs[i][: len(response)]
                truncated_logprobs.append(sample_logprobs)

            evidence = RolloutEvidence(
                messages=tuple(init_prompts[i]),
                response=output,
                stop_reason=stop_reasons[i],
                generated_token_count=len(response),
                response_token_ids=tuple(response),
                behavior_logprobs=None if logprobs is None else tuple(truncated_logprobs[-1]),
            )
            reward_result = reward_from_env_step(env_step_output, verification)
            reward_result.validate_for(evidence)
            disposition = TrainingDisposition.train()
            rewards.append(reward_result.optimization_reward)
            unshaped_rewards.append(reward_result.unshaped_reward)
            successes.append(
                verification.passed
                if verification.passed is not None
                else verification.score is not None and verification.score > 0.0
            )
            exclude_from_baseline.append(not disposition.baseline_eligible)

            # Get environment-specific metrics
            env_metrics.append(environment_metrics_from_step(env_step_output, env.get_metrics()))
            # Close the environment
            await self._run_in_executor_if_available(env.close)

        # init_prompts is a BATCH (list of conversations), so this returns per-sample
        # rows (list[list[int]]). On transformers 5.x a bare tokenize=True yields a
        # BatchEncoding (mapping) rather than the list rows; extract input_ids in that
        # case. normalize_token_ids is NOT used here — its singleton-unwrap would
        # corrupt a one-element batch — and we key off the mapping interface (not
        # return_dict) so a tokenizer/mock that already returns list rows is unchanged.
        prompt_encodings = self.tokenizer.apply_chat_template(
            init_prompts,
            add_generation_prompt=True,
            tokenize=True,
        )
        prompt_token_ids = prompt_encodings["input_ids"] if hasattr(prompt_encodings, "keys") else prompt_encodings
        rollout_metrics = get_rollout_metrics(
            responses,
            rewards,
            env_metrics,
            env_classes,
            successes=successes,
        )

        if self.trajectory_runner_cfg.apply_overlong_filtering:
            loss_masks = apply_overlong_filtering(loss_masks, responses, self.tokenizer.eos_token_id)

        trajectory_batch: TrajectoryBatch = {
            "prompt_token_ids": prompt_token_ids,
            "response_ids": truncated_responses,
            "rewards": rewards,
            "loss_masks": loss_masks,
            "stop_reasons": stop_reasons,
            "rollout_metrics": rollout_metrics,
            "rollout_logprobs": truncated_logprobs,
            "exclude_from_baseline": exclude_from_baseline,
        }
        attach_unshaped_rewards(trajectory_batch, unshaped_rewards)

        return trajectory_batch

    async def _run(self, input_batch: TrajectoryRequestBatch, disable_tqdm: bool = False) -> TrajectoryBatch:
        """Run the configured environment loop and project its interaction records."""
        outputs = await self.collector.collect(input_batch, disable_tqdm=disable_tqdm)
        if isinstance(outputs, list) and outputs and isinstance(outputs[0], AgentLoopOutput):
            await self._apply_genrm_cohort_rewards(outputs, input_batch)
        return self.projection.project(outputs, input_batch)

    async def _apply_genrm_cohort_rewards(
        self,
        outputs: list[AgentLoopOutput],
        input_batch: TrajectoryRequestBatch,
    ) -> None:
        env_extras = input_batch.get("env_extras") or []
        genrm_agents = {"genrm_simple_agent", "genrm_simple_agent_reasoning_off"}

        def ultra_at(index: int) -> dict[str, Any] | None:
            extra_info = env_extras[index].get("extra_info") if index < len(env_extras) else None
            ultra = extra_info.get("nemotron_ultra") if isinstance(extra_info, dict) else None
            return ultra if isinstance(ultra, dict) else None

        genrm_indices = [index for index in range(len(outputs)) if (ultra_at(index) or {}).get("agent") in genrm_agents]
        if not genrm_indices:
            return
        batch_metadata = input_batch.get("batch_metadata")
        if batch_metadata is not None and batch_metadata.training_phase == "eval":
            for index in genrm_indices:
                outputs[index].env_metrics["genrm/cohort_skipped_eval"] = 1.0
            return
        if self.genrm_judge is None:
            raise RuntimeError("Nemotron Ultra GenRM rows require environment.skyrl_gym.nemotron_ultra.genrm.judge")
        trajectory_ids = input_batch.get("trajectory_ids")
        if trajectory_ids is None:
            raise ValueError("GenRM cohort rewards require trajectory IDs")

        groups: dict[str, list[int]] = {}
        for index in genrm_indices:
            groups.setdefault(trajectory_ids[index].instance_id, []).append(index)
        expected_size = int(self.genrm_config.get("num_rollouts_per_prompt", 16))
        for indices in groups.values():
            if len(indices) != expected_size:
                raise ValueError(
                    f"GenRM cohort requires {expected_size} rollouts for a prompt, received {len(indices)}"
                )
            records = [json.loads((ultra_at(index) or {})["record_json"]) for index in indices]
            if not all(isinstance(record, dict) for record in records):
                raise TypeError("GenRM record_json must decode to an object")
            principles = {record.get("principle") for record in records}
            if len(principles) != 1 or None in principles:
                raise ValueError("GenRM cohort rows must agree on a non-empty principle")
            response_objects = []
            for index in indices:
                messages = outputs[index].evidence.messages
                assistant_message = next(
                    (dict(message) for message in reversed(messages) if message.get("role") == "assistant"),
                    {},
                )
                response_objects.append(response_object(assistant_message, outputs[index].evidence.response or ""))
            rewards, metrics = await asyncio.to_thread(
                grade_genrm_group,
                conversation_history=input_batch["prompts"][indices[0]],
                response_objects=response_objects,
                principle=next(iter(principles)),
                judge=self.genrm_judge,
                config=self.genrm_config,
            )
            for index, reward in zip(indices, rewards, strict=True):
                old_token_rewards = outputs[index].reward.token_rewards
                token_rewards = None
                if old_token_rewards is not None:
                    token_rewards_list = [0.0] * len(old_token_rewards)
                    credited = [position for position, value in enumerate(old_token_rewards) if value]
                    if credited:
                        token_rewards_list[credited[-1]] = reward
                    token_rewards = tuple(token_rewards_list)
                outputs[index].reward = RewardResult(
                    unshaped_reward=reward,
                    optimization_reward=reward,
                    token_rewards=token_rewards,
                )
                outputs[index].verification = VerificationResult.verified(
                    reward,
                    diagnostics={"agent": (ultra_at(index) or {})["agent"], "genrm_metrics": metrics},
                )
                outputs[index].env_metrics.update({f"genrm/{name}": value for name, value in metrics.items()})

    @staticmethod
    def _mask_canonicalized_chat_prefix(
        input_ids: List[int],
        rendered_prompt_ids: List[int],
        initial_prompt_length: int,
        rollout_logprobs: Optional[List[float]],
        per_step_rewards: List[Tuple[float, Optional[int]]],
    ) -> _CanonicalizedChatPrefix:
        """Mask reconstructed context and mark its reward positions for later placement."""
        if input_ids[:initial_prompt_length] != rendered_prompt_ids[:initial_prompt_length]:
            raise RuntimeError("vLLM changed the tokenization of the original prompt while rendering a later chat turn")

        reconstructed_token_count = len(rendered_prompt_ids) - initial_prompt_length
        loss_mask = [0] * reconstructed_token_count
        if rollout_logprobs is not None:
            rollout_logprobs = [0.0] * reconstructed_token_count
        per_step_rewards = [(reward, None) for reward, _ in per_step_rewards]
        return _CanonicalizedChatPrefix(loss_mask, rollout_logprobs, per_step_rewards)

    @staticmethod
    def _place_step_rewards(
        response_ids: List[int],
        loss_mask: List[int],
        per_step_rewards: List[Tuple[float, Optional[int]]],
    ) -> List[float]:
        """Credit turn rewards at their boundaries, or at the last exact action after reconstruction."""
        token_level_rewards = [0.0] * len(response_ids)
        if not response_ids:
            return token_level_rewards
        fallback_index = next(
            (index for index in range(len(loss_mask) - 1, -1, -1) if loss_mask[index]),
            len(response_ids) - 1,
        )
        for step_reward, index in per_step_rewards:
            reward_index = fallback_index if index is None else index
            if reward_index >= len(response_ids):
                break
            token_level_rewards[reward_index] += step_reward
        return token_level_rewards

    # ----------------------------------------------------------------------------
    # Three methods of managing chat history and input ids in `agent_loop()`
    # ----------------------------------------------------------------------------

    def _get_next_input_ids_by_retokenizing_chat_history(
        self,
        chat_history: ConversationType,
        chat_end_index: int,
        output: str,
        new_obs: ConversationType,
    ):
        """
        Update the chat history and input ids given a new model response and observation by retokenizing
        the entire chat history. Hence token-in-token-out is not followed.

        loss_mask is not maintained because we get it at the end of the trajectory with
        `response_encodings["assistant_masks"]`.

        Returns:
            chat_history: The updated chat history.
            chat_end_index: The updated chat end index.
            input_ids: The new input IDs after tokenizing the chat history.
        """
        assert self.use_conversation_multi_turn and self.custom_chat_template
        # remove eos token from end of output if it exists, since it will be reapplied by the chat template
        if output.endswith(self.tokenizer.eos_token):
            output = output[: -len(self.tokenizer.eos_token)]

        # Add assistant response to chat history
        chat_history += [{"role": "assistant", "content": output}]
        chat_end_index += 1

        # Add observations to chat history
        if len(new_obs) > 0:
            chat_history += new_obs
            chat_end_index += len(new_obs)

        # re-apply whole chat template so length check is correct
        input_ids = normalize_token_ids(
            self.tokenizer.apply_chat_template(
                chat_history[:chat_end_index],
                chat_template=self.custom_chat_template,
                add_generation_prompt=False,
                tokenize=True,
                **self.trajectory_runner_cfg.chat_template_kwargs,
            )
        )
        return chat_history, chat_end_index, input_ids

    def _get_next_input_ids_with_multiturn_chat_template(
        self,
        input_ids: List[int],
        loss_mask: List[int],
        rollout_logprobs: Optional[List[float]],
        output_ids: List[int],
        output_logprobs: Optional[List[float]],
        new_obs: ConversationType,
        done: bool,
        added_eos: bool,
    ):
        """
        Update the loss mask and input ids given a new model response and observation, following
        token-in-token-out.

        This function is used if `use_conversation_multi_turn` is True. It assumes that the input to the LLM is formatted as a list of messages, with observations
        stored in user messages.

        For example (using the Qwen 2.5 chat template), a trajectory for multi-turn generation would look like:
        <|im_start|>system
        ...
        <|im_end|>
        <|im_start|>user
                            question goes here
        <|im_end|>
        <|im_start|>assistant
                            turn 1 model response goes here
                            <think>... </think>
                            ...
        <|im_end|>
        <|im_start|>user
                            turn 1 env observation goes here
                            <observation>...</observation>
        <|im_end|>
        ...

        the chat template is applied without tokenization before and after the chat history is appended to
        in order to get new token ids in the chat template format (but without re-tokenizing the entire chat history every turn)

        Args:
            chat_history: ConversationType
            chat_end_index: int
            loss_mask: List[int]
            input_ids: List[int]
            output: str
            new_obs: ConversationType
        Returns:
            chat_history: ConversationType
            chat_end_index: int
            loss_mask: List[int]
            input_ids: List[int]
        """
        assert self.use_conversation_multi_turn and not self.custom_chat_template

        # 1. Directly append generated output
        input_ids += output_ids
        response_end_idx = len(input_ids) - 1
        # if `added_eos` is `True`, then  the EOS token was not generated and only added in the
        # `agent_loop` function. For consistency with other entities like logprobs , we ignore it in the loss
        # mask
        loss_mask += [1] * len(output_ids) if not added_eos else [1] * (len(output_ids) - 1) + [0]
        if rollout_logprobs is not None:
            if output_logprobs is None:
                rollout_logprobs = None
            else:
                rollout_logprobs += output_logprobs

        # 2. apply chat template for observations, also generate generation prompt for next turn
        if len(new_obs) > 0:
            # For Qwen, this will generate `\n<|user|>Some observation<|im_end|>\n`. Note that the
            # first `\n` is generated since we stripped it in ``base_conversation_token_ids``.
            observation_ids = normalize_token_ids(
                self.tokenizer.apply_chat_template(
                    [*self.base_conversation, *new_obs],
                    add_generation_prompt=not done,
                    tokenize=True,
                    **self.trajectory_runner_cfg.chat_template_kwargs,
                )
            )[len(self.base_conversation_token_ids) :]
            input_ids += observation_ids
            loss_mask += [0] * len(observation_ids)
            if rollout_logprobs is not None:
                rollout_logprobs += [0.0] * len(observation_ids)
        else:
            if not done:
                input_ids += self.generation_prompt_ids
                loss_mask += [0] * len(self.generation_prompt_ids)
                if rollout_logprobs is not None:
                    rollout_logprobs += [0.0] * len(self.generation_prompt_ids)

        return input_ids, loss_mask, rollout_logprobs, response_end_idx

    def _get_next_input_ids_with_single_turn_chat_template(
        self,
        output_ids: List[int],
        output_logprobs: Optional[List[float]],
        new_obs: ConversationType,
        loss_mask: List[int],
        input_ids: List[int],
        logprobs: Optional[List[float]],
        done: bool,
    ):
        """
        Update the loss mask and input ids given a new model response and observation, following
        token-in-token-out.

        This function is used if `use_conversation_multi_turn` is False. It assumes that the input to the LLM is a list of token ids
        and that the multi-turn conversation happens in a single assistant message.

        For example (using the Qwen 2.5 chat template), a trajectory for single-turn generation would look like:
        <|im_start|>system
        ...
        <|im_end|>
        <|im_start|>user
                            question goes here
        <|im_end|>
        <|im_start|>assistant
                            turn 1 model response goes here
                            <think>... </think>
                            ...

                            turn 1 env observation goes here
                            <observation>...</observation>

                            turn 2 model response goes here:
                            <think>... </think>
                            ...
        Args:
            output_ids: List[int]
            new_obs: ConversationType
            loss_mask: List[int]
            input_ids: List[int]
        Returns:
            loss_mask: List[int]
            input_ids: List[int]
            logprobs: Optional[List[float]]
        """
        # just update raw tokens and loss mask
        new_resp_tokens = output_ids.copy()
        if not done and new_resp_tokens and new_resp_tokens[-1] == self.tokenizer.eos_token_id:
            # remove the eos token since we are continuing the current assistant message
            new_resp_tokens = new_resp_tokens[:-1]
        loss_mask += [1] * len(new_resp_tokens)
        input_ids += new_resp_tokens
        response_end_idx = len(input_ids) - 1
        if logprobs is not None:
            if output_logprobs is None:
                logprobs = None
            else:
                logprobs += output_logprobs[: len(new_resp_tokens)]

        if len(new_obs) > 0:
            for obs in new_obs:
                obs_tokens = self.tokenizer.encode(obs["content"], add_special_tokens=False)
                loss_mask += [0] * len(obs_tokens)
                # logprobs for observation tokens doesn't matter since they will be masked out during loss computation
                if logprobs is not None:
                    logprobs += [0.0] * len(obs_tokens)
                input_ids += obs_tokens

        return loss_mask, input_ids, logprobs, response_end_idx
