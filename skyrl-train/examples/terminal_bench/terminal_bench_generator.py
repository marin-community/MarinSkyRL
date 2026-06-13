import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, List, Optional, Dict, Any, Set, Tuple
from loguru import logger
from uuid import uuid4
from skyrl_train.generators.base import GeneratorInterface, GeneratorInput, GeneratorOutput, TrajectoryID
from skyrl_train.generators.utils import (
    get_rollout_metrics,
    get_response_ids_and_loss_mask_from_messages,
    extract_logprobs_from_rollout_details,
    extract_token_ids_from_rollout_details,
    extract_routed_experts_from_rollout_details,
    normalize_token_ids,
    AlignmentStats,
    _sentinel_routed_experts_row,
    SENTINEL_EXPERT_ID,
)
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.base import ConversationType
from skyrl_train.utils.reward_shaping import shape_reward_from_output, shape_reward_with_components
from omegaconf import DictConfig
from pathlib import Path

# Harbor orchestrator and trial imports.
# QueueOrchestrator + OrchestratorEvent come through a compat shim because
# Harbor removed ``harbor.orchestrators`` in favor of ``harbor.trial.queue``
# and ``harbor.trial.hooks``. See _harbor_compat.py for the wrapper.
from examples.terminal_bench._harbor_compat import (
    OrchestratorEvent,
    QueueOrchestrator,
    create_rollback_hook,
)
from harbor.models.trial.config import TrialConfig
from harbor.models.trial.result import TrialResult

# Schema-driven Harbor config mapping
from examples.terminal_bench.harbor_config import HarborConfigBuilder

# Maximum restart attempts for orchestrator recovery
MAX_ORCHESTRATOR_RESTART_ATTEMPTS = 3


# Backward-compatible alias. The coercion now lives in
# ``skyrl_train.generators.utils.normalize_token_ids`` so it can be shared with
# the response-side ``len()``-slicing sites (``get_generation_prompt_ids`` /
# ``encode_messages_subset``) that hit the same Qwen3-Next-80B BatchEncoding leak.
# Kept here (delegating) so the existing prompt-side call site and tests are
# unchanged.
_normalize_prompt_token_ids = normalize_token_ids


@dataclass
class TerminalBenchAgentOutput:
    response_ids: List[int]
    reward: float
    stop_reason: str
    loss_mask: List[int]
    prompt_ids: List[int]
    trajectory_id: TrajectoryID
    summarization_count: Optional[int] = None
    rollout_logprobs: Optional[List[float]] = None
    # MoE router-replay (Stage 1 capture rail): per-token [L, K] expert-selection
    # rows aligned 1:1 with response_ids. None unless moe_router_replay is on.
    rollout_routed_experts: Optional[List[List[List[int]]]] = None
    # For RLOO-N: True = exclude from baseline (infrastructure failure)
    # False = include in baseline (agent failure or success)
    exclude_from_baseline: bool = False
    # Store the exception type for debugging/logging
    exception_type: Optional[str] = None
    # Per-component reward breakdown (composite shaper only)
    reward_components: Optional[Dict[str, float]] = None
    # TIS logprob-alignment bookkeeping (exact-vs-LCS-vs-failed token counts).
    # Aggregated into rollout_metrics as tis/* so an LCS fallback or alignment
    # failure can never silently degrade TIS. None when no logprobs were present.
    alignment_stats: Optional[AlignmentStats] = None


class TerminalBenchGenerator(GeneratorInterface):
    def __init__(
        self,
        generator_cfg: DictConfig,
        terminal_bench_cfg: DictConfig,
        inference_engine_client: InferenceEngineClient,
        tokenizer,
        moe_router_replay: bool = False,
    ):
        """
        Args:
            generator_cfg: DictConfig object containing the generator configuration
            terminal_bench_cfg: DictConfig object containing the terminal bench configuration
            inference_engine_client: InferenceEngineClient object for interacting with the inference engines
            tokenizer: tokenizer object for encoding and decoding text
            moe_router_replay: when True, capture per-token MoE routed_experts from
                Harbor rollout_details and plumb them through to the training batch
                (Stage 1 of the FSDP2 EP/router-replay port). Default False keeps the
                GeneratorOutput byte-identical to today.
        """
        self.base_url = f"http://{generator_cfg.http_endpoint_host}:{generator_cfg.http_endpoint_port}"
        self.generator_cfg = generator_cfg
        self.tokenizer = tokenizer
        self.model_name = generator_cfg.model_name
        self._moe_router_replay = moe_router_replay

        # Core terminal bench config
        self.trials_dir = terminal_bench_cfg.trials_dir

        # Schema-driven Harbor config builder
        # Automatically maps YAML fields to Harbor's TrialConfig with validation
        self._harbor_config_builder = HarborConfigBuilder(terminal_bench_cfg)

        # Configure Harbor log level (default WARNING to reduce noise)
        harbor_log_level = self._harbor_config_builder.get_log_level(default="WARNING")
        self._configure_harbor_logging(harbor_log_level)

        # Store model_info for external access (e.g., metrics)
        self.model_info = self._harbor_config_builder.model_info

        # Build retry config for QueueOrchestrator (handles backoff, exception filtering)
        self._retry_config = self._harbor_config_builder.build_retry_config()
        self._n_concurrent_trials = self._harbor_config_builder.get_n_concurrent_trials(
            default=16  # Reasonable default for parallel trial execution
        )

        # Reward shaping config (parses test output for partial credit)
        self._reward_shaping_config = self._harbor_config_builder.get_reward_shaping_config()

        # Error handling config (for RLOO-N advantage estimator)
        self._error_handling_config = self._harbor_config_builder.get_error_handling_config()

        # TIS (Truncated Importance Sampling) config
        # Only show TIS-related warnings when collect_rollout_details is enabled
        self._collect_rollout_details = self._harbor_config_builder.get_collect_rollout_details()

        # Tracked exception types for per-step error counters.
        # Sourced from the retry config's exclude_exceptions (the terminal failures
        # that Harbor will NOT retry).  These are always emitted as generate/ metrics
        # so that dashboards see a consistent zero-baseline time-series.
        self._tracked_exceptions = self._harbor_config_builder.get_exclude_exceptions()

        logger.info(
            f"TerminalBenchGenerator initialized with HarborConfigBuilder. "
            f"Exposed fields: {list(self._harbor_config_builder._harbor_cfg.keys())}. "
            f"Retry config: max_retries={self._retry_config.max_retries}, "
            f"backoff={self._retry_config.min_wait_sec}-{self._retry_config.max_wait_sec}s. "
            f"Concurrent trials: {self._n_concurrent_trials}. "
            f"Reward shaping: enabled={self._reward_shaping_config.get('enable_reward_shaping', True)}, "
            f"shaper={self._reward_shaping_config.get('reward_shaper', 'pass_ratio')}. "
            f"Error classification: enabled={self._error_handling_config.get('enable_error_classification', False)}"
        )

        # Read custom chat template
        custom_chat_template_path = generator_cfg.engine_init_kwargs.get(
            "custom_chat_template_chat_completion_path", None
        )
        if custom_chat_template_path:
            with open(custom_chat_template_path, "r") as f:
                self.custom_chat_template_content = f.read()
            logger.info(
                f"TerminalBenchGenerator initialized with custom chat template read from: {custom_chat_template_path}"
            )
        else:
            self.custom_chat_template_content = None

        # Shared QueueOrchestrator state (initialized in startup())
        # This ensures all concurrent generate() calls share a single orchestrator
        # with a global n_concurrent_trials limit, rather than each worker creating
        # its own orchestrator (which would multiply the concurrency limit)
        self._orchestrator: Optional[QueueOrchestrator] = None
        self._orchestrator_lock: Optional[asyncio.Lock] = None  # Protects orchestrator lifecycle
        self._orchestrator_started: bool = False
        self._orchestrator_restart_count: int = 0

        # Eval session state (separate orchestrator for eval runs)
        # Each eval run gets a fresh orchestrator that is destroyed after the eval completes
        self._eval_orchestrator: Optional[QueueOrchestrator] = None
        self._eval_orchestrator_lock: Optional[asyncio.Lock] = None
        self._eval_session_active: bool = False
        self._eval_session_name: Optional[str] = None
        self._eval_trials_dir: Optional[str] = None

        # Eval-specific timeout (default 900s = 15 minutes)
        self._eval_timeout_override_sec = self._harbor_config_builder.get_eval_timeout_override_sec(default=900)

        # Staleness tracking — captures the global_step that was current at each
        # trial's Harbor pickup time (= the moment SkyRL/Harbor first attempted to
        # dispatch the trial to vLLM). Set externally by FullyAsyncRayPPOTrainer.
        self.global_step_fn: Optional[Callable[[], int]] = None
        # Rolling (global_step, time.time()) history; we look up `started_at` for
        # each trial against this history to estimate `actual_global_step`. Bounded
        # to keep memory flat; long-tail trials past the window fall back to the
        # earliest retained step (conservative — biases staleness slightly higher).
        self._step_time_history: Deque[Tuple[int, float]] = deque(maxlen=512)

    def _record_step_time(self) -> None:
        """Append (global_step, now) to the step-time history if the step has advanced."""
        if self.global_step_fn is None:
            return
        try:
            step = self.global_step_fn()
        except Exception:
            return
        now = time.time()
        if not self._step_time_history or self._step_time_history[-1][0] != step:
            self._step_time_history.append((step, now))

    def _step_at_time(self, t: float) -> Optional[int]:
        """Return the global_step that was active at wall-clock time `t` per history."""
        if not self._step_time_history:
            return None
        result = self._step_time_history[0][0]
        for step, ts in self._step_time_history:
            if ts <= t:
                result = step
            else:
                break
        return result

    def _configure_harbor_logging(self, level: str) -> None:
        """
        Configure Harbor's logging level.

        Args:
            level: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        log_level = getattr(logging, level.upper(), logging.WARNING)

        # Set level for Harbor's main logger and all child loggers
        harbor_loggers = [
            "harbor",
            "harbor.trial",
            "harbor.agents",
            "harbor.verifier",
            "harbor.orchestrators",
            "harbor.environments",
            "harbor.utils.logger",
        ]

        for logger_name in harbor_loggers:
            logging.getLogger(logger_name).setLevel(log_level)

        # Also set the root harbor logger
        logging.getLogger("harbor").setLevel(log_level)

        logger.info(f"Harbor logging level set to {level}")

    async def startup(self) -> None:
        """Initialize shared QueueOrchestrator for all generate() calls.

        This creates a single orchestrator that enforces the n_concurrent_trials
        limit globally across all async workers. Without this, each generate()
        call would create its own orchestrator, multiplying the concurrency limit.

        Called once by the trainer before the first generate() call.
        """
        self._orchestrator_lock = asyncio.Lock()
        await self._create_orchestrator()
        logger.info(
            f"TerminalBenchGenerator startup complete. "
            f"Shared orchestrator ready with n_concurrent_trials={self._n_concurrent_trials}"
        )

    async def _create_orchestrator(self) -> None:
        """Create and start a new QueueOrchestrator.

        Used for initial startup and for recovery after orchestrator failures.
        """
        self._orchestrator = QueueOrchestrator(
            trial_configs=[],  # We submit dynamically via submit_batch()
            n_concurrent_trials=self._n_concurrent_trials,
            metrics={},  # SkyRL handles its own metrics
            quiet=True,
            retry_config=self._retry_config,
        )

        # Register rollback hook to ensure conversation consistency on exceptions.
        # When a trial fails with one of the gated exception types, this hook
        # rolls back rollout_details to the last complete turn, ensuring that
        # any turn with a prompt has a matching response (and logprobs if
        # collected). v2 broaden (2026-05-25): added `ValueError` because the
        # vLLM serving_chat path raises ValueError for 32k-token validation
        # failures (request exceeds max_input_tokens). Those used to be
        # bypassing the rollback path and leaving dangling Ray ObjectRefs,
        # which the v6a / v3 maxgn09 post-mortem identified as the dominant
        # trigger of the Ray ref_count race that's still killing chain links
        # even after the original ContextLengthExceededError / AgentTimeoutError
        # patch landed (commit f8205b1).
        rollback_hook = create_rollback_hook(
            exception_types={
                "ContextLengthExceededError",
                "AgentTimeoutError",
                "ValueError",
            },
            on_complete_failure="mark_metadata",
            preserve_partial_logprobs=False,
        )
        self._orchestrator.add_hook(OrchestratorEvent.TRIAL_COMPLETED, rollback_hook)

        await self._orchestrator.start()
        self._orchestrator_started = True
        logger.info(
            f"QueueOrchestrator created and started with "
            f"n_concurrent_trials={self._n_concurrent_trials}, "
            f"rollback_hook registered for ContextLengthExceededError/AgentTimeoutError"
        )

    async def _restart_orchestrator(self) -> bool:
        """Restart the orchestrator after a failure.

        Uses locking to ensure only one restart happens at a time, even with
        concurrent generate() calls. Other callers wait for the restart to complete.

        Returns:
            True if restart succeeded, False if max attempts exceeded.
        """
        async with self._orchestrator_lock:
            # Check if another caller already restarted while we were waiting
            if self._orchestrator_started and self._orchestrator is not None:
                logger.info("Orchestrator already restarted by another caller")
                return True

            self._orchestrator_restart_count += 1
            if self._orchestrator_restart_count > MAX_ORCHESTRATOR_RESTART_ATTEMPTS:
                logger.error(
                    f"Max orchestrator restart attempts ({MAX_ORCHESTRATOR_RESTART_ATTEMPTS}) " f"exceeded. Giving up."
                )
                return False

            logger.warning(
                f"Restarting QueueOrchestrator (attempt {self._orchestrator_restart_count}/"
                f"{MAX_ORCHESTRATOR_RESTART_ATTEMPTS})"
            )

            # Shutdown the failed orchestrator if it exists
            if self._orchestrator is not None:
                try:
                    await self._orchestrator.shutdown(wait=False)
                except Exception as e:
                    logger.warning(f"Error shutting down failed orchestrator: {e}")
                finally:
                    self._orchestrator = None
                    self._orchestrator_started = False

            # Create a new orchestrator
            try:
                await self._create_orchestrator()
                return True
            except Exception as e:
                logger.error(f"Failed to create new orchestrator: {e}")
                self._orchestrator_started = False
                return False

    async def shutdown(self) -> None:
        """Cleanup shared QueueOrchestrator.

        Called once by the trainer after the last generate() call.
        Safe to call multiple times (idempotent).
        """
        if self._orchestrator_lock is None:
            # startup() was never called
            return

        async with self._orchestrator_lock:
            if self._orchestrator is not None and self._orchestrator_started:
                # Mark as not started BEFORE shutdown to prevent race condition.
                # This ensures concurrent generate() calls fail fast rather than
                # trying to submit to an orchestrator that's in the process of
                # shutting down (which can take time with wait=True).
                self._orchestrator_started = False
                try:
                    logger.info("Shutting down shared QueueOrchestrator...")
                    await self._orchestrator.shutdown(wait=True)
                    logger.info("QueueOrchestrator shutdown complete")
                except Exception as e:
                    logger.warning(f"Error during orchestrator shutdown: {e}")
                finally:
                    self._orchestrator = None

    async def start_eval_session(
        self,
        run_name: str,
        eval_step: int,
        val_set_name: Optional[str] = None,
    ) -> None:
        """Start a fresh eval session with its own QueueOrchestrator.

        Each eval run gets a dedicated orchestrator that is destroyed after eval completes.
        This ensures eval trials don't interfere with training trials and provides
        clean isolation for metrics and artifacts.

        Args:
            run_name: The job run name (from cfg.trainer.run_name).
            eval_step: The current global step (for unique naming).
            val_set_name: Optional name of the validation set being evaluated.
        """
        if self._eval_orchestrator_lock is None:
            self._eval_orchestrator_lock = asyncio.Lock()

        async with self._eval_orchestrator_lock:
            # Ensure any previous eval session is cleaned up
            if self._eval_session_active and self._eval_orchestrator is not None:
                logger.warning("Previous eval session still active, shutting it down first")
                try:
                    await self._eval_orchestrator.shutdown(wait=True)
                except Exception as e:
                    logger.warning(f"Error shutting down previous eval orchestrator: {e}")
                finally:
                    self._eval_orchestrator = None
                    self._eval_session_active = False

            # Build unique session name
            val_set_suffix = f"_{val_set_name}" if val_set_name else ""
            self._eval_session_name = f"{run_name}_eval{val_set_suffix}_step{eval_step}"

            # Create unique trials directory for this eval session
            if self.trials_dir:
                self._eval_trials_dir = str(Path(self.trials_dir) / "eval_sessions" / self._eval_session_name)
                Path(self._eval_trials_dir).mkdir(parents=True, exist_ok=True)
            else:
                self._eval_trials_dir = self.trials_dir

            logger.info(
                f"Starting eval session: {self._eval_session_name} "
                f"(timeout={self._eval_timeout_override_sec}s, trials_dir={self._eval_trials_dir})"
            )

            # Create fresh orchestrator for eval with eval-specific timeout
            self._eval_orchestrator = QueueOrchestrator(
                trial_configs=[],  # We submit dynamically via submit_batch()
                n_concurrent_trials=self._n_concurrent_trials,
                metrics={},  # SkyRL handles its own metrics
                quiet=True,
                retry_config=self._retry_config,
            )

            # Register rollback hook for eval orchestrator (same as training).
            # See the training-side comment above re: ValueError addition.
            rollback_hook = create_rollback_hook(
                exception_types={
                    "ContextLengthExceededError",
                    "AgentTimeoutError",
                    "ValueError",
                },
                on_complete_failure="mark_metadata",
                preserve_partial_logprobs=False,
            )
            self._eval_orchestrator.add_hook(OrchestratorEvent.TRIAL_COMPLETED, rollback_hook)

            await self._eval_orchestrator.start()
            self._eval_session_active = True

            logger.info(
                f"Eval session {self._eval_session_name} started with fresh QueueOrchestrator "
                f"(n_concurrent_trials={self._n_concurrent_trials}, rollback_hook registered)"
            )

    async def stop_eval_session(self) -> None:
        """Stop the current eval session and destroy its orchestrator.

        Should be called after each evaluation run completes.
        Safe to call multiple times (idempotent).
        """
        if self._eval_orchestrator_lock is None:
            return

        async with self._eval_orchestrator_lock:
            if self._eval_orchestrator is not None and self._eval_session_active:
                session_name = self._eval_session_name or "unknown"
                try:
                    logger.info(f"Stopping eval session: {session_name}")
                    await self._eval_orchestrator.shutdown(wait=True)
                    logger.info(f"Eval session {session_name} orchestrator shutdown complete")
                except Exception as e:
                    logger.warning(f"Error during eval orchestrator shutdown: {e}")
                finally:
                    self._eval_orchestrator = None
                    self._eval_session_active = False
                    self._eval_session_name = None
                    self._eval_trials_dir = None

    def _get_active_orchestrator(self) -> Optional[QueueOrchestrator]:
        """Get the currently active orchestrator (eval or training).

        Returns:
            The eval orchestrator if an eval session is active, otherwise the training orchestrator.
        """
        if self._eval_session_active and self._eval_orchestrator is not None:
            return self._eval_orchestrator
        return self._orchestrator

    def _get_active_trials_dir(self) -> Optional[str]:
        """Get the trials directory for the currently active mode (eval or training).

        Returns:
            The eval trials directory if an eval session is active, otherwise the training trials_dir.
        """
        if self._eval_session_active and self._eval_trials_dir is not None:
            return self._eval_trials_dir
        return self.trials_dir

    def _get_active_timeout_override(self) -> Optional[int]:
        """Get the timeout override for the currently active mode (eval or training).

        Returns:
            The eval timeout if an eval session is active, otherwise None (use config default).
        """
        if self._eval_session_active:
            return self._eval_timeout_override_sec
        return None

    def _create_all_failed_output(
        self,
        trajectory_ids: List[TrajectoryID],
        exception_type: str = "OrchestratorFailure",
    ) -> GeneratorOutput:
        """Create a GeneratorOutput where all trajectories failed.

        Used when the orchestrator itself fails and cannot process any trials.
        All outputs are marked as infrastructure failures (excluded from baseline).
        """
        num_trials = len(trajectory_ids)
        return {
            "prompt_token_ids": [[0] for _ in range(num_trials)],
            "response_ids": [[0] for _ in range(num_trials)],
            "rewards": [0.0 for _ in range(num_trials)],
            "loss_masks": [[0] for _ in range(num_trials)],
            "stop_reasons": ["error" for _ in range(num_trials)],
            "rollout_metrics": {
                "generate/num_failed_instances": num_trials,
                "generate/num_failed_trajectories": num_trials,
                "generate/num_masked_trajectories": num_trials,
                f"generate/exception_{exception_type}": num_trials,
            },
            "rollout_logprobs": None,
            "exclude_from_baseline": [True for _ in range(num_trials)],  # Infrastructure failure
        }

    async def generate(self, input_batch: GeneratorInput) -> GeneratorOutput:
        """
        Generate rollouts for a batch of prompts using the active QueueOrchestrator.

        The active orchestrator (eval or training) handles:
        - Global concurrency control across all async workers
        - Retry logic with exponential backoff
        - Exception filtering (retry transient errors, skip permanent ones)

        During eval sessions (started via start_eval_session()), uses a dedicated
        eval orchestrator with its own trials directory and timeout settings.

        This method includes restart logic to recover from orchestrator failures
        without killing the entire training job.
        """
        # Record current global_step at the moment we enter generate(). Used as
        # both a history checkpoint and the conservative fallback for
        # actual_global_step if no trial reports a started_at.
        self._record_step_time()
        entry_global_step = self.global_step_fn() if self.global_step_fn is not None else None

        num_trials = len(input_batch["prompts"])
        is_eval = self._eval_session_active
        mode_str = f"eval ({self._eval_session_name})" if is_eval else "training"
        logger.info(f"Starting batch generation for {num_trials} trials (mode={mode_str})")

        # Get active trials directory and timeout override
        active_trials_dir = self._get_active_trials_dir()
        timeout_override = self._get_active_timeout_override()

        # Build all TrialConfigs upfront
        trial_configs: List[TrialConfig] = []
        trajectory_ids: List[TrajectoryID] = []

        # Harbor expects hosted_vllm model names with exactly one '/'.
        # Convert HuggingFace-style "org/model" to just "model" for the alias.
        model_alias = self.model_name.split("/")[-1] if "/" in self.model_name else self.model_name

        for i in range(num_trials):
            prompt = input_batch["prompts"][i]
            trajectory_id = input_batch["trajectory_ids"][i]

            # Generate session_id for sticky routing to inference engines
            session_id = uuid4().hex

            trial_config = self._harbor_config_builder.build_trial_config(
                task_path=prompt,
                trials_dir=active_trials_dir,
                model_name=f"hosted_vllm/{model_alias}",
                api_base=f"{self.base_url}/v1",
                session_id=session_id,
                timeout_override_sec=timeout_override,
            )
            trial_configs.append(trial_config)
            trajectory_ids.append(trajectory_id)

        # Get the active orchestrator (eval or training)
        # Note: We check without lock first for performance, then re-check with lock if needed
        active_orchestrator = self._get_active_orchestrator()
        orchestrator_started = self._eval_session_active if is_eval else self._orchestrator_started

        # Check if orchestrator is available
        # If it appears unavailable, acquire lock and re-check to avoid race condition
        # where multiple workers see temporary False state during restart
        if not orchestrator_started or active_orchestrator is None:
            async with self._orchestrator_lock:
                # Re-check after acquiring lock - another worker may have restarted
                active_orchestrator = self._get_active_orchestrator()
                orchestrator_started = self._eval_session_active if is_eval else self._orchestrator_started
                if orchestrator_started and active_orchestrator is not None:
                    # Another worker already restarted, continue with the fresh orchestrator
                    logger.debug("Orchestrator was restarted by another worker, continuing")
                else:
                    # Still unavailable, we need to restart
                    logger.warning(
                        "QueueOrchestrator not available. Was startup() called? " "Attempting emergency restart..."
                    )
            # Release lock before calling _restart_orchestrator (it acquires its own lock)
            if not orchestrator_started or active_orchestrator is None:
                restart_success = await self._restart_orchestrator()
                if not restart_success:
                    logger.error("Emergency orchestrator restart failed. Returning all-failed output.")
                    return self._create_all_failed_output(trajectory_ids, "OrchestratorNotStarted")
                # Refresh the orchestrator reference after restart
                active_orchestrator = self._get_active_orchestrator()

        # Submit trials to active orchestrator with restart logic on failure
        results: List[TrialResult | Exception] = []
        try:
            # Submit all trials and collect futures
            futures = await active_orchestrator.submit_batch(trial_configs)

            # Wait for all trials to complete
            # Note: return_exceptions=True ensures individual trial failures don't
            # bubble up as exceptions - they're returned as Exception objects in results
            results = await asyncio.gather(*futures, return_exceptions=True)

        except Exception as orchestrator_error:
            # Orchestrator-level failure (not individual trial failures)
            # This indicates something is wrong with the orchestrator itself
            logger.error(
                f"Orchestrator-level failure during batch submission/gather: "
                f"{type(orchestrator_error).__name__}: {orchestrator_error}"
            )

            # For eval sessions, we don't retry - just fail
            if is_eval:
                logger.error("Eval session orchestrator failed. Returning all-failed output.")
                return self._create_all_failed_output(
                    trajectory_ids, f"EvalOrchestratorFailure_{type(orchestrator_error).__name__}"
                )

            # Attempt to restart the training orchestrator
            restart_success = await self._restart_orchestrator()
            if not restart_success:
                logger.error(
                    "Orchestrator restart failed. Returning all-failed output " "to avoid killing training job."
                )
                return self._create_all_failed_output(
                    trajectory_ids, f"OrchestratorFailure_{type(orchestrator_error).__name__}"
                )

            # Retry once with the fresh orchestrator
            try:
                logger.info(f"Retrying batch of {num_trials} trials with restarted orchestrator")
                active_orchestrator = self._get_active_orchestrator()  # Refresh reference
                futures = await active_orchestrator.submit_batch(trial_configs)
                results = await asyncio.gather(*futures, return_exceptions=True)
            except Exception as retry_error:
                logger.error(
                    f"Retry after orchestrator restart also failed: " f"{type(retry_error).__name__}: {retry_error}"
                )
                return self._create_all_failed_output(
                    trajectory_ids, f"OrchestratorRetryFailure_{type(retry_error).__name__}"
                )

        # Process results into TerminalBenchAgentOutput
        all_outputs: List[TerminalBenchAgentOutput] = []
        for i, result in enumerate(results):
            trajectory_id = trajectory_ids[i]
            # Defense-in-depth: a SUCCESSFUL trial can still raise inside
            # _process_trial_result during the train-side tokenization/loss-mask
            # reconstruction — most notably a jinja2 TemplateError from
            # tokenizer.apply_chat_template (e.g. a chat template that rejects the
            # agentic multi-turn / consecutive-same-role / tool message structure).
            # That render error is a property of THIS ONE trajectory, not the job,
            # so it must be a per-trial skip, not a fatal crash. If it propagates
            # out of run_shard it ALSO triggers `_pickle.PicklingError: Can't
            # pickle RayTaskError(TemplateError)` when Ray serializes the nested
            # exception back to the driver — turning a skippable per-trial error
            # into a deterministic job kill. Catch it here, on the generator-worker
            # side, BEFORE it can cross the Ray boundary: classify it (mask by
            # default — it's an infrastructure/serialization-class failure,
            # excluded from the RLOO-N baseline) and emit an error output, exactly
            # like the orchestrator/exception paths above.
            try:
                output = self._process_trial_result(result, trajectory_id)
            except Exception as process_error:
                exclude_from_baseline, exception_type = self._classify_exception(process_error)
                # _classify_exception may return the _PASSTHROUGH sentinel; for a
                # processing-time render error there is no usable verifier reward
                # to pass through, so coerce it to a masked (excluded) failure.
                if exclude_from_baseline is self._PASSTHROUGH:
                    exclude_from_baseline = True
                logger.warning(
                    f"Trajectory {trajectory_id} failed during result processing "
                    f"(NOT fatal — skipping this trial): "
                    f"{type(process_error).__name__}: {process_error} "
                    f"(exception_type={exception_type}, "
                    f"exclude_from_baseline={exclude_from_baseline})"
                )
                output = TerminalBenchAgentOutput(
                    response_ids=[0],
                    reward=0,
                    stop_reason="error",
                    loss_mask=[0],
                    prompt_ids=[0],
                    trajectory_id=trajectory_id,
                    exclude_from_baseline=bool(exclude_from_baseline),
                    exception_type=exception_type,
                )
            all_outputs.append(output)

        # For a group of trajectories (n_samples_per_prompt trajectories for the same prompt):
        # - If error classification is DISABLED: if ANY trajectory fails, zero ALL trajectories in the group
        # - If error classification is ENABLED (RLOO-N mode):
        #   - Infrastructure failures (exclude_from_baseline=True): mark for exclusion from baseline
        #   - Agent failures (exclude_from_baseline=False): include in baseline with reward=0
        #   - If ALL trajectories in a group fail, they all get excluded from baseline
        enable_error_classification = self._error_handling_config.get("enable_error_classification", False)

        failed_instance_ids = set()
        num_failed_trajectories = 0  # per-trajectory, rather than per-instance
        num_masked_trajectories = 0  # trajectories excluded from baseline
        successful_outputs: List[TerminalBenchAgentOutput] = []  # only for metrics purpose

        # Track failure types per instance for RLOO-N
        instance_has_infra_failure: Dict[str, bool] = {}
        instance_has_agent_failure: Dict[str, bool] = {}

        for output in all_outputs:
            if output.stop_reason == "error":
                failed_instance_ids.add(output.trajectory_id.instance_id)
                num_failed_trajectories += 1
                if output.exclude_from_baseline:
                    num_masked_trajectories += 1
                    instance_has_infra_failure[output.trajectory_id.instance_id] = True
                else:
                    instance_has_agent_failure[output.trajectory_id.instance_id] = True

        if enable_error_classification:
            # RLOO-N mode: preserve exclude_from_baseline flags, don't cascade failures
            for output in all_outputs:
                if output.stop_reason == "error":
                    # Error outputs already have correct exclude_from_baseline set
                    output.response_ids = [0]
                    output.loss_mask = [0]
                    output.prompt_ids = [0]
                    output.reward = 0
                    output.rollout_logprobs = None  # Clear logprobs to match response_ids length
                    output.rollout_routed_experts = None  # Clear routed_experts to match response_ids length
                else:
                    successful_outputs.append(output)
        else:
            # Legacy mode: if any trajectory fails, zero entire group
            for output in all_outputs:
                if output.trajectory_id.instance_id in failed_instance_ids:
                    output.response_ids = [0]
                    output.stop_reason = "error"
                    output.loss_mask = [0]
                    output.prompt_ids = [0]
                    output.reward = 0
                    output.rollout_logprobs = None  # Clear logprobs to match response_ids length
                    output.rollout_routed_experts = None  # Clear routed_experts to match response_ids length
                    output.exclude_from_baseline = False  # Legacy: include in baseline
                else:
                    successful_outputs.append(output)

        # Calculate rollout metrics for successful outputs
        if len(successful_outputs) > 0:
            rollout_metrics = get_rollout_metrics(
                [output.response_ids for output in successful_outputs],
                [output.reward for output in successful_outputs],
            )
            rollout_metrics["generate/trajectories_summarized"] = sum(
                1 for output in successful_outputs if output.summarization_count > 0
            )
            rollout_metrics["generate/trajectories_truncated"] = sum(
                1 for output in successful_outputs if output.stop_reason == "length"
            )
        else:
            rollout_metrics = {}
        rollout_metrics["generate/num_failed_instances"] = len(failed_instance_ids)
        rollout_metrics["generate/num_failed_trajectories"] = num_failed_trajectories
        rollout_metrics["generate/num_masked_trajectories"] = num_masked_trajectories

        # TIS logprob-alignment metrics (aggregated across all trajectories with
        # logprobs). These make an LCS fallback or alignment failure ALWAYS visible
        # on the dashboard instead of silently degrading TIS:
        #   tis/exact_match_fraction   — fraction of training tokens mapped exactly
        #                                 by token id (target ~1.0 when on-policy)
        #   tis/lcs_fallback_fraction  — fraction recovered only via LCS string match
        #                                 (target ~0.0; > 0 signals tokenizer mismatch)
        #   tis/unaligned_fraction     — fraction with NO recoverable logprob (holes)
        #   tis/alignment_fail_count   — assistant messages where alignment fully failed
        #   tis/lcs_fallback_messages  — assistant messages that took the LCS path
        batch_align = AlignmentStats()
        any_align = False
        for output in all_outputs:
            if getattr(output, "alignment_stats", None) is not None:
                batch_align.merge(output.alignment_stats)
                any_align = True
        if any_align:
            rollout_metrics.update(batch_align.as_metrics(prefix="generate/tis/"))
            if batch_align.n_lcs_messages > 0 or batch_align.n_failed_messages > 0:
                logger.warning(
                    f"TIS alignment: {batch_align.n_exact}/{batch_align.n_tokens} tokens exact, "
                    f"{batch_align.n_lcs} via LCS fallback, {batch_align.n_unaligned} unaligned; "
                    f"{batch_align.n_lcs_messages} LCS-fallback messages, "
                    f"{batch_align.n_failed_messages} failed messages "
                    f"(of {batch_align.n_messages} assistant messages). "
                    f"Non-zero LCS/failure means serving↔training tokenizer divergence."
                )

        # Per-step error counters for tracked exception types.
        # Pre-populate with zeros so every configured exception appears as a
        # consistent time-series on dashboards, then overlay actual counts.
        for exc_type in self._tracked_exceptions:
            rollout_metrics[f"generate/errors/{exc_type}"] = 0

        exception_counts: Dict[str, int] = {}
        for output in all_outputs:
            if output.exception_type:
                exception_counts[output.exception_type] = exception_counts.get(output.exception_type, 0) + 1
        if exception_counts:
            logger.info(f"Exception breakdown: {exception_counts}")
            for exc_type, count in exception_counts.items():
                rollout_metrics[f"generate/errors/{exc_type}"] = count

        logger.info(
            f"Batch generation complete: {num_trials - num_failed_trajectories}/{num_trials} successful, "
            f"{len(failed_instance_ids)} failed instances, "
            f"{num_masked_trajectories} masked (excluded from baseline)"
        )

        # Collect rollout_logprobs if any outputs have them (required for TIS training)
        # For zeroed/failed trajectories, use [0.0] to match response_ids length.
        #
        # SKIP LOGPROBS FOR EVAL: TIS is a training technique - during eval we don't
        # compute gradients, so logprobs are unnecessary. Skipping them avoids issues
        # where failed trials (e.g., DaytonaRateLimitError) block eval progress.
        #
        # EDGE CASE (training only): When TIS is enabled but some trajectories have no logprobs
        # (e.g., due to ContextLengthExceededError mid-turn where Harbor couldn't
        # collect logprobs before the error), we need to handle this gracefully:
        # - If ALL trajectories have None logprobs → return None (TIS will fail upstream)
        # - If SOME have logprobs → fill missing with zeros (TIS can still train on valid ones)
        # - Log a warning when trajectories are missing logprobs for debugging
        rollout_logprobs_list = None

        if is_eval:
            # Skip logprobs processing for eval - TIS only applies to training
            pass
        else:
            has_any_logprobs = any(output.rollout_logprobs is not None for output in all_outputs)
            missing_logprobs_count = sum(1 for output in all_outputs if output.rollout_logprobs is None)

            if has_any_logprobs:
                rollout_logprobs_list = []
                for output in all_outputs:
                    if output.rollout_logprobs is not None:
                        rollout_logprobs_list.append(output.rollout_logprobs)
                    else:
                        # For trajectories missing logprobs, fill with zeros
                        # This allows partial training on trajectories that have valid logprobs
                        rollout_logprobs_list.append([0.0] * len(output.response_ids))

                if missing_logprobs_count > 0 and self._collect_rollout_details:
                    # Only warn about missing logprobs if TIS is expected (collect_rollout_details=true)
                    logger.warning(
                        f"TIS mode: {missing_logprobs_count}/{num_trials} trajectories missing logprobs "
                        f"(likely due to context length errors). Filled with zeros. "
                        f"These trajectories will have no gradient contribution from TIS."
                    )
            elif missing_logprobs_count > 0 and self._collect_rollout_details:
                # All trajectories missing logprobs - this is a problem for TIS
                # Only log error if TIS is expected (collect_rollout_details=true)
                logger.error(
                    f"TIS mode: ALL {num_trials} trajectories missing logprobs. "
                    f"This batch cannot be used for TIS training. "
                    f"Check if Harbor is collecting rollout_details (collect_rollout_details=true) "
                    f"and if context length errors are preventing logprob collection."
                )

        # Collect routed_experts (Stage 1 MoE router-replay capture rail). Mirrors
        # the rollout_logprobs gather and mixed-presence handling. Gated on
        # moe_router_replay so the GeneratorOutput is byte-identical when off (the
        # key is omitted entirely, not set to None). Skipped for eval like logprobs.
        rollout_routed_experts_list = None
        if self._moe_router_replay and not is_eval:
            has_any_re = any(output.rollout_routed_experts is not None for output in all_outputs)
            if has_any_re:
                # Learn the [L, K] sentinel-row shape from the first real sample so
                # missing/failed samples are sentinel-filled at the correct width.
                sentinel_row = [[SENTINEL_EXPERT_ID]]
                for output in all_outputs:
                    if output.rollout_routed_experts:
                        sentinel_row = _sentinel_routed_experts_row(output.rollout_routed_experts[0])
                        break
                rollout_routed_experts_list = []
                for output in all_outputs:
                    if output.rollout_routed_experts is not None:
                        rollout_routed_experts_list.append(output.rollout_routed_experts)
                    else:
                        # Sentinel-fill missing samples to match response_ids length.
                        rollout_routed_experts_list.append(
                            [list(sentinel_row) for _ in range(len(output.response_ids))]
                        )

        # Aggregate per-component reward metrics (composite shaper only)
        component_outputs = [o for o in all_outputs if o.reward_components is not None]
        if component_outputs:
            component_names = set()
            for o in component_outputs:
                component_names.update(o.reward_components.keys())
            component_metrics = {}
            for name in sorted(component_names):
                values = [o.reward_components.get(name, 0.0) for o in component_outputs]
                avg = sum(values) / len(values) if values else 0.0
                component_metrics[f"reward/component_{name}"] = avg
            rollout_metrics.update(component_metrics)
            logger.info(
                f"Reward component averages (n={len(component_outputs)}): "
                + ", ".join(f"{k.split('_', 2)[-1]}={v:.3f}" for k, v in sorted(component_metrics.items()))
            )

        # Estimate actual_global_step for staleness tracking. Use the EARLIEST
        # Harbor pickup time (`started_at`) across the group's trials — that's
        # the moment the first trial transitioned from queued-in-Harbor to
        # actually-running, i.e. the first attempt to dispatch to vLLM.
        # Robust to vLLM fragility (no dependence on vLLM responses) and to
        # individual-trial failures (we take whatever started). Falls back to
        # the global_step captured at generate() entry (worst case = same as
        # the pre-patch behavior).
        actual_global_step: Optional[int] = None
        earliest_started_ts: Optional[float] = None
        for r in results:
            if isinstance(r, TrialResult) and r.started_at is not None:
                ts = r.started_at.timestamp()
                if earliest_started_ts is None or ts < earliest_started_ts:
                    earliest_started_ts = ts
        # Record current step+time again so the history covers gather completion.
        self._record_step_time()
        if earliest_started_ts is not None:
            actual_global_step = self._step_at_time(earliest_started_ts)
        if actual_global_step is None:
            actual_global_step = entry_global_step

        generator_output: GeneratorOutput = {
            "prompt_token_ids": [output.prompt_ids for output in all_outputs],
            "response_ids": [output.response_ids for output in all_outputs],
            "rewards": [output.reward for output in all_outputs],
            "loss_masks": [output.loss_mask for output in all_outputs],
            "stop_reasons": [output.stop_reason for output in all_outputs],
            "rollout_metrics": rollout_metrics,
            "rollout_logprobs": rollout_logprobs_list,
            "exclude_from_baseline": [output.exclude_from_baseline for output in all_outputs],
            "actual_global_step": actual_global_step,
        }

        # Only attach routed_experts when router-replay is on, so the flag-off
        # GeneratorOutput dict is byte-identical to today (key absent, not None).
        if rollout_routed_experts_list is not None:
            generator_output["rollout_routed_experts"] = rollout_routed_experts_list

        return generator_output

    # Sentinel: returned by _classify_exception when the exception should be
    # treated as if it never happened (fall through to normal verifier flow).
    _PASSTHROUGH = object()

    def _classify_exception(self, exception: Exception) -> tuple[bool | object, str]:
        """
        Classify an exception as infrastructure failure (mask), agent failure
        (zero), or passthrough (ignore the exception and use the verifier
        reward as-is).

        Args:
            exception: The exception to classify.

        Returns:
            Tuple of (treatment, exception_type_name)
            - treatment=True: Infrastructure failure, exclude from RLOO-N baseline
            - treatment=False: Agent failure, include in baseline with reward=0
            - treatment=_PASSTHROUGH: Ignore exception, use verifier result normally
        """
        exception_type = type(exception).__name__

        # If error classification is disabled, treat all errors as agent failures
        if not self._error_handling_config.get("enable_error_classification", False):
            return False, exception_type

        passthrough_exceptions = self._error_handling_config.get("passthrough_exceptions", set())
        mask_exceptions = self._error_handling_config.get("mask_exceptions", set())
        zero_exceptions = self._error_handling_config.get("zero_exceptions", set())
        default_treatment = self._error_handling_config.get("default_error_treatment", "zero")

        # Check if this exception type should be passed through (use verifier reward)
        if exception_type in passthrough_exceptions:
            logger.debug(f"Exception {exception_type} classified as PASSTHROUGH (use verifier reward)")
            return self._PASSTHROUGH, exception_type

        # Check if this exception type should be masked (excluded from baseline)
        if exception_type in mask_exceptions:
            logger.debug(f"Exception {exception_type} classified as MASK (infrastructure failure)")
            return True, exception_type

        # Check if this exception type should be zeroed (included in baseline)
        if exception_type in zero_exceptions:
            logger.debug(f"Exception {exception_type} classified as ZERO (agent failure)")
            return False, exception_type

        # Default treatment for unclassified exceptions
        if default_treatment == "passthrough":
            logger.debug(f"Exception {exception_type} not in config, using default: PASSTHROUGH")
            return self._PASSTHROUGH, exception_type
        exclude = default_treatment == "mask"
        logger.debug(
            f"Exception {exception_type} not in config, using default treatment: " f"{'MASK' if exclude else 'ZERO'}"
        )
        return exclude, exception_type

    def _process_trial_result(
        self,
        result: TrialResult | Exception,
        trajectory_id: TrajectoryID,
    ) -> TerminalBenchAgentOutput:
        """
        Process a TrialResult from QueueOrchestrator into TerminalBenchAgentOutput.

        Args:
            result: TrialResult from Harbor or an Exception if the trial failed completely.
            trajectory_id: The trajectory ID for this trial.

        Returns:
            TerminalBenchAgentOutput with processed rollout data.
        """
        # Handle exceptions from the orchestrator
        if isinstance(result, Exception):
            exclude_from_baseline, exception_type = self._classify_exception(result)
            logger.warning(
                f"Trajectory {trajectory_id} failed with exception: {result} "
                f"(type={exception_type}, exclude_from_baseline={exclude_from_baseline})"
            )
            return TerminalBenchAgentOutput(
                response_ids=[0],
                reward=0,
                stop_reason="error",
                loss_mask=[0],
                prompt_ids=[0],
                trajectory_id=trajectory_id,
                exclude_from_baseline=exclude_from_baseline,
                exception_type=exception_type,
            )

        # Check for exception_info - Harbor may return both exception_info AND
        # verifier_result when a trial had an error (e.g., AgentTimeoutError,
        # ContextLengthExceededError) but the verifier still ran.
        exception_info = getattr(result, "exception_info", None)
        if exception_info is not None:
            exception_type = "UnknownError"

            if hasattr(exception_info, "exception_type"):
                exception_type = exception_info.exception_type
            elif hasattr(exception_info, "__class__"):
                exception_type = type(exception_info).__name__

            # Create a mock exception to classify
            class MockException(Exception):
                pass

            MockException.__name__ = exception_type
            treatment, _ = self._classify_exception(MockException())

            # Passthrough: ignore the exception and fall through to normal
            # verifier processing. The agent hit a soft limit (timeout, context
            # length) but the verifier still ran and produced a real reward.
            if treatment is self._PASSTHROUGH:
                if result.verifier_result:
                    logger.info(
                        f"Trajectory {trajectory_id}: {exception_type} classified as PASSTHROUGH, "
                        f"using verifier reward"
                    )
                    # Fall through to normal processing below
                else:
                    # Passthrough requested but no verifier result — treat as mask
                    logger.warning(
                        f"Trajectory {trajectory_id}: {exception_type} classified as PASSTHROUGH "
                        f"but no verifier result available, masking instead"
                    )
                    return TerminalBenchAgentOutput(
                        response_ids=[0],
                        reward=0,
                        stop_reason="error",
                        loss_mask=[0],
                        prompt_ids=[0],
                        trajectory_id=trajectory_id,
                        exclude_from_baseline=True,
                        exception_type=exception_type,
                    )
            else:
                exclude_from_baseline = bool(treatment)
                logger.warning(
                    f"Trajectory {trajectory_id} failed with Harbor exception: "
                    f"{exception_info.exception_message if hasattr(exception_info, 'exception_message') else exception_info} "
                    f"(type={exception_type}, exclude_from_baseline={exclude_from_baseline})"
                )
                return TerminalBenchAgentOutput(
                    response_ids=[0],
                    reward=0,
                    stop_reason="error",
                    loss_mask=[0],
                    prompt_ids=[0],
                    trajectory_id=trajectory_id,
                    exclude_from_baseline=exclude_from_baseline,
                    exception_type=exception_type,
                )

        # Check for missing verifier result (trial ran but didn't produce valid output)
        # Note: exception_info is already handled above, so if we reach here it's None
        if not result.verifier_result:
            logger.warning(
                f"Trajectory {trajectory_id} failed: No verifier result and no exception info. "
                f"This is unexpected - marking as infrastructure failure."
            )
            return TerminalBenchAgentOutput(
                response_ids=[0],
                reward=0,
                stop_reason="error",
                loss_mask=[0],
                prompt_ids=[0],
                trajectory_id=trajectory_id,
                exclude_from_baseline=True,  # Infrastructure issue - exclude from baseline
                exception_type="MissingVerifierResult",
            )

        # Extract data from successful trial
        try:
            original_reward = result.verifier_result.rewards["reward"]
            chat_history = result.agent_result.metadata["all_messages"]
            summarization_count = result.agent_result.metadata["summarization_count"]
        except (KeyError, AttributeError, TypeError) as e:
            # Data extraction failure is typically an infrastructure issue
            exception_type = type(e).__name__
            exclude_from_baseline, _ = self._classify_exception(e)
            logger.warning(
                f"Trajectory {trajectory_id} failed: Could not extract results. "
                f"Error: {e}, Result: {result} "
                f"(type={exception_type}, exclude_from_baseline={exclude_from_baseline})"
            )
            return TerminalBenchAgentOutput(
                response_ids=[0],
                reward=0,
                stop_reason="error",
                loss_mask=[0],
                prompt_ids=[0],
                trajectory_id=trajectory_id,
                exclude_from_baseline=exclude_from_baseline,
                exception_type=exception_type,
            )

        # Apply reward shaping if enabled
        reward_components: Optional[Dict[str, float]] = None
        if self._reward_shaping_config.get("enable_reward_shaping", True):
            verifier_stdout = getattr(result.verifier_result, "stdout", None)
            shaper_name = self._reward_shaping_config.get("reward_shaper", "pass_ratio")
            shaper_kwargs = self._reward_shaping_config.get("shaper_kwargs", {})

            # For container shapers (composite / composite_loop), capture the
            # per-component breakdown.
            if shaper_name in ("composite", "composite_loop"):
                reward, reward_components = shape_reward_with_components(
                    stdout=verifier_stdout,
                    original_reward=original_reward,
                    parser_name=self._reward_shaping_config.get("reward_parser"),
                    shaper_kwargs=shaper_kwargs,
                    chat_history=chat_history,
                    shaper_name=shaper_name,
                )
            else:
                reward = shape_reward_from_output(
                    stdout=verifier_stdout,
                    original_reward=original_reward,
                    parser_name=self._reward_shaping_config.get("reward_parser"),
                    shaper_name=shaper_name,
                    shaper_kwargs=shaper_kwargs,
                    fallback_to_original=self._reward_shaping_config.get("reward_shaping_fallback", True),
                    chat_history=chat_history,
                )
            if reward != original_reward:
                logger.debug(
                    f"Trajectory {trajectory_id}: reward shaped {original_reward:.3f} -> {reward:.3f}"
                    + (f" (components={reward_components})" if reward_components else "")
                )
        else:
            reward = original_reward

        # Separate system messages from the conversation.
        # Some agents (e.g. terminus-kira) include a system prompt;
        # others (e.g. terminus-2) do not.
        system_msgs = []
        conversation = []
        for msg in chat_history or []:
            if msg["role"] == "system":
                system_msgs.append(msg)
            else:
                conversation.append(msg)

        # Validate: need at least a user message and one response
        if len(conversation) < 2 or conversation[0]["role"] != "user":
            # Invalid chat history is typically an infrastructure/serialization issue
            logger.warning(
                f"Trajectory {trajectory_id} failed: Invalid chat history structure. " f"chat_history: {chat_history}"
            )
            return TerminalBenchAgentOutput(
                response_ids=[0],
                reward=0,
                stop_reason="error",
                loss_mask=[0],
                prompt_ids=[0],
                trajectory_id=trajectory_id,
                exclude_from_baseline=True,  # Infrastructure issue
                exception_type="InvalidChatHistory",
            )

        # Process successful trial
        # Prompt = system messages (if any) + first user message
        prompt = system_msgs + [conversation[0]]
        prompt_ids = _normalize_prompt_token_ids(
            self.tokenizer.apply_chat_template(
                prompt,
                add_generation_prompt=False,
                tokenize=True,
                chat_template=self.custom_chat_template_content,
            )
        )
        initial_prompt_length = len(prompt_ids)

        # Process response messages (everything after the first message)
        response_messages = conversation[1:]

        # Extract per-turn logprobs from Harbor's rollout_details (required for TIS)
        rollout_details = getattr(result.agent_result, "rollout_details", None)
        assistant_logprobs = extract_logprobs_from_rollout_details(rollout_details)
        # Exact-alignment ids: Harbor's per-turn completion_token_ids, index-aligned
        # with assistant_logprobs. Enables the exact (no re-tokenization guess) TIS path.
        assistant_token_ids = extract_token_ids_from_rollout_details(rollout_details)
        # Accumulate per-message exact/LCS/fail counts; surfaced as tis/* metrics.
        alignment_stats = AlignmentStats() if assistant_logprobs is not None else None

        # Extract per-turn MoE routed_experts (Stage 1 capture rail). Gated on
        # moe_router_replay so the flag-off path is byte-identical: when off we
        # never pass assistant_routed_experts, so the chokepoint returns its 3-tuple.
        #
        # IMPORTANT: even with moe_router_replay ON, the capture rail can legitimately
        # produce NO routed_experts for a given trial — extract_routed_experts_from_rollout_details
        # returns None whenever rollout_details is empty/missing or simply doesn't carry
        # routed_experts (e.g. a trial that errored mid-rollout, or a turn the P1 HTTP
        # capture didn't tag). In that case we MUST take the 3-tuple path: passing
        # assistant_routed_experts=None makes get_response_ids_and_loss_mask_from_messages
        # return a 3-tuple, and unpacking 4 here raised
        # `ValueError: not enough values to unpack (expected 4, got 3)` — which crashed the
        # RolloutCoordinator shard on the first completed 80B trial. The downstream batch
        # collation (generators/utils.py concatenate_generator_outputs) already tolerates
        # mixed presence/absence of rollout_routed_experts across trials via its
        # has_routed_experts any-check + sentinel fill, so leaving this trial's
        # rollout_routed_experts=None is safe.
        rollout_routed_experts = None
        assistant_routed_experts = (
            extract_routed_experts_from_rollout_details(rollout_details) if self._moe_router_replay else None
        )
        if assistant_routed_experts is not None:
            (
                response_ids,
                loss_mask,
                rollout_logprobs,
                rollout_routed_experts,
            ) = get_response_ids_and_loss_mask_from_messages(
                response_messages,
                self.tokenizer,
                assistant_logprobs,
                custom_chat_template=self.custom_chat_template_content,
                assistant_routed_experts=assistant_routed_experts,
                assistant_token_ids=assistant_token_ids,
                alignment_stats=alignment_stats,
            )
        else:
            response_ids, loss_mask, rollout_logprobs = get_response_ids_and_loss_mask_from_messages(
                response_messages,
                self.tokenizer,
                assistant_logprobs,
                custom_chat_template=self.custom_chat_template_content,
                assistant_token_ids=assistant_token_ids,
                alignment_stats=alignment_stats,
            )

        # Determine stop reason
        max_response_tokens = (
            self.generator_cfg.sampling_params.max_generate_length
            + self.generator_cfg.max_input_length
            - initial_prompt_length
        )
        stop_reason = "complete"  # Default for trial completion
        if len(response_ids) > max_response_tokens:
            stop_reason = "length"

        # Truncate to maximum allowed length
        response_ids = response_ids[:max_response_tokens]
        loss_mask = loss_mask[:max_response_tokens]
        if rollout_logprobs is not None:
            rollout_logprobs = rollout_logprobs[:max_response_tokens]
        if rollout_routed_experts is not None:
            rollout_routed_experts = rollout_routed_experts[:max_response_tokens]

        return TerminalBenchAgentOutput(
            response_ids=response_ids,
            reward=reward,
            stop_reason=stop_reason,
            loss_mask=loss_mask,
            prompt_ids=prompt_ids,
            trajectory_id=trajectory_id,
            rollout_logprobs=rollout_logprobs,
            rollout_routed_experts=rollout_routed_experts,
            summarization_count=summarization_count,
            reward_components=reward_components,
            alignment_stats=alignment_stats,
        )
