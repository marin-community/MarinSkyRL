import asyncio
import logging
import os
import re
import time
from collections import deque
from dataclasses import dataclass, replace
from typing import Callable, Deque, List, Optional, Dict, Any, Tuple

import numpy as np
from skyrl_gym.verification import (
    RewardResult,
    RolloutEvidence,
    TrainingDisposition,
    VerificationResult,
    VerificationStatus,
)
from loguru import logger
from uuid import uuid4
from skyrl_train.trajectory_runners.base import TrajectoryRunner, TrajectoryRequestBatch, TrajectoryBatch, TrajectoryID
from skyrl_train.trajectory_runners.projections import project_loss_mask
from skyrl_train.metric_names import TIS_LCS_FALLBACK_ALERT_METRIC, TIS_METRIC_PREFIX
from skyrl_train.trajectory_runners.trajectory_processing import (
    BATCH_ERROR_METRIC_PREFIX,
    get_batch_failure_metrics,
    get_rollout_metrics,
    get_response_ids_and_loss_mask_from_messages,
    get_generation_prompt_ids,
    detect_qwen3_5_empty_think_prefix,
    extract_logprobs_from_rollout_details,
    extract_token_ids_from_rollout_details,
    extract_prompt_token_ids_from_rollout_details,
    extract_routed_experts_from_rollout_details,
    normalize_token_ids,
    AlignmentStats,
    _sentinel_routed_experts_row,
    SENTINEL_EXPERT_ID,
)
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.utils.reward_shaping import shape_reward_from_output, shape_reward_with_components
from skyrl_train.utils.harbor_errors import (
    ErrorTreatment,
    classify_exception_type,
    passthrough_logprob_error_type,
    treatment_excludes_from_baseline,
)
from skyrl_train.utils.span_tagger import tag_response_spans
from skyrl_train.utils.pbs_shaping import compute_pbs_token_shaping
from omegaconf import DictConfig
from pathlib import Path

# Harbor orchestrator and trial imports.
# QueueOrchestrator + OrchestratorEvent come through a compat shim because
# Harbor removed ``harbor.orchestrators`` in favor of ``harbor.trial.queue``
# and ``harbor.trial.hooks``. See _harbor_compat.py for the wrapper.
from skyrl_train.trajectory_runners.harbor._harbor_compat import (
    OrchestratorEvent,
    QueueOrchestrator,
    create_rollback_hook,
)
from harbor.models.trial.config import TrialConfig
from harbor.models.trial.result import TrialResult
from harbor.utils.traces_utils import normalize_message

# Schema-driven Harbor config mapping
from skyrl_train.trajectory_runners.harbor.configuration import HarborConfigBuilder
from skyrl_train.trajectory_runners.harbor.contracts import verification_from_harbor_result
from skyrl_train.trajectory_runners.harbor.truncation_penalty import apply_truncation_penalty, detect_turn_truncation

# Incremental, trial-indexed reader for the shared opencode literal log.
from skyrl_train.trajectory_runners.harbor.literal_log_store import LiteralLogStore

# Maximum restart attempts for orchestrator recovery
MAX_ORCHESTRATOR_RESTART_ATTEMPTS = 3


def _select_opencode_literal_chain(entries: List[Dict[str, Any]], trial_id: str) -> List[Dict[str, Any]]:
    """Return the final continuous agent-call chain for one opencode trial.

    The controller RecordProxy captures every request carrying the trial header.
    OpenCode also makes auxiliary requests, and a context reset begins a fresh,
    summary-seeded conversation. Neither belongs in the causal sequence assembled
    from the final request's chat history. A real next agent turn has the previous
    served prompt and completion as an exact prefix of its prompt, so retain the
    longest such route ending at the final captured call.

    Entries without complete token-id streams keep the existing all-entry fallback:
    they can still support the re-tokenized TIS path but cannot prove a TITO route.
    """
    candidates = [
        entry
        for entry in entries
        if entry.get("trial_id") == trial_id
        and entry.get("status_code") == 200
        and isinstance(entry.get("literal"), dict)
        and entry["literal"].get("completion_token_ids")
    ]
    candidates.sort(key=lambda entry: entry.get("timestamp") or 0.0)
    if len(candidates) < 2:
        return candidates

    def has_complete_ids(entry: Dict[str, Any]) -> bool:
        literal = entry["literal"]
        return isinstance(literal.get("prompt_token_ids"), list) and isinstance(
            literal.get("completion_token_ids"), list
        )

    if not all(has_complete_ids(entry) for entry in candidates):
        return candidates

    parents: List[Optional[int]] = [None] * len(candidates)
    lengths = [1] * len(candidates)
    for current_idx, current in enumerate(candidates):
        current_prompt = current["literal"]["prompt_token_ids"]
        for previous_idx, previous in enumerate(candidates[:current_idx]):
            previous_literal = previous["literal"]
            expected_prefix = previous_literal["prompt_token_ids"] + previous_literal["completion_token_ids"]
            if current_prompt[: len(expected_prefix)] != expected_prefix:
                continue
            candidate_length = lengths[previous_idx] + 1
            if candidate_length > lengths[current_idx]:
                lengths[current_idx] = candidate_length
                parents[current_idx] = previous_idx

    route_indices = []
    current_idx: Optional[int] = len(candidates) - 1
    while current_idx is not None:
        route_indices.append(current_idx)
        current_idx = parents[current_idx]
    route_indices.reverse()
    return [candidates[index] for index in route_indices]


# Markers indicating the agent confirmed task completion (terminus-2 family).
# The agent's raw assistant response carries the completion marker in-band:
#   - XML parser: ``<task_complete>true</task_complete>``
#   - JSON parser: ``"task_complete": true``
# and Harbor's ATIF serialization emits a ``mark_task_complete`` tool call.
# We scan assistant message content for any of these so the detector is robust
# across the terminus-2 response formats (matches the agent's own
# ``_check_task_complete`` logic).
_TASK_COMPLETE_PATTERNS = (
    re.compile(r"<task_complete>\s*true\s*</task_complete>", re.IGNORECASE),
    re.compile(r'["\']task_complete["\']\s*:\s*true', re.IGNORECASE),
    re.compile(r"mark_task_complete", re.IGNORECASE),
)


def detect_termination_signals(
    chat_history: Optional[List[Dict[str, Any]]],
    verifier_reward: float,
) -> Dict[str, Any]:
    """Derive the Stage-1 (M2) termination signals from a finished trajectory.

    Returns a ``trajectory_context`` dict consumed by the ``composite_loop``
    terminate component:
      - ``mark_complete`` (bool): an assistant turn confirmed task completion
        (matched any ``_TASK_COMPLETE_PATTERNS`` marker).
      - ``verifier_reward`` (float): the ground-truth verifier verdict (passed
        through unchanged).
      - ``premature_stop`` (bool): the trajectory ended WITHOUT a confirmed
        completion (ran to the agent/step wall / timeout) — the never-terminated
        mode. This is the complement of ``mark_complete``.

    Detection is read-only and never mutates ``chat_history``.
    """
    mark_complete = False
    if chat_history and isinstance(chat_history, list):
        for msg in chat_history:
            if not isinstance(msg, dict) or msg.get("role") != "assistant":
                continue
            content = msg.get("content") or ""
            if not isinstance(content, str):
                content = str(content)
            # Also surface any structured tool_calls (ATIF) carrying the marker.
            tool_calls = msg.get("tool_calls") or []
            tool_blob = ""
            if isinstance(tool_calls, list):
                tool_blob = " ".join(
                    str(tc.get("function_name") or (tc.get("function") or {}).get("name") or "")
                    for tc in tool_calls
                    if isinstance(tc, dict)
                )
            haystack = content + " " + tool_blob
            if any(p.search(haystack) for p in _TASK_COMPLETE_PATTERNS):
                mark_complete = True
                break

    try:
        vr = float(verifier_reward)
    except (TypeError, ValueError):
        vr = 0.0

    return {
        "mark_complete": mark_complete,
        "verifier_reward": vr,
        # Never-terminated == ran to the wall without confirming completion.
        "premature_stop": not mark_complete,
    }


@dataclass
class TerminalBenchAgentOutput:
    evidence: RolloutEvidence
    verification: VerificationResult
    reward_result: RewardResult
    disposition: TrainingDisposition
    loss_mask: List[int]
    trajectory_id: TrajectoryID
    summarization_count: Optional[int] = None
    # TIS logprob-alignment bookkeeping (exact-vs-LCS-vs-failed token counts).
    # Aggregated into rollout_metrics as tis/* so an LCS fallback or alignment
    # failure can never silently degrade TIS. None when no logprobs were present.
    alignment_stats: Optional[AlignmentStats] = None
    # Loop-behavior span tags aligned 1:1 with response token IDs.
    response_span_tags: Optional[List[int]] = None
    # True when the truncation penalty was applied (stop_reason=="length" +
    # original_reward==0 + truncation_penalty>0). Counted into rollout_metrics.
    truncation_penalized: bool = False


def _failed_agent_output(
    trajectory_id: TrajectoryID,
    *,
    verification: VerificationResult,
    exception_type: str,
    exclude_from_baseline: bool,
) -> TerminalBenchAgentOutput:
    return TerminalBenchAgentOutput(
        evidence=_failure_evidence(),
        verification=verification,
        reward_result=_zero_reward(),
        disposition=TrainingDisposition(
            loss_eligible=False,
            baseline_eligible=not exclude_from_baseline,
            reason=verification.reason or "trajectory failure",
            exception_type=exception_type,
        ),
        loss_mask=[0],
        trajectory_id=trajectory_id,
    )


def _failure_evidence() -> RolloutEvidence:
    return RolloutEvidence(
        stop_reason="error",
        generated_token_count=0,
        prompt_token_ids=(0,),
        response_token_ids=(0,),
    )


def _zero_reward() -> RewardResult:
    return RewardResult(unshaped_reward=None, optimization_reward=0.0)


def _rollout_evidence_from_harbor(
    *,
    chat_history: List[Dict[str, Any]],
    stop_reason: str,
    prompt_ids: List[int],
    response_ids: List[int],
    loss_mask: List[int],
    rollout_logprobs: Optional[List[float]],
    rollout_routed_experts: Optional[List[List[List[int]]]],
) -> RolloutEvidence:
    final_response = next(
        (str(message.get("content") or "") for message in reversed(chat_history) if message.get("role") == "assistant"),
        None,
    )
    return RolloutEvidence(
        messages=tuple(chat_history),
        response=final_response,
        stop_reason=stop_reason,
        generated_token_count=sum(bool(value) for value in loss_mask),
        prompt_token_ids=tuple(prompt_ids),
        response_token_ids=tuple(response_ids),
        behavior_logprobs=None if rollout_logprobs is None else tuple(rollout_logprobs),
        routed_experts=(
            None
            if rollout_routed_experts is None
            else tuple(tuple(tuple(layer) for layer in token) for token in rollout_routed_experts)
        ),
    )


def _completed_disposition(
    *,
    preserve_timeout: bool,
    preserve_exclude_from_baseline: bool,
    preserve_exception_type: Optional[str],
) -> TrainingDisposition:
    return TrainingDisposition(
        loss_eligible=True,
        baseline_eligible=not preserve_exclude_from_baseline if preserve_timeout else True,
        reason=f"preserved {preserve_exception_type}" if preserve_timeout else "verified",
        exception_type=preserve_exception_type if preserve_timeout else None,
    )


def _clear_failed_trajectory(output: TerminalBenchAgentOutput) -> None:
    """Replace an invalidated trajectory with an aligned zero-reward stub."""
    output.evidence = _failure_evidence()
    output.loss_mask = [0]
    output.response_span_tags = None
    output.alignment_stats = None
    output.truncation_penalized = False
    output.reward_result = _zero_reward()
    if output.disposition.loss_eligible:
        output.disposition = replace(
            output.disposition,
            loss_eligible=False,
            reason="trajectory group invalidated by peer failure",
        )


class HarborTrajectoryRunner(TrajectoryRunner):
    def __init__(
        self,
        trajectory_runner_cfg: DictConfig,
        terminal_bench_cfg: DictConfig,
        inference_engine_client: InferenceEngineClient,
        tokenizer,
        tis_lcs_alert_threshold: float,
        moe_router_replay: bool = False,
        rollout_logprobs_required: bool = False,
        tito_full: Optional[bool] = None,
        tis_splice: bool = True,
    ):
        """
        Args:
            trajectory_runner_cfg: trajectory-runner configuration
            terminal_bench_cfg: DictConfig object containing the terminal bench configuration
            inference_engine_client: InferenceEngineClient object for interacting with the inference engines
            tokenizer: tokenizer object for encoding and decoding text
            moe_router_replay: when True, capture per-token MoE routed_experts from
                Harbor rollout_details and plumb them through to the training batch
                (Stage 1 of the FSDP2 EP/router-replay port). Default False keeps the
                TrajectoryBatch byte-identical to today.
            rollout_logprobs_required: Whether the selected policy objective consumes
                behavior-policy logprobs. Full-TITO rollout assembly defaults to this.
            tito_full: ``trainer.algorithm.tito_full`` — explicit full-TITO override.
                None = auto (default to ``rollout_logprobs_required``); True/False = force.
        """
        self.base_url = f"http://{trajectory_runner_cfg.http_endpoint_host}:{trajectory_runner_cfg.http_endpoint_port}"
        # Native controller-ingress (opencode-RL literal capture): when the runner stood up
        # a controller-ingress endpoint it publishes the minted capability URL. The AGENT
        # (opencode, in a Daytona sandbox) must reach vLLM over the public internet at that
        # URL — NOT the loopback self.base_url, which only resolves in-cluster. So the
        # per-trial agent api_base prefers the ingress URL when present; unset
        # (default/direct) => exactly f"{self.base_url}/v1" as before (byte-identical).
        # The verifier + re-tokenize/TIS paths keep using self.base_url.
        #
        # SOURCE PRECEDENCE (cfg first, env fallback). This runner is constructed
        # INSIDE a Ray task/actor (skyrl_entrypoint / RolloutCoordinator) that does NOT
        # inherit the run_rl driver's late HARBOR_MODEL_ENDPOINT env mutation (the runner
        # attaches to a Ray cluster started BEFORE the mint). run_rl therefore threads the
        # URL through the cfg as ``terminal_bench_config.agent_api_base``, which crosses the
        # .remote() boundary as DATA and reaches every worker. The os.environ fallback keeps
        # the driver-side / direct-launch paths working unchanged.
        _cfg_endpoint = ""
        try:
            _cfg_endpoint = str(terminal_bench_cfg.get("agent_api_base", "") or "").strip()
        except Exception:  # pragma: no cover - defensive: cfg may not be a mapping
            _cfg_endpoint = ""
        _ingress_endpoint = _cfg_endpoint or os.environ.get("HARBOR_MODEL_ENDPOINT", "").strip()
        self._agent_api_base = _ingress_endpoint or f"{self.base_url}/v1"
        if _ingress_endpoint:
            logging.getLogger("harbor").info(
                "[terminal_bench] agent api_base <- %s (controller ingress)",
                "terminal_bench_config.agent_api_base" if _cfg_endpoint else "HARBOR_MODEL_ENDPOINT",
            )
        # Shared RecordProxy literal-log path, resolved cfg-FIRST then env-fallback for
        # the SAME Ray process-boundary reason as agent_api_base above: run_rl publishes
        # it on os.environ['OTAGENT_LITERAL_LOG_PATH'] in the driver, but THIS runner
        # is constructed inside a Ray worker that never inherits that late env mutation,
        # so the driver threads it through the cfg as terminal_bench_config.literal_log_path.
        # None when record_literal is off / direct launch → the opencode correlation +
        # chat_history recovery no-op exactly as before. Used by
        # _maybe_correlate_opencode_rollout_details / _maybe_build_opencode_chat_history.
        _cfg_literal_log = ""
        try:
            _cfg_literal_log = str(terminal_bench_cfg.get("literal_log_path", "") or "").strip()
        except Exception:  # pragma: no cover - defensive: cfg may not be a mapping
            _cfg_literal_log = ""
        self._literal_log_path = _cfg_literal_log or os.environ.get("OTAGENT_LITERAL_LOG_PATH") or None
        # Incremental, trial-indexed reader over that shared log (owns its own lock +
        # typed cursor state). Read-only; the writer is the external harbor RecordProxy.
        self._literal_log_store = LiteralLogStore()
        self.trajectory_runner_cfg = trajectory_runner_cfg
        self.tokenizer = tokenizer
        self.model_name = trajectory_runner_cfg.model_name
        self._moe_router_replay = moe_router_replay
        # Full-TITO follows the explicit setting when present, otherwise logprob consumption.
        self._rollout_logprobs_required = rollout_logprobs_required
        self._tito_full = tito_full
        self._tis_splice = tis_splice
        self._tis_lcs_alert_threshold = tis_lcs_alert_threshold

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

        # Loop-behavior reward shaping (Stage B / F5 + F4): master gate for the
        # per-token shaping channel + span tagger. Default False -> the runner
        # emits neither field, so the TrajectoryBatch is byte-identical to today.
        self._enable_token_reward_channel = bool(self._reward_shaping_config.get("enable_token_reward_channel", False))
        self._enable_span_tagging = bool(self._reward_shaping_config.get("enable_span_tagging", True))

        # Loop-behavior reward shaping (Stage C / F2 + F6): potential-based
        # shaping (PBS) of the EDIT-token span from the in-trajectory test-delta.
        # Default False -> the channel ships Stage-B ZEROS (no-op). Requires the
        # token reward channel AND span tagging to be on (PBS scatters onto the
        # F4 EDIT tags). PBS is policy-invariant (Ng 1999) and bounded.
        self._enable_pbs_shaping = bool(self._reward_shaping_config.get("enable_pbs_shaping", False))
        self._pbs_gamma = float(self._reward_shaping_config.get("pbs_gamma", 1.0))
        self._pbs_max_total_shaping = float(self._reward_shaping_config.get("pbs_max_total_shaping", 0.3))
        self._pbs_potential_shape = str(self._reward_shaping_config.get("pbs_potential_shape", "linear"))

        # Truncation penalty: penalize cap-truncated trials so they are
        # distinguishable from honest failures in the advantage signal. Default
        # 0.0 -> no-op (byte-identical). Applied post-shaping to trials with
        # stop_reason == "length" and original_reward == 0.
        self._truncation_penalty = float(self._reward_shaping_config.get("truncation_penalty", 0.0))

        # Error handling config (for RLOO-N advantage estimator)
        self._error_handling_config = self._harbor_config_builder.get_error_handling_config()

        # Preserve-logprobs-on-soft-timeout: when a trajectory was FULLY generated
        # but a POST-generation step failed (VerifierTimeoutError -> no reward, or a
        # soft AgentTimeout without a verifier result), keep the generated
        # tokens+logprobs at reward=0 instead of discarding them, so an RLOO-N group
        # is not dropped for all-missing-logprobs (global_step stuck at 0). Gated +
        # length-checked at the discard sites; falls back to the discard stub when the
        # generation has no TIS-valid (length-matched) logprobs. Default ON; set
        # harbor.error_handling.preserve_logprobs_on_timeout=false for the old
        # discard-everything behavior.
        self._preserve_logprobs_on_timeout = self._error_handling_config.preserve_logprobs_on_timeout

        # TIS (Truncated Importance Sampling) config
        # Only show TIS-related warnings when collect_rollout_details is enabled
        self._collect_rollout_details = self._harbor_config_builder.get_collect_rollout_details()

        # Tracked exception types for per-step error counters.
        # Sourced from the retry config's exclude_exceptions (the terminal failures
        # that Harbor will NOT retry).  These are always emitted as generate/ metrics
        # so that dashboards see a consistent zero-baseline time-series.
        self._tracked_exceptions = sorted(self._retry_config.exclude_exceptions or ())

        logger.info(
            f"HarborTrajectoryRunner initialized with HarborConfigBuilder. "
            f"Exposed fields: {list(self._harbor_config_builder._harbor_cfg.keys())}. "
            f"Retry config: max_retries={self._retry_config.max_retries}, "
            f"backoff={self._retry_config.min_wait_sec}-{self._retry_config.max_wait_sec}s. "
            f"Concurrent trials: {self._n_concurrent_trials}. "
            f"Reward shaping: enabled={self._reward_shaping_config.get('enable_reward_shaping', True)}, "
            f"shaper={self._reward_shaping_config.get('reward_shaper', 'pass_ratio')}. "
            f"Error classification: enabled={self._error_handling_config.enable_error_classification}"
        )

        # Read custom chat template
        custom_chat_template_path = trajectory_runner_cfg.engine_init_kwargs.get(
            "custom_chat_template_chat_completion_path", None
        )
        if custom_chat_template_path:
            with open(custom_chat_template_path, "r") as f:
                self.custom_chat_template_content = f.read()
            logger.info(
                f"HarborTrajectoryRunner initialized with custom chat template read from: {custom_chat_template_path}"
            )
        else:
            self.custom_chat_template_content = None

        # --- ARCH-GATED qwen3_5/3.6 thinking-enable for the re-tokenize / TIS path ---
        # The Qwen3.5/3.6 chat template's DEFAULT generation prompt (enable_thinking
        # unset/false) auto-injects an EMPTY ``<think>\n\n</think>`` block, which the
        # served completion_token_ids never contain -> TIS prefix divergence (the
        # 4b76d41 splice exists to paper over exactly this). The ROOT fix is to render
        # the generation prompt WITH thinking (``enable_thinking=True``), mirroring the
        # eval path + the served vLLM request (harbor terminus-2 ``extra_body``):
        # ``enable_thinking=True`` renders the bare ``<think>`` open, matching the
        # served stream, so the re-tokenized prefix matches naturally and the splice
        # goes INERT (a clean backstop).
        #
        # Arch-gating mirrors the SPLICE's own tokenizer-level predicate
        # (detect_qwen3_5_empty_think_prefix): we set enable_thinking=True ONLY for a
        # tokenizer+template whose DEFAULT generation prompt renders the empty-think
        # block. For every non-qwen3_5 / non-empty-think model this stays None ->
        # no chat_template_kwargs forwarded -> byte-identical old behavior.
        self._chat_template_kwargs: Optional[Dict[str, Any]] = None
        try:
            default_gen_prompt_ids = get_generation_prompt_ids(
                self.tokenizer, custom_chat_template=self.custom_chat_template_content
            )
            if detect_qwen3_5_empty_think_prefix(self.tokenizer, default_gen_prompt_ids) is not None:
                self._chat_template_kwargs = {"enable_thinking": True}
                logger.info(
                    "HarborTrajectoryRunner: detected Qwen3.5/3.6 empty-think generation "
                    "prompt -> enabling chat_template_kwargs={'enable_thinking': True} for "
                    "the re-tokenize/TIS path (the served vLLM is enabled via harbor "
                    "extra_body.chat_template_kwargs). The 4b76d41 empty-think splice "
                    "becomes an inert backstop."
                )
        except Exception as e:
            logger.warning(
                f"HarborTrajectoryRunner: qwen3_5 thinking-enable probe failed ({e}); "
                f"leaving chat_template_kwargs unset (the empty-think splice still applies)."
            )

        # Shared QueueOrchestrator state (initialized in startup())
        # This ensures all concurrent run() calls share a single orchestrator
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
        """Initialize shared QueueOrchestrator for all run() calls.

        This creates a single orchestrator that enforces the n_concurrent_trials
        limit globally across all async workers. Without this, each run()
        call would create its own orchestrator, multiplying the concurrency limit.

        Called once by the trainer before the first run() call.
        """
        self._orchestrator_lock = asyncio.Lock()
        await self._create_orchestrator()
        logger.info(
            f"HarborTrajectoryRunner startup complete. "
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
        concurrent run() calls. Other callers wait for the restart to complete.

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
                    f"Max orchestrator restart attempts ({MAX_ORCHESTRATOR_RESTART_ATTEMPTS}) exceeded. Giving up."
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

        Called once by the trainer after the last run() call.
        Safe to call multiple times (idempotent).
        """
        if self._orchestrator_lock is None:
            # startup() was never called
            return

        async with self._orchestrator_lock:
            if self._orchestrator is not None and self._orchestrator_started:
                # Mark as not started BEFORE shutdown to prevent race condition.
                # This ensures concurrent run() calls fail fast rather than
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

            # Create unique trials directory for this eval session.
            #
            # Join as a RAW string, not pathlib.Path: pathlib collapses the "//" in a
            # remote URI (Path("s3://bucket/k") -> "s3:/bucket/k"), which would corrupt
            # the eval trials_dir Harbor receives AND make the mkdir below create a bogus
            # LOCAL "s3:/..." directory. Only mkdir for local paths — object stores
            # (s3://, gs://) create the prefix implicitly on first write, exactly as the
            # training trials_dir path does (it is never pre-mkdir'd). Mirrors the
            # raw-string handling in HarborConfigBuilder.build_trial_config.
            if self.trials_dir:
                self._eval_trials_dir = f"{self.trials_dir.rstrip('/')}/eval_sessions/{self._eval_session_name}"
                if "://" not in self.trials_dir:
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
        *,
        exclude_from_baseline: bool = True,
    ) -> TrajectoryBatch:
        """Create a TrajectoryBatch where all trajectories share one terminal failure."""
        num_trials = len(trajectory_ids)
        verification = VerificationResult.unavailable(exception_type)
        reward = _zero_reward()
        disposition = TrainingDisposition(
            loss_eligible=False,
            baseline_eligible=not exclude_from_baseline,
            reason=exception_type,
            exception_type=exception_type,
        )
        return {
            "prompt_token_ids": [[0] for _ in range(num_trials)],
            "response_ids": [[0] for _ in range(num_trials)],
            "rewards": [float(reward.optimization_reward) for _ in range(num_trials)],
            "unshaped_rewards": [float(verification.score or 0.0) for _ in range(num_trials)],
            "loss_masks": [[0] for _ in range(num_trials)],
            "stop_reasons": ["error" for _ in range(num_trials)],
            "rollout_metrics": {
                **get_batch_failure_metrics(
                    num_trials,
                    num_failed_trajectories=num_trials,
                    num_failed_instances=num_trials,
                    num_masked_trajectories=num_trials if exclude_from_baseline else 0,
                ),
                f"{BATCH_ERROR_METRIC_PREFIX}{exception_type}": num_trials,
            },
            "rollout_logprobs": None,
            "exclude_from_baseline": [not disposition.baseline_eligible for _ in range(num_trials)],
        }

    async def _run(self, input_batch: TrajectoryRequestBatch, disable_tqdm: bool = False) -> TrajectoryBatch:
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
        # Record current global_step at the moment we enter run(). Used as
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
                api_base=self._agent_api_base,
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
                        "QueueOrchestrator not available. Was startup() called? Attempting emergency restart..."
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
                logger.error("Orchestrator restart failed. Returning all-failed output to avoid killing training job.")
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
                    f"Retry after orchestrator restart also failed: {type(retry_error).__name__}: {retry_error}"
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
            # into a deterministic job kill. Catch it here, on the trajectory-worker
            # side, BEFORE it can cross the Ray boundary: classify it (mask by
            # default — it's an infrastructure/serialization-class failure,
            # excluded from the RLOO-N baseline) and emit an error output, exactly
            # like the orchestrator/exception paths above.
            try:
                output = self._process_trial_result(result, trajectory_id)
            except Exception as process_error:
                treatment, exception_type = self._classify_exception(process_error)
                # A processing-time render error has no verifier reward to pass
                # through, so only an agent-category error remains in the baseline.
                exclude_from_baseline = treatment_excludes_from_baseline(treatment, verifier_available=False)
                logger.warning(
                    f"Trajectory {trajectory_id} failed during result processing "
                    f"(NOT fatal — skipping this trial): "
                    f"{type(process_error).__name__}: {process_error} "
                    f"(exception_type={exception_type}, "
                    f"exclude_from_baseline={exclude_from_baseline})"
                )
                output = _failed_agent_output(
                    trajectory_id=trajectory_id,
                    verification=VerificationResult.error(
                        "trajectory result processing failed",
                        diagnostics={"exception_type": exception_type},
                    ),
                    exception_type=exception_type,
                    exclude_from_baseline=bool(exclude_from_baseline),
                )
            all_outputs.append(output)

        # For a group of trajectories (n_samples_per_prompt trajectories for the same prompt):
        # - If error classification is DISABLED: if ANY trajectory fails, zero ALL trajectories in the group
        # - If error classification is ENABLED (RLOO-N mode):
        #   - Infrastructure failures (exclude_from_baseline=True): mark for exclusion from baseline
        #   - Agent failures (exclude_from_baseline=False): include in baseline with reward=0
        #   - If ALL trajectories in a group fail, they all get excluded from baseline
        enable_error_classification = self._error_handling_config.enable_error_classification

        failed_instance_ids = set()
        num_failed_trajectories = 0  # per-trajectory, rather than per-instance
        num_masked_trajectories = 0  # trajectories excluded from baseline
        successful_outputs: List[TerminalBenchAgentOutput] = []  # only for metrics purpose

        # Track failure types per instance for RLOO-N
        instance_has_infra_failure: Dict[str, bool] = {}
        instance_has_agent_failure: Dict[str, bool] = {}

        for output in all_outputs:
            if output.evidence.stop_reason == "error":
                failed_instance_ids.add(output.trajectory_id.instance_id)
                num_failed_trajectories += 1
                if not output.disposition.baseline_eligible:
                    num_masked_trajectories += 1
                    instance_has_infra_failure[output.trajectory_id.instance_id] = True
                else:
                    instance_has_agent_failure[output.trajectory_id.instance_id] = True

        if enable_error_classification:
            # RLOO-N mode: preserve exclude_from_baseline flags, don't cascade failures
            for output in all_outputs:
                if output.evidence.stop_reason == "error":
                    # Error outputs already have correct exclude_from_baseline set
                    _clear_failed_trajectory(output)
                else:
                    successful_outputs.append(output)
        else:
            # Legacy mode: if any trajectory fails, zero entire group
            for output in all_outputs:
                if output.trajectory_id.instance_id in failed_instance_ids:
                    _clear_failed_trajectory(output)
                    output.disposition = replace(output.disposition, baseline_eligible=True)
                else:
                    successful_outputs.append(output)

        # Calculate rollout metrics for successful outputs
        if len(successful_outputs) > 0:
            rollout_metrics = get_rollout_metrics(
                [list(output.evidence.response_token_ids) for output in successful_outputs],
                [output.reward_result.optimization_reward for output in successful_outputs],
            )
            rollout_metrics["generate/trajectories_summarized"] = sum(
                1 for output in successful_outputs if output.summarization_count > 0
            )
            rollout_metrics["generate/trajectories_truncated"] = sum(
                1 for output in successful_outputs if output.evidence.stop_reason == "length"
            )
            num_successful = len(successful_outputs)
            num_truncated = sum(1 for o in successful_outputs if o.evidence.stop_reason == "length")
            rollout_metrics["generate/truncated_fraction"] = (
                num_truncated / num_successful if num_successful > 0 else 0.0
            )
            rollout_metrics["generate/truncation_penalty_applied"] = sum(
                1 for o in successful_outputs if o.truncation_penalized
            )
            # Output-length percentiles and their ratio: a healthy arm's
            # p90/p25 widens as it lengthens; a collapsing arm's narrows
            # toward the cap. Moves before the distribution finishes
            # collapsing, so it flags the failure earlier than a cap-hit count.
            tok_lengths = [len(o.evidence.response_token_ids) for o in successful_outputs]
            if tok_lengths:
                tok_arr = np.array(tok_lengths)
                p25 = float(np.percentile(tok_arr, 25))
                p90 = float(np.percentile(tok_arr, 90))
                rollout_metrics["generate/out_tok_p25"] = p25
                rollout_metrics["generate/out_tok_p90"] = p90
                rollout_metrics["generate/out_tok_p90_p25_ratio"] = p90 / p25 if p25 > 0 else 0.0
        else:
            rollout_metrics = {}
        rollout_metrics.update(
            get_batch_failure_metrics(
                num_trials,
                num_failed_trajectories=num_failed_trajectories,
                num_failed_instances=len(failed_instance_ids),
                num_masked_trajectories=num_masked_trajectories,
            )
        )

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
            align_metrics = batch_align.as_metrics(
                prefix=TIS_METRIC_PREFIX, lcs_alert_threshold=self._tis_lcs_alert_threshold
            )
            rollout_metrics.update(align_metrics)
            if batch_align.n_lcs_messages > 0 or batch_align.n_failed_messages > 0:
                # Metered LCS defensive guard: escalate to ERROR when the alert metric
                # trips (lcs_fallback_fraction over the configured threshold),
                # else WARNING. Under full TITO this should never fire.
                alert = align_metrics.get(TIS_LCS_FALLBACK_ALERT_METRIC, 0.0) >= 1.0
                log_fn = logger.error if alert else logger.warning
                log_fn(
                    f"TIS alignment{' ALERT' if alert else ''}: "
                    f"{batch_align.n_exact}/{batch_align.n_tokens} tokens exact, "
                    f"{batch_align.n_lcs} via LCS fallback, {batch_align.n_unaligned} unaligned; "
                    f"{batch_align.n_lcs_messages} LCS-fallback messages, "
                    f"{batch_align.n_failed_messages} failed messages "
                    f"(of {batch_align.n_messages} assistant messages). "
                    f"Non-zero LCS/failure means serving↔training tokenizer divergence"
                    + (
                        f" ABOVE the {self._tis_lcs_alert_threshold} alert threshold — investigate before trusting TIS."
                        if alert
                        else "."
                    )
                )

        # Per-step error counters for tracked exception types.
        # Pre-populate with zeros so every configured exception appears as a
        # consistent time-series on dashboards, then overlay actual counts.
        for exc_type in self._tracked_exceptions:
            rollout_metrics[f"{BATCH_ERROR_METRIC_PREFIX}{exc_type}"] = 0

        exception_counts: Dict[str, int] = {}
        for output in all_outputs:
            if output.disposition.exception_type:
                exception_type = output.disposition.exception_type
                exception_counts[exception_type] = exception_counts.get(exception_type, 0) + 1
        if exception_counts:
            logger.info(f"Exception breakdown: {exception_counts}")
            for exc_type, count in exception_counts.items():
                rollout_metrics[f"{BATCH_ERROR_METRIC_PREFIX}{exc_type}"] = count

        log_fn = logger.warning if num_failed_trajectories else logger.info
        log_fn(
            f"Batch generation complete: {num_trials - num_failed_trajectories}/{num_trials} successful, "
            f"{len(failed_instance_ids)} failed instances, "
            f"{num_masked_trajectories} masked (excluded from baseline)"
        )

        # Collect rollout_logprobs if any outputs have them.
        # For zeroed/failed trajectories, use [0.0] to match response_ids length.
        #
        # Skip logprobs for eval: policy objectives do not run during evaluation.
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
            has_any_logprobs = any(output.evidence.behavior_logprobs is not None for output in all_outputs)
            missing_logprobs_count = sum(1 for output in all_outputs if output.evidence.behavior_logprobs is None)

            if has_any_logprobs:
                rollout_logprobs_list = []
                for output in all_outputs:
                    if output.evidence.behavior_logprobs is not None:
                        rollout_logprobs_list.append(list(output.evidence.behavior_logprobs))
                    else:
                        if self._rollout_logprobs_required and any(output.loss_mask):
                            raise ValueError("rollout_logprobs are required for every trainable trajectory")
                        # Failed trajectories are fully masked, so aligned placeholders
                        # cannot affect the objective.
                        rollout_logprobs_list.append([0.0] * len(output.evidence.response_token_ids))

                if missing_logprobs_count > 0 and self._collect_rollout_details:
                    # Only warn about missing logprobs if TIS is expected (collect_rollout_details=true)
                    logger.warning(
                        f"Rollout-logprob mode: {missing_logprobs_count}/{num_trials} trajectories missing logprobs "
                        f"(likely due to context length errors). Filled with zeros. "
                        f"These trajectories will have no gradient contribution from TIS."
                    )
            elif self._rollout_logprobs_required and any(any(output.loss_mask) for output in all_outputs):
                raise ValueError("rollout_logprobs are required for every trainable trajectory")
            elif missing_logprobs_count > 0 and self._collect_rollout_details:
                # All trajectories missing logprobs - this is a problem for TIS
                # Only log error if TIS is expected (collect_rollout_details=true)
                logger.error(
                    f"Rollout-logprob mode: ALL {num_trials} trajectories missing logprobs. "
                    f"This batch cannot use a behavior-referenced policy objective. "
                    f"Check if Harbor is collecting rollout_details (collect_rollout_details=true) "
                    f"and if context length errors are preventing logprob collection."
                )

        # Collect routed_experts (Stage 1 MoE router-replay capture rail). Mirrors
        # the rollout_logprobs gather and mixed-presence handling. Gated on
        # moe_router_replay so the TrajectoryBatch is byte-identical when off (the
        # key is omitted entirely, not set to None). Skipped for eval like logprobs.
        rollout_routed_experts_list = None
        if self._moe_router_replay and not is_eval:
            has_any_routed_experts = any(output.evidence.routed_experts is not None for output in all_outputs)
            if has_any_routed_experts:
                # Learn the [L, K] sentinel-row shape from the first real sample so
                # missing/failed samples are sentinel-filled at the correct width.
                sentinel_row = [[SENTINEL_EXPERT_ID]]
                for output in all_outputs:
                    if output.evidence.routed_experts:
                        sentinel_row = _sentinel_routed_experts_row(output.evidence.routed_experts[0])
                        break
                rollout_routed_experts_list = []
                for output in all_outputs:
                    if output.evidence.routed_experts is not None:
                        rollout_routed_experts_list.append(
                            [[list(layer) for layer in token] for token in output.evidence.routed_experts]
                        )
                    else:
                        # Sentinel-fill missing samples to match response_ids length.
                        rollout_routed_experts_list.append(
                            [list(sentinel_row) for _ in range(len(output.evidence.response_token_ids))]
                        )

        # Collect the Stage B per-token shaping channel + span tags. Gated on
        # enable_token_reward_channel so the TrajectoryBatch is byte-identical when
        # off (keys omitted entirely). Sentinel-fill (zeros) any sample missing them
        # so the lists stay 1:1 with response_ids. Emitted for train AND eval is
        # harmless (channel is zeros), but mirror the routed_experts train-only gate
        # to keep eval batches identical.
        token_level_shaping_list = None
        response_span_tags_list = None
        if self._enable_token_reward_channel and not is_eval:
            if any(output.reward_result.token_credit is not None for output in all_outputs):
                token_level_shaping_list = [
                    list(output.reward_result.token_credit)
                    if output.reward_result.token_credit is not None
                    else [0.0] * len(output.evidence.response_token_ids)
                    for output in all_outputs
                ]
            if any(output.response_span_tags is not None for output in all_outputs):
                response_span_tags_list = [
                    output.response_span_tags
                    if output.response_span_tags is not None
                    else [0] * len(output.evidence.response_token_ids)
                    for output in all_outputs
                ]

        # Aggregate per-component reward metrics (composite shaper only)
        component_outputs = [output for output in all_outputs if output.reward_result.components]
        if component_outputs:
            component_names = set()
            for o in component_outputs:
                component_names.update(o.reward_result.components.keys())
            component_metrics = {}
            for name in sorted(component_names):
                values = [o.reward_result.components.get(name, 0.0) for o in component_outputs]
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
        # the global_step captured at run() entry (worst case = same as
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

        trajectory_batch: TrajectoryBatch = {
            "prompt_token_ids": [list(output.evidence.prompt_token_ids) for output in all_outputs],
            "response_ids": [list(output.evidence.response_token_ids) for output in all_outputs],
            "rewards": [output.reward_result.optimization_reward for output in all_outputs],
            "unshaped_rewards": [float(output.reward_result.unshaped_reward or 0.0) for output in all_outputs],
            "loss_masks": [
                project_loss_mask(output, list(output.evidence.response_token_ids)) for output in all_outputs
            ],
            "stop_reasons": [output.evidence.stop_reason for output in all_outputs],
            "rollout_metrics": rollout_metrics,
            "rollout_logprobs": rollout_logprobs_list,
            "exclude_from_baseline": [not output.disposition.baseline_eligible for output in all_outputs],
            "actual_global_step": actual_global_step,
        }

        # Only attach routed_experts when router-replay is on, so the flag-off
        # TrajectoryBatch dict is byte-identical to today (key absent, not None).
        if rollout_routed_experts_list is not None:
            trajectory_batch["rollout_routed_experts"] = rollout_routed_experts_list

        # Attach the Stage B channel + tags only when present, so the flag-off
        # TrajectoryBatch dict is byte-identical to today (keys absent, not None).
        if token_level_shaping_list is not None:
            trajectory_batch["token_level_shaping"] = token_level_shaping_list
        if response_span_tags_list is not None:
            trajectory_batch["response_span_tags"] = response_span_tags_list

        return trajectory_batch

    def _classify_exception(self, exception: Exception) -> tuple[ErrorTreatment, str]:
        """Classify an exception and return its treatment and persisted type name."""
        exception_type = type(exception).__name__
        return self._classify_exception_type(exception_type)

    def _classify_exception_type(self, exception_type: str) -> tuple[ErrorTreatment, str]:
        """Classify a persisted Harbor exception type name."""

        # If error classification is disabled, treat all errors as agent failures
        if not self._error_handling_config.enable_error_classification:
            return ErrorTreatment.ZERO, exception_type

        treatment = classify_exception_type(exception_type, self._error_handling_config)
        return treatment, exception_type

    def _should_preserve_timeout_trajectory(self, result) -> bool:
        """Whether to KEEP a fully-generated trajectory whose POST-generation step
        failed (VerifierTimeoutError -> no reward, or a soft AgentTimeout without a
        verifier result) instead of discarding it.

        Only True when preserve-on-timeout is enabled AND the agent actually
        produced ``rollout_details`` carrying logprobs -- so we never synthesize a
        training sample from a trial that never generated. This is a cheap
        prerequisite gate; the AUTHORITATIVE TIS-validity check (``rollout_logprobs``
        length == ``response_ids`` length) still runs after extraction and falls
        back to the discard stub on any mismatch, so malformed / length-mismatched
        logprobs are never fed to TIS.
        """
        if not self._preserve_logprobs_on_timeout:
            return False
        # Only in RLOO-N error-classification mode; legacy group-zeroing mode
        # (classification off) keeps its byte-identical behavior.
        if not self._error_handling_config.enable_error_classification:
            return False
        agent_result = getattr(result, "agent_result", None)
        rollout_details = getattr(agent_result, "rollout_details", None)
        if not rollout_details:
            return False
        main = rollout_details[0]
        if not isinstance(main, dict):
            return False
        return bool(main.get("logprobs"))

    def _maybe_correlate_opencode_rollout_details(
        self,
        result: "TrialResult",
        rollout_details: Optional[List[Dict[str, Any]]],
    ) -> Optional[List[Dict[str, Any]]]:
        """Recover a CLI agent's (opencode's) rollout_details from the shared proxy log.

        opencode talks to vLLM over its own transport, bypassing harbor Chat, so its
        ``rollout_details`` is empty even behind a co-located RecordProxy (which writes
        a single shared worker-side log, not the in-sandbox trial dir). Harbor stamped a
        per-trial correlation id (``x-ot-trial-id``) into every request and surfaced it
        on ``context.metadata['rollout_correlation_id']``; the proxy recorded it on each
        log entry. Here we filter the shared log by this trial's id and build its
        RolloutDetail (id-based, so concurrent GRPO same-seed trials never bleed).

        No-op (returns the input unchanged) when: rollout_details already populated
        (terminus native), collect_rollout_details off, no correlation id present, or no
        proxy log path published (``OTAGENT_LITERAL_LOG_PATH``). Never synthesizes.
        """
        if rollout_details:
            return rollout_details
        if not self._collect_rollout_details:
            return rollout_details
        agent_result = getattr(result, "agent_result", None)
        metadata = getattr(agent_result, "metadata", None)
        trial_id = metadata.get("rollout_correlation_id") if isinstance(metadata, dict) else None
        if not trial_id:
            return rollout_details
        # cfg-threaded path (Ray-boundary-safe) with an env fallback for direct launches.
        log_path = self._literal_log_path or os.environ.get("OTAGENT_LITERAL_LOG_PATH")
        if not log_path:
            return rollout_details
        # Per-trial indexed lookup (O(this trial's entries)). The RecordProxy also
        # sees auxiliary OpenCode calls and context-reset sessions under the same
        # trial header, so retain only the final continuous agent-call route before
        # constructing the rollout detail used by TIS/TITO.
        entries = self._literal_log_store.entries_for_trial(log_path, trial_id)
        # This is the SECOND (last) opencode literal consumer for the trial —
        # _maybe_build_opencode_chat_history already ran earlier in _process_trial_result.
        # Once both have consumed this trial's rows, release them so the shared-log store
        # stays O(in-flight trials) instead of retaining every row for the whole run (a
        # long high-concurrency opencode RL run's shared log is many GB). release_trial is
        # idempotent and also fences out any late orphaned-opencode appends for this trial.
        try:
            if not entries:
                return rollout_details
            try:
                from harbor.literal.rollout_build import build_rollout_details_from_pairs
            except Exception:  # harbor without the bridge → no-op
                return rollout_details
            selected_entries = _select_opencode_literal_chain(entries, trial_id)
            built = build_rollout_details_from_pairs([entry["literal"] for entry in selected_entries])
            if not built:
                return rollout_details
            if len(selected_entries) != len(entries):
                logger.info(
                    f"[literal-bridge] selected {len(selected_entries)}/{len(entries)} continuous opencode "
                    f"turn(s) for trial {trial_id}"
                )
            # Persist onto the result so downstream consumers see a consistent view.
            try:
                result.agent_result.rollout_details = built
            except Exception:
                pass
            n_turns = len(built[0].get("completion_token_ids", []))
            logger.info(
                f"[literal-bridge] correlated {n_turns} opencode turn(s) for trial {trial_id} from shared proxy log"
            )
            return built
        finally:
            self._literal_log_store.release_trial(trial_id)

    def _maybe_build_opencode_chat_history(
        self,
        result: "TrialResult",
    ) -> Optional[List[Dict[str, Any]]]:
        """Reconstruct a CLI agent's (opencode's) chat_history from the shared proxy log.

        ``_process_trial_result`` reads ``chat_history`` from
        ``agent_result.metadata['all_messages']``, which ONLY the harbor terminus-2
        agent sets (it drives harbor ``Chat``). opencode is a CLI agent that talks to
        vLLM over its own transport and bypasses harbor ``Chat``, so it never populates
        ``all_messages`` → the terminus path ``KeyError``\\ s and every opencode
        trajectory is dropped with no logprobs (TIS then degrades to non-TIS on 100% of
        the batch). We instead recover the served conversation from the SAME shared
        RecordProxy log the sibling :meth:`_maybe_correlate_opencode_rollout_details`
        uses, filtered by this trial's correlation id.

        The reconstruction: take the LAST (latest-timestamp) matched request's
        ``messages`` — opencode sends the FULL accumulated conversation as the prompt of
        each call, so the final call carries every prior system/user/assistant/tool
        message — and append the final turn's assistant completion (decoded from its
        exact ``completion_token_ids``). The returned list is a standard
        ``[{role, content}, ...]`` conversation; the caller feeds it through the same
        prompt/response tokenization + per-turn logprob alignment as terminus. Alignment
        is NOT trusted blindly: the exact-id path (``assistant_token_ids``) +
        LCS-fallback + ``AlignmentStats`` downstream flag any per-turn mismatch as
        ``tis/lcs_fallback_fraction`` rather than silently corrupting training.

        Returns ``None`` (caller then drops the trajectory honestly, exactly as the
        pre-existing ``KeyError`` branch did) when: rollout-detail collection is off, no
        correlation id is present, no proxy-log path is published
        (``OTAGENT_LITERAL_LOG_PATH``), no matching completion-bearing record exists, or
        the recovered request carries no usable ``messages``. Never synthesizes content.
        """
        if not self._collect_rollout_details:
            return None
        agent_result = getattr(result, "agent_result", None)
        metadata = getattr(agent_result, "metadata", None)
        trial_id = metadata.get("rollout_correlation_id") if isinstance(metadata, dict) else None
        if not trial_id:
            return None
        # cfg-threaded path (Ray-boundary-safe) with an env fallback for direct launches.
        log_path = self._literal_log_path or os.environ.get("OTAGENT_LITERAL_LOG_PATH")
        if not log_path:
            return None
        # Per-trial indexed lookup (O(this trial's entries)); already trial-scoped, so the
        # comprehension only applies the remaining status-200 / completion-bearing filter.
        entries = self._literal_log_store.entries_for_trial(log_path, trial_id)
        if not entries:
            return None
        # Same selection as build_rollout_details_for_trial (trial-scoped, status-200,
        # completion-bearing) so chat_history and rollout_details are built from the
        # IDENTICAL turn set — the final turn's completion becomes the last assistant
        # message, keeping the two views consistent.
        matched = [
            e
            for e in entries
            if e.get("trial_id") == trial_id
            and e.get("status_code") == 200
            and isinstance(e.get("literal"), dict)
            and e["literal"].get("completion_token_ids")
        ]
        if not matched:
            return None
        matched = _select_opencode_literal_chain(matched, trial_id)
        if not matched:
            return None
        last = matched[-1]
        request = last.get("request")
        base_messages = request.get("messages") if isinstance(request, dict) else None
        if not isinstance(base_messages, list) or not base_messages:
            return None
        # opencode sends OpenAI-shaped messages: ``content`` may be a LIST of parts and
        # assistant turns may carry structured ``tool_calls`` (plus ``tool``-role results).
        # The downstream re-tok+splice fallback in
        # ``get_response_ids_and_loss_mask_from_messages`` runs ``apply_chat_template``,
        # which can only render plain ``{role, content:str}`` messages — the raw opencode
        # shape makes it raise ``TypeError: Can only get item pairs from a mapping``, which
        # the caller classifies as a zero-reward trajectory (silently starved ~87% of the
        # keep1-v24 batch to ``response_ids=[0]`` / reward 0). Normalize each kept message
        # via harbor's canonical ``normalize_message`` (flattens list content, serializes
        # tool_calls; reuses ``normalize_message_content`` so the ShareGPT datagen path is
        # untouched). Guard against a malformed record (missing/non-str role) rather than
        # crash.
        chat_history: List[Dict[str, Any]] = [
            normalize_message(m) for m in base_messages if isinstance(m, dict) and isinstance(m.get("role"), str)
        ]
        if not chat_history:
            return None
        # Append the final turn's assistant message, decoded from its EXACT served
        # completion ids (round-trips to the same ids the exact-alignment path consumes).
        final_completion_ids = last["literal"].get("completion_token_ids") or []
        final_text = ""
        if final_completion_ids:
            try:
                final_text = self.tokenizer.decode(final_completion_ids, skip_special_tokens=True)
            except Exception:
                final_text = ""
        chat_history.append({"role": "assistant", "content": final_text})
        logger.info(
            f"[literal-bridge] reconstructed opencode chat_history for trial {trial_id}: "
            f"{len(chat_history)} message(s) from the shared proxy log"
        )
        return chat_history

    def _shape_harbor_reward(
        self,
        result: TrialResult,
        verification: VerificationResult,
        chat_history: List[Dict[str, Any]],
        trajectory_id: TrajectoryID,
        *,
        preserve_timeout: bool,
    ) -> RewardResult:
        """Adapt a Harbor verdict and configured shaper to the shared reward contract."""
        if preserve_timeout:
            return RewardResult(unshaped_reward=None, optimization_reward=0.0)

        if verification.score is None:
            raise ValueError("verified Harbor results require a score")
        original_reward = verification.score
        reward = original_reward
        reward_components: Optional[Dict[str, float]] = None
        if self._reward_shaping_config.get("enable_reward_shaping", True):
            verifier_stdout = getattr(result.verifier_result, "stdout", None)
            shaper_name = self._reward_shaping_config.get("reward_shaper", "pass_ratio")
            shaper_kwargs = self._reward_shaping_config.get("shaper_kwargs", {})
            if shaper_name in ("composite", "composite_loop"):
                trajectory_context = (
                    detect_termination_signals(chat_history, original_reward)
                    if shaper_name == "composite_loop"
                    else None
                )
                reward, reward_components = shape_reward_with_components(
                    stdout=verifier_stdout,
                    original_reward=original_reward,
                    parser_name=self._reward_shaping_config.get("reward_parser"),
                    shaper_kwargs=shaper_kwargs,
                    chat_history=chat_history,
                    shaper_name=shaper_name,
                    trajectory_context=trajectory_context,
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
        return RewardResult(
            unshaped_reward=original_reward,
            optimization_reward=reward,
            components={} if reward_components is None else reward_components,
        )

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
            treatment, exception_type = self._classify_exception(result)
            exclude_from_baseline = treatment_excludes_from_baseline(treatment, verifier_available=False)
            logger.warning(
                f"Trajectory {trajectory_id} failed with exception: {result} "
                f"(type={exception_type}, exclude_from_baseline={exclude_from_baseline})"
            )
            return _failed_agent_output(
                trajectory_id=trajectory_id,
                verification=VerificationResult.error(
                    "trajectory orchestration failed",
                    diagnostics={"exception_type": exception_type},
                ),
                exception_type=exception_type,
                exclude_from_baseline=exclude_from_baseline,
            )

        verification = verification_from_harbor_result(result)

        # Preserve-on-soft-timeout state (see _should_preserve_timeout_trajectory).
        # When set, a POST-generation failure (no verifier reward) does NOT discard
        # the generated trajectory: we fall through to build it at reward=0 and gate
        # on TIS-valid logprobs before returning it. Default-off values keep the
        # non-preserve paths byte-identical.
        preserve_timeout = False
        preserve_exclude_from_baseline = False
        preserve_exception_type: Optional[str] = None

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

            treatment, _ = self._classify_exception_type(exception_type)

            # Passthrough: ignore the exception and fall through to normal
            # verifier processing. The agent hit a soft limit (timeout, context
            # length) but the verifier still ran and produced a real reward.
            if treatment is ErrorTreatment.PASSTHROUGH:
                missing_logprobs_error = (
                    passthrough_logprob_error_type(
                        treatment,
                        has_rollout_logprobs=extract_logprobs_from_rollout_details(
                            getattr(result.agent_result, "rollout_details", None)
                        )
                        is not None,
                        rollout_logprobs_required=self._rollout_logprobs_required,
                    )
                    if verification.status is VerificationStatus.VERIFIED
                    else None
                )
                if missing_logprobs_error is not None:
                    logger.warning(
                        f"Trajectory {trajectory_id}: {exception_type} classified as PASSTHROUGH but has no "
                        "behavior logprobs; masking it from behavior-referenced training"
                    )
                    return _failed_agent_output(
                        trajectory_id=trajectory_id,
                        verification=verification,
                        exception_type=missing_logprobs_error,
                        exclude_from_baseline=True,
                    )
                if verification.status is VerificationStatus.VERIFIED:
                    logger.info(
                        f"Trajectory {trajectory_id}: {exception_type} classified as PASSTHROUGH, using verifier reward"
                    )
                    # Fall through to normal processing below
                elif self._should_preserve_timeout_trajectory(result):
                    # Passthrough w/o verifier result, but the agent produced a full
                    # trajectory with logprobs. Keep it at reward=0 (excluded from the
                    # baseline) instead of discarding, so the RLOO-N group survives.
                    preserve_timeout = True
                    preserve_exclude_from_baseline = True
                    preserve_exception_type = exception_type
                    logger.info(
                        f"Trajectory {trajectory_id}: {exception_type} PASSTHROUGH w/o verifier "
                        f"result — preserving generated logprobs at reward=0 (excluded from baseline)"
                    )
                else:
                    # Passthrough requested but no verifier result and no usable
                    # generation — treat as mask.
                    logger.warning(
                        f"Trajectory {trajectory_id}: {exception_type} classified as PASSTHROUGH "
                        f"but no verifier result available, masking instead"
                    )
                    return _failed_agent_output(
                        trajectory_id=trajectory_id,
                        verification=verification,
                        exception_type=exception_type,
                        exclude_from_baseline=True,
                    )
            else:
                exclude_from_baseline = treatment_excludes_from_baseline(
                    treatment, verifier_available=verification.status is VerificationStatus.VERIFIED
                )
                if self._should_preserve_timeout_trajectory(result):
                    # POST-generation failure classified ZERO/MASK (e.g.
                    # VerifierTimeoutError). The agent generated a full trajectory
                    # with logprobs; keep it at reward=0 with this exclude flag so the
                    # RLOO-N group is not dropped for all-missing-logprobs.
                    preserve_timeout = True
                    preserve_exclude_from_baseline = exclude_from_baseline
                    preserve_exception_type = exception_type
                    logger.info(
                        f"Trajectory {trajectory_id}: {exception_type} (exclude_from_baseline="
                        f"{exclude_from_baseline}) — preserving generated logprobs at reward=0"
                    )
                else:
                    logger.warning(
                        f"Trajectory {trajectory_id} failed with Harbor exception: "
                        f"{exception_info.exception_message if hasattr(exception_info, 'exception_message') else exception_info} "
                        f"(type={exception_type}, exclude_from_baseline={exclude_from_baseline})"
                    )
                    return _failed_agent_output(
                        trajectory_id=trajectory_id,
                        verification=verification,
                        exception_type=exception_type,
                        exclude_from_baseline=exclude_from_baseline,
                    )

        # Check for missing verifier result (trial ran but didn't produce valid output)
        # Note: exception_info is already handled above, so if we reach here it's None.
        # A preserved timeout trajectory legitimately has no verifier_result -> skip.
        if verification.status is not VerificationStatus.VERIFIED and not preserve_timeout:
            logger.warning(
                f"Trajectory {trajectory_id} failed: No verifier result and no exception info. "
                f"This is unexpected - marking as infrastructure failure."
            )
            return _failed_agent_output(
                trajectory_id=trajectory_id,
                verification=verification,
                exception_type="MissingVerifierResult",
                exclude_from_baseline=True,  # Infrastructure issue - exclude from baseline
            )

        # Extract data from successful trial. A preserved timeout trajectory has no
        # verifier reward -> reward 0.0, but its generated chat_history/logprobs are
        # extracted exactly like a successful trial.
        try:
            metadata = result.agent_result.metadata
            # terminus (harbor Chat) publishes the conversation on
            # metadata['all_messages']; opencode (a CLI agent that bypasses harbor Chat)
            # does not, so for it we recover chat_history from the shared RecordProxy
            # log by correlation id (the SAME source as the per-turn logprobs below).
            # Branch on all_messages PRESENCE, not agent name, so the terminus path
            # stays byte-identical and any future Chat-driven agent keeps working.
            if isinstance(metadata, dict) and "all_messages" not in metadata:
                chat_history = self._maybe_build_opencode_chat_history(result)
                if not chat_history:
                    # No recoverable conversation → drop the trajectory honestly,
                    # exactly as the pre-existing KeyError branch did (no silent
                    # fabrication; TIS flags the trial via the missing-logprobs gate).
                    raise KeyError("all_messages")
                summarization_count = metadata.get("summarization_count", 0)
            else:
                chat_history = metadata["all_messages"]
                summarization_count = metadata["summarization_count"]
        except (KeyError, AttributeError, TypeError) as e:
            # Data extraction failure is typically an infrastructure issue
            exception_type = type(e).__name__
            treatment, _ = self._classify_exception(e)
            exclude_from_baseline = treatment_excludes_from_baseline(treatment, verifier_available=False)
            logger.warning(
                f"Trajectory {trajectory_id} failed: Could not extract results. "
                f"Error: {e}, Result: {result} "
                f"(type={exception_type}, exclude_from_baseline={exclude_from_baseline})"
            )
            return _failed_agent_output(
                trajectory_id=trajectory_id,
                verification=VerificationResult.error(
                    "could not extract Harbor rollout evidence",
                    diagnostics={"exception_type": exception_type},
                ),
                exception_type=exception_type,
                exclude_from_baseline=exclude_from_baseline,
            )

        reward_result = self._shape_harbor_reward(
            result,
            verification,
            chat_history,
            trajectory_id,
            preserve_timeout=preserve_timeout,
        )
        original_reward = reward_result.unshaped_reward or 0.0
        reward = reward_result.optimization_reward

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
                f"Trajectory {trajectory_id} failed: Invalid chat history structure. chat_history: {chat_history}"
            )
            return _failed_agent_output(
                trajectory_id=trajectory_id,
                verification=VerificationResult.error("Harbor returned an invalid conversation"),
                exception_type="InvalidChatHistory",
                exclude_from_baseline=True,  # Infrastructure issue
            )

        # Process successful trial
        # Prompt = system messages (if any) + first user message
        prompt = system_msgs + [conversation[0]]
        prompt_ids = normalize_token_ids(
            self.tokenizer.apply_chat_template(
                prompt,
                add_generation_prompt=False,
                tokenize=True,
                chat_template=self.custom_chat_template_content,
                **(self._chat_template_kwargs or {}),
            )
        )
        initial_prompt_length = len(prompt_ids)

        # Process response messages (everything after the first message)
        response_messages = conversation[1:]

        # Extract per-turn behavior logprobs from Harbor's rollout details.
        rollout_details = getattr(result.agent_result, "rollout_details", None)
        # opencode is a CLI agent that bypasses harbor Chat, so it returns EMPTY
        # rollout_details even under a co-located RecordProxy (the proxy writes a
        # shared worker-side log, not the in-sandbox trial dir). Recover this trial's
        # token_ids/logprobs from that shared log by the per-trial correlation id
        # harbor stamped (x-ot-trial-id). No-op when rollout_details is already
        # populated (terminus native), the flag is off, or no proxy log is present.
        rollout_details = self._maybe_correlate_opencode_rollout_details(result, rollout_details)
        assistant_logprobs = extract_logprobs_from_rollout_details(rollout_details)
        # Exact-alignment ids: Harbor's per-turn completion_token_ids, index-aligned
        # with assistant_logprobs. Enables the exact (no re-tokenization guess) TIS path.
        assistant_token_ids = extract_token_ids_from_rollout_details(rollout_details)
        # Full-TITO context ids: Harbor's per-turn prompt_token_ids (the exact served
        # context stream). When full TITO is selected, this lets the trainer assemble the masked context from exact ids
        # instead of re-tokenizing. None-safe (absent on non-token-id captures).
        assistant_prompt_token_ids = extract_prompt_token_ids_from_rollout_details(rollout_details)
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
        # collation (trajectory_runners/trajectory_processing.py concatenate_trajectory_batches) already tolerates
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
                chat_template_kwargs=self._chat_template_kwargs,
                assistant_prompt_token_ids=assistant_prompt_token_ids,
                rollout_logprobs_required=self._rollout_logprobs_required,
                tito_full=self._tito_full,
                tis_splice=self._tis_splice,
            )
        else:
            response_ids, loss_mask, rollout_logprobs = get_response_ids_and_loss_mask_from_messages(
                response_messages,
                self.tokenizer,
                assistant_logprobs,
                custom_chat_template=self.custom_chat_template_content,
                assistant_token_ids=assistant_token_ids,
                alignment_stats=alignment_stats,
                chat_template_kwargs=self._chat_template_kwargs,
                assistant_prompt_token_ids=assistant_prompt_token_ids,
                rollout_logprobs_required=self._rollout_logprobs_required,
                tito_full=self._tito_full,
                tis_splice=self._tis_splice,
            )

        # Determine stop reason
        max_response_tokens = (
            self.trajectory_runner_cfg.sampling_params.max_generate_length
            + self.trajectory_runner_cfg.max_input_length
            - initial_prompt_length
        )
        stop_reason = "complete"  # Default for trial completion
        if len(response_ids) > max_response_tokens:
            stop_reason = "length"

        max_gen_len = self.trajectory_runner_cfg.sampling_params.max_generate_length
        per_turn_counts = [len(t) for t in assistant_token_ids] if assistant_token_ids else None
        turn_truncated = detect_turn_truncation(per_turn_counts, max_gen_len)

        truncation_penalized = False
        reward, truncation_penalized = apply_truncation_penalty(
            reward, original_reward, turn_truncated, self._truncation_penalty
        )
        if truncation_penalized:
            logger.debug(
                f"Trajectory {trajectory_id}: truncation penalty applied "
                f"({self._truncation_penalty}), reward {original_reward:.3f} -> {reward:.3f}"
            )

        # Loop-behavior reward shaping (Stage B / F5 + F4): when the channel is
        # enabled, emit a per-token shaping vector (ZEROS in Stage B — no-op) and,
        # optionally, the F4 span tags. Both are computed on the SAME response_ids
        # layout (the tagger re-walks the exact per-turn segmentation), so they
        # align 1:1 with the training tokens. None when the channel is off ->
        # byte-identical TrajectoryBatch.
        token_level_shaping: Optional[List[float]] = None
        response_span_tags: Optional[List[int]] = None
        if self._enable_token_reward_channel:
            token_level_shaping = [0.0] * len(response_ids)
            if self._enable_span_tagging:
                try:
                    tags = tag_response_spans(
                        response_messages,
                        self.tokenizer,
                        custom_chat_template=self.custom_chat_template_content,
                        assistant_token_ids=assistant_token_ids,
                    )
                except Exception as e:
                    logger.warning(f"Trajectory {trajectory_id}: span tagging failed ({e}); emitting all-OTHER tags.")
                    tags = [0] * len(response_ids)
                # Length-parity guard: the tagger re-walks the same segmentation,
                # but truncate/pad to response_ids length to stay 1:1 even if a
                # chat-template edge case drifts (mirrors the truncation below).
                if len(tags) < len(response_ids):
                    tags = tags + [0] * (len(response_ids) - len(tags))
                response_span_tags = tags

            # Stage C (F2 + F6): replace the Stage-B zeros with potential-based
            # shaping of the EDIT-token span from the in-trajectory test-delta.
            # Requires span tags (PBS scatters onto SPAN_EDIT tokens). When the
            # trajectory has no recognized test output / no edit turn, PBS returns
            # an all-zero vector -> that trajectory stays pure-RLOO-N. PBS is
            # policy-invariant (Ng 1999) and bounded to ±pbs_max_total_shaping.
            if self._enable_pbs_shaping and response_span_tags is not None:
                try:
                    pbs_vector = compute_pbs_token_shaping(
                        chat_history,
                        response_span_tags,
                        gamma=self._pbs_gamma,
                        max_total_shaping=self._pbs_max_total_shaping,
                        potential_shape=self._pbs_potential_shape,
                    )
                except Exception as e:
                    logger.warning(
                        f"Trajectory {trajectory_id}: PBS shaping failed ({e}); falling back to zeros (outcome-only)."
                    )
                    pbs_vector = [0.0] * len(response_span_tags)
                # Length-parity: pbs_vector is built on response_span_tags length.
                if len(pbs_vector) < len(token_level_shaping):
                    pbs_vector = pbs_vector + [0.0] * (len(token_level_shaping) - len(pbs_vector))
                token_level_shaping = pbs_vector[: len(token_level_shaping)]

        # Truncate to maximum allowed length
        response_ids = response_ids[:max_response_tokens]
        loss_mask = loss_mask[:max_response_tokens]
        if rollout_logprobs is not None:
            rollout_logprobs = rollout_logprobs[:max_response_tokens]
        if rollout_routed_experts is not None:
            rollout_routed_experts = rollout_routed_experts[:max_response_tokens]
        if token_level_shaping is not None:
            token_level_shaping = token_level_shaping[:max_response_tokens]
        if response_span_tags is not None:
            response_span_tags = response_span_tags[:max_response_tokens]

        # Authoritative behavior-logprob gate for a preserved timeout trajectory: keep it
        # ONLY if it produced length-matched logprobs (one logprob per response
        # token). Otherwise fall back to the discard stub (current behavior) so a
        # malformed or length-mismatched artifact is never used as a policy reference.
        if preserve_timeout:
            if rollout_logprobs is None or len(rollout_logprobs) != len(response_ids):
                logger.warning(
                    f"Trajectory {trajectory_id}: {preserve_exception_type} preserve gate FAILED "
                    f"(rollout_logprobs="
                    f"{'None' if rollout_logprobs is None else len(rollout_logprobs)} "
                    f"!= response_ids={len(response_ids)}); discarding for behavior-logprob safety"
                )
                return _failed_agent_output(
                    trajectory_id=trajectory_id,
                    verification=verification,
                    exception_type=preserve_exception_type,
                    exclude_from_baseline=preserve_exclude_from_baseline,
                )
            logger.info(
                f"Trajectory {trajectory_id}: {preserve_exception_type} PRESERVED at reward=0 "
                f"({len(rollout_logprobs)} logprobs kept, TIS-valid)"
            )

        evidence = _rollout_evidence_from_harbor(
            chat_history=chat_history,
            stop_reason=stop_reason,
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            loss_mask=loss_mask,
            rollout_logprobs=rollout_logprobs,
            rollout_routed_experts=rollout_routed_experts,
        )
        reward_result = replace(
            reward_result,
            optimization_reward=reward,
            token_credit=None if token_level_shaping is None else tuple(token_level_shaping),
        )
        reward_result.validate_for(evidence)
        disposition = _completed_disposition(
            preserve_timeout=preserve_timeout,
            preserve_exclude_from_baseline=preserve_exclude_from_baseline,
            preserve_exception_type=preserve_exception_type,
        )
        return TerminalBenchAgentOutput(
            evidence=evidence,
            verification=verification,
            reward_result=reward_result,
            disposition=disposition,
            loss_mask=loss_mask,
            trajectory_id=trajectory_id,
            summarization_count=summarization_count,
            alignment_stats=alignment_stats,
            response_span_tags=response_span_tags,
            truncation_penalized=truncation_penalized,
        )
