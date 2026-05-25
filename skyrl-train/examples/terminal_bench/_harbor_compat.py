"""Harbor orchestrator compatibility shim for SkyRL's terminal-bench generator.

Harbor recently removed the ``harbor.orchestrators`` package entirely:

  - ``harbor.orchestrators.base.OrchestratorEvent`` → ``harbor.trial.hooks.TrialEvent``
    (``TRIAL_COMPLETED`` → ``END``)
  - ``harbor.orchestrators.queue.QueueOrchestrator`` → ``harbor.trial.queue.TrialQueue``,
    with a different constructor signature, no ``start()``/``shutdown()``
    lifecycle methods, and a non-async ``submit_batch`` returning bare
    coroutines.

This shim exposes the legacy ``QueueOrchestrator`` and ``OrchestratorEvent``
names so existing SkyRL callers keep working. On legacy Harbor it re-exports
the originals; on unified Harbor it wraps ``TrialQueue`` to preserve the
constructor signature, lifecycle methods, and the awaitable
``submit_batch`` shape that ``terminal_bench_generator.py`` relies on.

Drop this file once we drop pre-unification Harbor support.
"""

from __future__ import annotations

import asyncio
from typing import Any, Iterable, Optional


# ---------------------------------------------------------------------------
# Event enum: OrchestratorEvent → TrialEvent
# ---------------------------------------------------------------------------

try:
    # Legacy Harbor.
    from harbor.orchestrators.base import OrchestratorEvent  # type: ignore[import-not-found]

    TrialEvent = OrchestratorEvent
    TRIAL_COMPLETED_EVENT = OrchestratorEvent.TRIAL_COMPLETED
    _UNIFIED_HARBOR = False
except ImportError:
    # Unified Harbor.
    from harbor.trial.hooks import TrialEvent  # type: ignore[attr-defined]

    # Legacy callers reference `OrchestratorEvent.TRIAL_COMPLETED`. Unified
    # Harbor's TrialEvent enum doesn't have that member (its terminal event is
    # `END`), so a bare `OrchestratorEvent = TrialEvent` alias would explode at
    # attribute access time. Provide a wrapper class that maps the legacy
    # `TRIAL_COMPLETED` name onto `TrialEvent.END` and re-exposes every other
    # TrialEvent member by its actual name. The values are real TrialEvent enum
    # instances, so they're accepted by the underlying TrialQueue.add_hook(...).
    class OrchestratorEvent:  # type: ignore[no-redef]
        """Legacy enum-shaped re-export. `TRIAL_COMPLETED` → `TrialEvent.END`."""

        TRIAL_COMPLETED = TrialEvent.END
        # Mirror unified TrialEvent so newer callers can still use the right names.
        START = TrialEvent.START
        ENVIRONMENT_START = TrialEvent.ENVIRONMENT_START
        AGENT_START = TrialEvent.AGENT_START
        VERIFICATION_START = TrialEvent.VERIFICATION_START
        END = TrialEvent.END
        CANCEL = TrialEvent.CANCEL

    TRIAL_COMPLETED_EVENT = TrialEvent.END
    _UNIFIED_HARBOR = True


# ---------------------------------------------------------------------------
# QueueOrchestrator: legacy class OR wrapper around TrialQueue
# ---------------------------------------------------------------------------

if not _UNIFIED_HARBOR:
    # Legacy: just re-export the original.
    from harbor.orchestrators.queue import QueueOrchestrator  # type: ignore[import-not-found]
else:
    # Unified Harbor: wrap TrialQueue to preserve the legacy API surface
    # that ``terminal_bench_generator.py`` relies on:
    #
    #   - Constructor accepts ``trial_configs``, ``n_concurrent_trials``,
    #     ``metrics``, ``quiet``, ``retry_config`` (legacy names).
    #   - ``add_hook(event, callback)`` registers lifecycle callbacks.
    #   - ``start()`` is awaitable (no-op on new Harbor; TrialQueue has no
    #     startup phase, semaphores are created in __init__).
    #   - ``shutdown(wait=bool)`` is awaitable (no-op; nothing to tear down).
    #   - ``submit_batch(configs)`` is **awaitable** and returns a list of
    #     **scheduled** asyncio.Tasks (matching legacy eager-scheduling).
    #     TrialQueue.submit_batch returns bare coroutines synchronously, so
    #     we wrap each in ``asyncio.create_task`` to mirror the legacy
    #     semantics where futures begin running immediately.
    from harbor.trial.queue import TrialQueue  # type: ignore[attr-defined]

    class QueueOrchestrator:
        """Legacy-API wrapper around ``harbor.trial.queue.TrialQueue``.

        Exposes the surface that SkyRL's terminal-bench generator uses:
        ``__init__``, ``add_hook``, ``start``, ``shutdown``, ``submit_batch``.

        Ignores fields that no longer have a counterpart (``trial_configs``,
        ``metrics``, ``quiet``) — they remain accepted for ABI stability but
        are no-ops on unified Harbor.
        """

        def __init__(
            self,
            *,
            trial_configs: Optional[Iterable[Any]] = None,
            n_concurrent_trials: int,
            metrics: Optional[Any] = None,
            quiet: bool = True,
            retry_config: Optional[Any] = None,
        ) -> None:
            # ``trial_configs``, ``metrics``, ``quiet`` had meaning on legacy
            # QueueOrchestrator but have no counterpart on TrialQueue. SkyRL
            # passes them but always uses ``submit_batch`` for actual work
            # (see ``terminal_bench_generator.py``), so dropping them is safe.
            del trial_configs, metrics, quiet
            self._queue = TrialQueue(
                n_concurrent=n_concurrent_trials,
                retry_config=retry_config,
            )

        def add_hook(self, event, callback):
            """Register a trial-lifecycle hook. Returns self for chaining."""
            self._queue.add_hook(event, callback)
            return self

        async def start(self) -> None:
            """No-op on unified Harbor; preserved for API compatibility."""
            return None

        async def shutdown(self, wait: bool = True) -> None:
            """No-op on unified Harbor; preserved for API compatibility.

            Legacy QueueOrchestrator tore down background tasks; TrialQueue
            has none (it's a coroutine factory, not a worker pool).
            """
            del wait
            return None

        async def submit_batch(self, configs):
            """Schedule trial configs and return a list of asyncio.Tasks.

            Legacy semantics: futures begin running immediately. We
            preserve that by wrapping each TrialQueue-produced coroutine
            in ``asyncio.create_task``. Callers can ``await`` the tasks
            or ``asyncio.gather`` them as before.
            """
            return [asyncio.create_task(coro) for coro in self._queue.submit_batch(configs)]

        # Pass-through accessors callers may need
        @property
        def _trial_queue(self):  # type: ignore[no-untyped-def]
            """Escape hatch for code that needs the underlying TrialQueue."""
            return self._queue


# ---------------------------------------------------------------------------
# create_rollback_hook: legacy callback OR ported in-tree implementation
# ---------------------------------------------------------------------------
# Legacy Harbor had ``harbor.callbacks.create_rollback_hook(...)`` which built a
# callback that, on certain exception types (ContextLengthExceededError,
# AgentTimeoutError), truncated ``agent_result.rollout_details`` to the last
# complete turn so RL training never saw a prompt-without-response pair.
#
# Unified Harbor removed the ``harbor.callbacks`` package entirely. Earlier
# versions of this shim exposed a no-op fallback, but that turned out to be
# load-bearing: when a Daytona trial ended with ContextLengthExceededError /
# AgentTimeoutError, the dangling ``rollout_details`` (mismatched prompt /
# completion / logprobs list lengths) propagated into SkyRL's Ray-driver
# bookkeeping and triggered the distributed-refcount race
# (``reference_count.cc:1619: ref already removed`` → SIGABRT in libuv).
# At least 6 RL jobs across Jupiter + Perlmutter died this way before this
# port landed.
#
# The implementation below is ported verbatim from harbor branch
# ``penfever/temp-override`` (``src/harbor/callbacks/rollback_on_exception.py``),
# adapted only to (a) import harbor model classes from the editable install
# rather than via ``harbor.callbacks`` (which still doesn't exist on unified
# Harbor) and (b) tolerate import failures of those model classes so this
# shim still loads on minimal harbor installs.
#
# Post-truncation guarantees:
#   - ``agent_result.rollout_details[0]`` has consistent list lengths across
#     ``prompt_token_ids`` / ``completion_token_ids`` / ``logprobs`` (or the
#     incomplete field is deleted, per ``preserve_partial_logprobs``).
#   - ``agent_result.metadata["rollback_info"]`` is populated with the action
#     taken, original/final turn counts, exception type, and timestamp.
#   - If no complete turns exist, rollout_details are cleared rather than left
#     in an inconsistent state. The trial keeps its empty rollout_details and
#     ``rollback_info`` records ``action="cleared"``.

try:
    # Prefer harbor's own implementation if a future harbor version restores it.
    from harbor.callbacks import create_rollback_hook  # type: ignore[import-not-found]
    _ROLLBACK_HOOK_SOURCE = "legacy"
except ImportError:
    # Port path. Import the harbor model classes we need; if any of those
    # fail, fall back to a no-op so this shim still imports cleanly.
    try:
        from dataclasses import dataclass as _dataclass, field as _field
        from datetime import datetime as _datetime
        from enum import Enum as _Enum
        from typing import (
            Any as _Any,
            Awaitable as _Awaitable,
            Callable as _Callable,
            Literal as _Literal,
        )
        from typing import Optional as _Optional

        from harbor.models.agent.context import AgentContext as _AgentContext
        from harbor.models.trial.result import TrialResult as _TrialResult
        from harbor.trial.hooks import TrialHookEvent as _TrialHookEvent
        from harbor.utils.logger import logger as _harbor_logger

        _ROLLBACK_DEPS_OK = True
    except ImportError as _rollback_import_err:  # pragma: no cover - defensive
        import logging as _logging

        _rollback_log = _logging.getLogger(__name__)
        _rollback_log.warning(
            "create_rollback_hook: harbor model classes unavailable (%s); "
            "falling back to a no-op stub. rollout_details on context-length "
            "/ agent-timeout trials may remain inconsistent.",
            _rollback_import_err,
        )
        _ROLLBACK_DEPS_OK = False

    if _ROLLBACK_DEPS_OK:

        class _RollbackAction(_Enum):
            """Actions taken during rollback."""

            NONE = "none"  # No action needed (no exception or already consistent)
            TRUNCATED = "truncated"  # Truncated to last complete turn
            CLEARED = "cleared"  # Cleared all data (no complete turns)
            NORMALIZED = "normalized"  # Normalized inconsistent list lengths

        @_dataclass
        class _RollbackResult:
            """Result of a rollback operation."""

            action: "_RollbackAction"
            original_turn_count: int
            final_turn_count: int
            exception_type: _Optional[str] = None
            details: dict = _field(default_factory=dict)

        class _RollbackOnExceptionCallback:
            """Roll ``agent_result.rollout_details`` back to the last complete turn.

            Ported from ``harbor/callbacks/rollback_on_exception.py`` on
            branch ``penfever/temp-override``. See module-level docstring for
            why this lives in SkyRL rather than harbor.
            """

            def __init__(
                self,
                on_complete_failure: str = "mark_metadata",
                exception_types: _Optional[set] = None,
                preserve_partial_logprobs: bool = False,
            ) -> None:
                self.on_complete_failure = on_complete_failure
                self.exception_types = exception_types
                self.preserve_partial_logprobs = preserve_partial_logprobs
                self._logger = _harbor_logger.getChild(__name__)

            async def __call__(self, event) -> None:
                # Accept both TrialHookEvent (new) and bare TrialResult (legacy).
                if isinstance(event, _TrialHookEvent):
                    result = event.result
                    if result is None:
                        self._logger.debug(
                            "TrialHookEvent has no result, skipping rollback"
                        )
                        return
                else:
                    result = event

                if result.exception_info is None:
                    return

                exception_type = result.exception_info.exception_type
                if self.exception_types and exception_type not in self.exception_types:
                    self._logger.debug(
                        "Skipping rollback for exception type %s (not in filter: %s)",
                        exception_type,
                        self.exception_types,
                    )
                    return

                self._logger.debug(
                    "Processing trial %s with exception %s",
                    result.trial_name,
                    exception_type,
                )

                rollback_result = self._rollback_to_last_complete_turn(result)

                if rollback_result.action == _RollbackAction.CLEARED:
                    if self.on_complete_failure == "raise":
                        self._logger.error(
                            "Trial %s has no complete turns and "
                            "on_complete_failure='raise'",
                            result.trial_name,
                        )
                        raise RuntimeError(
                            f"Trial {result.trial_name} failed with "
                            f"{exception_type} and no turns were completed. "
                            f"Original error: "
                            f"{result.exception_info.exception_message}"
                        )

                self._add_rollback_metadata(result, rollback_result)
                self._logger.debug(
                    "Rollback complete for %s: action=%s",
                    result.trial_name,
                    rollback_result.action.value,
                )

            def _rollback_to_last_complete_turn(self, result) -> "_RollbackResult":
                exception_type = (
                    result.exception_info.exception_type
                    if result.exception_info
                    else None
                )

                # Missing agent_result entirely -> CLEARED
                if result.agent_result is None:
                    self._logger.debug(
                        "No agent_result, creating empty AgentContext"
                    )
                    result.agent_result = _AgentContext()
                    return _RollbackResult(
                        action=_RollbackAction.CLEARED,
                        original_turn_count=0,
                        final_turn_count=0,
                        exception_type=exception_type,
                        details={"reason": "no_agent_result"},
                    )

                rollout_details = result.agent_result.rollout_details
                if not rollout_details:
                    # No rollout collected at all -> NONE (nothing to fix).
                    self._logger.debug(
                        "No rollout_details collected, nothing to rollback"
                    )
                    return _RollbackResult(
                        action=_RollbackAction.NONE,
                        original_turn_count=0,
                        final_turn_count=0,
                        exception_type=exception_type,
                        details={"reason": "rollout_details_not_collected"},
                    )

                # Index 0 is the main agent's conversation by convention.
                main_rollout = rollout_details[0]

                prompt_count = len(main_rollout.get("prompt_token_ids", []))
                completion_count = len(main_rollout.get("completion_token_ids", []))
                logprobs_count = len(main_rollout.get("logprobs", []))
                original_max_count = max(prompt_count, completion_count, logprobs_count)

                self._logger.debug(
                    "Rollout detail lengths: prompt=%d, completion=%d, logprobs=%d",
                    prompt_count,
                    completion_count,
                    logprobs_count,
                )

                # A turn is "complete" iff it has response data (logprobs or
                # completion_token_ids). vLLM may omit completion_token_ids
                # even when logprobs are collected, so we accept either.
                response_count = max(completion_count, logprobs_count)
                if response_count == 0:
                    self._logger.debug(
                        "No complete turns (no completion_token_ids or logprobs)"
                    )
                    self._clear_rollout_details(main_rollout)
                    return _RollbackResult(
                        action=_RollbackAction.CLEARED,
                        original_turn_count=original_max_count,
                        final_turn_count=0,
                        exception_type=exception_type,
                        details={
                            "reason": "no_response_data",
                            "original_prompt_count": prompt_count,
                            "original_completion_count": completion_count,
                            "original_logprobs_count": logprobs_count,
                        },
                    )

                # Target turn count = min over non-empty lists (keeps only
                # turns where every collected field is present).
                non_empty_counts = [
                    c
                    for c in (prompt_count, completion_count, logprobs_count)
                    if c > 0
                ]
                target_count = min(non_empty_counts)
                all_consistent = len(set(non_empty_counts)) <= 1

                if target_count == original_max_count and all_consistent:
                    self._logger.debug(
                        "Rollout details already consistent, no action needed"
                    )
                    return _RollbackResult(
                        action=_RollbackAction.NONE,
                        original_turn_count=original_max_count,
                        final_turn_count=target_count,
                        exception_type=exception_type,
                    )

                self._truncate_rollout_detail(main_rollout, target_count)
                action = (
                    _RollbackAction.TRUNCATED
                    if target_count < original_max_count
                    else _RollbackAction.NORMALIZED
                )

                return _RollbackResult(
                    action=action,
                    original_turn_count=original_max_count,
                    final_turn_count=target_count,
                    exception_type=exception_type,
                    details={
                        "original_prompt_count": prompt_count,
                        "original_completion_count": completion_count,
                        "original_logprobs_count": logprobs_count,
                        "truncated_to": target_count,
                    },
                )

            def _truncate_rollout_detail(self, rollout, target_count: int) -> None:
                """Truncate all per-turn lists in a RolloutDetail to ``target_count``."""
                if "prompt_token_ids" in rollout:
                    rollout["prompt_token_ids"] = rollout["prompt_token_ids"][
                        :target_count
                    ]
                if "completion_token_ids" in rollout:
                    rollout["completion_token_ids"] = rollout[
                        "completion_token_ids"
                    ][:target_count]
                if "logprobs" in rollout:
                    if self.preserve_partial_logprobs:
                        rollout["logprobs"] = rollout["logprobs"][:target_count]
                    else:
                        if len(rollout["logprobs"]) >= target_count:
                            rollout["logprobs"] = rollout["logprobs"][:target_count]
                        else:
                            # Incomplete logprobs are removed entirely rather
                            # than leaving a short list paired with full
                            # prompt/completion lists.
                            del rollout["logprobs"]

            def _clear_rollout_details(self, rollout) -> None:
                """Remove every per-turn list from a RolloutDetail."""
                for key in ("prompt_token_ids", "completion_token_ids", "logprobs"):
                    if key in rollout:
                        del rollout[key]

            def _add_rollback_metadata(self, result, rollback_result) -> None:
                """Attach rollback diagnostics to ``agent_result.metadata``."""
                if result.agent_result is None:
                    result.agent_result = _AgentContext()
                if result.agent_result.metadata is None:
                    result.agent_result.metadata = {}
                result.agent_result.metadata["rollback_info"] = {
                    "action": rollback_result.action.value,
                    "original_turn_count": rollback_result.original_turn_count,
                    "final_turn_count": rollback_result.final_turn_count,
                    "exception_type": rollback_result.exception_type,
                    "timestamp": _datetime.now().isoformat(),
                    **rollback_result.details,
                }

        def create_rollback_hook(  # type: ignore[no-redef]
            on_complete_failure: str = "mark_metadata",
            exception_types: _Optional[set] = None,
            preserve_partial_logprobs: bool = False,
        ):
            """Build a rollback hook for ``QueueOrchestrator.add_hook``.

            Args:
                on_complete_failure: ``"mark_metadata"`` (default) records the
                    failure on ``agent_result.metadata`` and keeps empty
                    rollout_details; ``"raise"`` re-raises as ``RuntimeError``.
                exception_types: Optional set of ``exception_info.exception_type``
                    strings to gate on (e.g. ``{"ContextLengthExceededError",
                    "AgentTimeoutError"}``). ``None`` means "any exception".
                preserve_partial_logprobs: When truncating, keep whatever
                    logprobs exist even if shorter than ``target_count``.
                    Default ``False`` deletes the logprobs field entirely
                    rather than risking a mismatched-length artifact.

            Returns:
                An async callable suitable for ``orchestrator.add_hook(
                TRIAL_COMPLETED_EVENT, hook)``. Post-trigger the trial's
                ``agent_result.rollout_details[0]`` is guaranteed to have
                matching per-turn list lengths, and
                ``agent_result.metadata["rollback_info"]`` records what
                happened.
            """
            return _RollbackOnExceptionCallback(
                on_complete_failure=on_complete_failure,
                exception_types=exception_types,
                preserve_partial_logprobs=preserve_partial_logprobs,
            )

        _ROLLBACK_HOOK_SOURCE = "ported"

    else:  # pragma: no cover - defensive (harbor model classes missing)

        async def _noop_rollback_hook(event: Any) -> None:  # type: ignore[no-untyped-def]
            return None

        def create_rollback_hook(  # type: ignore[no-redef]
            on_complete_failure: str = "mark_metadata",
            exception_types: Optional[Iterable[str]] = None,
            preserve_partial_logprobs: bool = False,
        ):
            """Fallback no-op: harbor model classes unavailable.

            Only reached if ``harbor.models.*`` imports fail. Real installs
            should always hit the ported implementation above.
            """
            del on_complete_failure, exception_types, preserve_partial_logprobs
            return _noop_rollback_hook

        _ROLLBACK_HOOK_SOURCE = "noop"


__all__ = [
    "OrchestratorEvent",
    "TrialEvent",
    "TRIAL_COMPLETED_EVENT",
    "QueueOrchestrator",
    "create_rollback_hook",
    "_UNIFIED_HARBOR",
    "_ROLLBACK_HOOK_SOURCE",
]
