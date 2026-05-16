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
# create_rollback_hook: legacy callback OR no-op stub
# ---------------------------------------------------------------------------
# Legacy Harbor had `harbor.callbacks.create_rollback_hook(...)` which built a
# callback that, on certain exception types (ContextLengthExceededError,
# AgentTimeoutError), truncated `agent_result.rollout_details` to the last
# complete turn so RL training never saw a prompt-without-response pair.
#
# Unified Harbor removed the `harbor.callbacks` package entirely along with
# the rollback feature. There is no in-tree replacement.
#
# We expose a no-op fallback so SkyRL can keep its existing import and
# `add_hook(TRIAL_COMPLETED_EVENT, rollback_hook)` calls working. The trade-off
# on unified Harbor: rollout_details may contain incomplete turns on the
# specific exception types listed above. If this degrades training, port the
# full logic from harbor commit ca3294a4:src/harbor/callbacks/rollback_on_exception.py
# into this file.

try:
    from harbor.callbacks import create_rollback_hook  # type: ignore[import-not-found]
    _ROLLBACK_HOOK_SOURCE = "legacy"
except ImportError:
    import logging as _logging
    _rollback_log = _logging.getLogger(__name__)
    _rollback_warning_emitted = False

    async def _noop_rollback_hook(event: Any) -> None:  # type: ignore[no-untyped-def]
        return None

    def create_rollback_hook(  # type: ignore[no-redef]
        exception_types: Optional[Iterable[str]] = None,
        on_complete_failure: str = "mark_metadata",
        preserve_partial_logprobs: bool = False,
    ):
        """No-op rollback hook for unified Harbor (harbor.callbacks removed).

        Returns an awaitable that does nothing. rollout_details on the trial
        result are left as-is; downstream callers that depend on truncation
        on ContextLengthExceededError / AgentTimeoutError may see incomplete
        turns. See module docstring above for the porting path.
        """
        global _rollback_warning_emitted
        if not _rollback_warning_emitted:
            _rollback_log.warning(
                "create_rollback_hook: harbor.callbacks is gone on unified "
                "Harbor; returning a no-op stub. rollout_details may include "
                "incomplete turns when trials end with %s. "
                "Port harbor commit ca3294a4 callbacks/rollback_on_exception.py "
                "into _harbor_compat.py if RL training needs the truncation.",
                sorted(exception_types) if exception_types else "any exception",
            )
            _rollback_warning_emitted = True
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
