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

    OrchestratorEvent = TrialEvent  # type: ignore[assignment]
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


__all__ = [
    "OrchestratorEvent",
    "TrialEvent",
    "TRIAL_COMPLETED_EVENT",
    "QueueOrchestrator",
    "_UNIFIED_HARBOR",
]
