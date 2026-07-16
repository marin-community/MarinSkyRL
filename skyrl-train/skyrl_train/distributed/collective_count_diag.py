"""GC-proof per-rank default-PG collective-count instrumentation (diagnostic).

Localizes the recurring 80B gs1 NCCL desync — a per-EP-group *default-PG
collective-count divergence*: at the gs1 old-logprob forward a subset of ranks
issues an extra/fewer default-PG collective UPSTREAM, so the ranks desync at the
first shared collective (fingerprint ``default_pg ALLREDUCE NumelIn=1`` ~#288605,
surfaced as a ``mesh_fsdp _all_gather_base`` unshard wedge). The FR dump that
would fingerprint the AWOL rank is a node-local ``/tmp`` file lost to pod GC, so
this instead logs each rank's default-PG collective count at forward PHASE
BOUNDARIES to the driver/actor stdout -> the iris FINELOG, which SURVIVES pod GC.
At the wedge, diffing the last-logged count across ranks reveals which
rank(s)/EP-group diverged.

Design constraints (this is a Heisenbug — the instrumentation must NOT perturb
NCCL timing enough to mask it):
  * O(phases), NOT O(collectives). The count is READ from torch's own per-PG
    sequence counter (``ProcessGroup._get_sequence_number_for_group()`` — the same
    ``seqCollective_`` the NCCL watchdog reports as "Last enqueued/completed NCCL
    work" and the desync fingerprint counts), which torch increments regardless.
    There is ZERO added per-collective work and NO collective wrapping.
  * A handful of log lines per forward (enter/exit of ``forward`` + ``_forward_impl``
    + the FIRST MoE-EP all-to-all per forward) — never per-op INFO spam.

Gated behind ``SKYRL_COLLECTIVE_COUNT_DIAG`` (default OFF), exposed as the
first-class ``--collective-count-diag`` launcher flag (deslop-stage-3 convention;
NOT a magic env var — the flag is the interface, the env var is the transport).
Every function is a fast no-op when disabled and NEVER raises.
"""

from __future__ import annotations

import os
import threading

from loguru import logger

_ENV = "SKYRL_COLLECTIVE_COUNT_DIAG"

# Reset once per forward region (a policy-worker ``_forward_impl`` call) so the
# MoE-EP boundary logs the FIRST all-to-all of that forward, not every one of the
# ~48 MoE layers. ``_forward_impl`` and the MoE ``forward`` it drives run on the
# SAME thread (inline, or one ``asyncio.to_thread`` worker), so thread-local state
# correctly scopes the once-guard to a single forward.
_state = threading.local()


def enabled() -> bool:
    return os.environ.get(_ENV, "0") == "1"


def _default_pg_seq():
    """The default process group's collective sequence number, or None if
    unreadable. This is torch's own ``seqCollective_`` counter (the NCCL watchdog's
    "Last enqueued/completed NCCL work") — an O(1) read of a value torch increments
    on every collective anyway, so reading it adds no collective and no wrapping."""
    try:
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized():
            return None
        return dist.distributed_c10d._get_default_group()._get_sequence_number_for_group()
    except Exception:
        return None


def _rank() -> int:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return -1


def log_phase(phase: str, rank=None) -> None:
    """Log this rank's default-PG collective count at a forward phase boundary.
    No-op unless ``SKYRL_COLLECTIVE_COUNT_DIAG=1``; never raises."""
    if not enabled():
        return
    try:
        r = rank if rank is not None else _rank()
        n = _default_pg_seq()
        logger.info(f"COLLECTIVE_COUNT_DIAG rank={r} phase={phase} default_pg_collective_count={n}")
    except Exception:
        pass


def begin_forward_region() -> None:
    """Mark the start of a policy-worker ``_forward_impl`` so the MoE-EP boundary
    logs only its FIRST all-to-all (keeps the MoE log O(1) per forward, not
    O(MoE layers))."""
    if not enabled():
        return
    _state.moe_logged = False


def log_moe_ep_boundary_once(rank=None) -> None:
    """Log the default-PG count at the FIRST MoE EP all-to-all of the current
    forward region (rate-limited to once per ``_forward_impl``). The torch-EP
    all-to-all fires inside ``MoE.forward`` (torchtitan ``_token_dispatch`` /
    ``_token_combine`` wrapping the experts call), so this captures the default-PG
    count entering the MoE-EP region — where the per-EP-group divergence lands."""
    if not enabled():
        return
    if getattr(_state, "moe_logged", False):
        return
    _state.moe_logged = True
    log_phase("moe_ep_a2a_first", rank=rank)
