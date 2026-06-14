# Stage 4 (FSDP2 Context Parallel) — torch-native CP context manager.
#
# Thin wrapper over the torch-native `context_parallel(...)` pattern (ported
# faithfully from NeMo-RL `dtensor_policy_worker.py:505-529`). The wrapper enters
# `torch.distributed.tensor.experimental.context_parallel`, which shards the
# listed sequence buffers across the `cp` device-mesh group using torch's
# built-in zigzag load balancer (requires `seq_len % (2*cp) == 0`, enforced by
# the padding in `model_wrapper.forward`) and routes SDPA attention through ring
# attention on that group.
#
# `maybe_cp_context(...)` returns `contextlib.nullcontext()` when `cp_size == 1`
# so the flag-off path is a LITERAL no-op (G1): no torch CP machinery is touched,
# no buffers are mutated, and the forward stays byte-identical to today.
#
# Import surface is pinned by the Stage-1 import test
# (`tests/cpu/distributed/test_torch_cp_available.py`) — if torch moves these
# private `_attention` symbols, that test fails loudly.

import contextlib
from typing import Iterable, List, Optional, Set

import torch
from loguru import logger

from torch.distributed.tensor.experimental import context_parallel
from torch.distributed.tensor.experimental._attention import (
    context_parallel_unshard,  # noqa: F401  (re-exported for Stage 5)
    set_rotate_method,
)
import torch.distributed._functional_collectives as funcol


def cp_unshard_grad_safe(cp_mesh, tensor: torch.Tensor, seq_dim: int) -> torch.Tensor:
    """Autograd-DIFFERENTIABLE context-parallel unshard of a per-token tensor.

    torch's ``context_parallel_unshard`` is decorated ``@torch.no_grad()`` and does
    an in-place index-restore (``unsharded[i] = index_select(...)``) that raises
    ``"cannot resize variables that require grad"`` when fed a grad-requiring tensor.
    That is fine for Stage-5's no-grad parity scoring, but a CP *training* step needs
    the gradient of the policy loss to flow back through the per-token logprob gather
    (the per-token logprobs feed ``ppo_policy_loss`` -> ``backward()``). This helper
    reproduces the SAME natural-order output as ``context_parallel_unshard`` but with
    a fully differentiable path:

      1. ``funcol.all_gather_tensor`` along ``seq_dim`` (autograd-aware: its backward
         is the matching reduce-scatter), giving the gathered-but-zigzag-ordered tensor.
      2. Restore to natural token order with an OUT-OF-PLACE ``index_select`` using a
         permutation discovered by running the stock (no_grad) unshard on an integer
         "tag" tensor of the SAME local length — so the reorder is byte-identical to
         torch's, but applied differentiably.

    For ``cp_size == 1`` the caller never invokes this (the forward skips the unshard).
    """
    cp_size = cp_mesh.size()
    if cp_size == 1:
        return tensor

    # (1) Differentiable gather (zigzag order across ranks).
    gathered = funcol.all_gather_tensor(tensor.contiguous(), gather_dim=seq_dim, group=cp_mesh)
    gathered = funcol.wait_tensor(gathered)

    # (2) Discover the gather->natural permutation with the STOCK unshard on a tag
    #     tensor. Tag value at local position j = global "slot" of this rank's j-th
    #     token in the gathered tensor (rank r occupies slots [r*L, (r+1)*L)). The
    #     stock unshard reorders those slots into natural token order; reading the
    #     resulting tag sequence gives perm such that gathered[perm] == natural.
    local_len = tensor.size(seq_dim)
    rank_in_cp = cp_mesh.get_local_rank()
    tag = torch.arange(
        rank_in_cp * local_len, (rank_in_cp + 1) * local_len, device=tensor.device, dtype=torch.long
    )
    # shape the tag like the buffer along seq_dim ([1, L] then unshard expects (B,S)).
    tag2d = tag.unsqueeze(0)  # [1, L]
    natural_slots = context_parallel_unshard(cp_mesh, [tag2d], [1])[0][0]  # [S] natural-order slot ids
    perm = natural_slots.to(gathered.device).long()

    out = torch.index_select(gathered, dim=seq_dim, index=perm)
    return out

# Track the last rotate method we set so `set_rotate_method` is called once
# (idempotent) rather than per-step. torch's `set_rotate_method` mutates a
# module-level global, so repeated identical calls are harmless but wasteful;
# we skip the redundant call when the method is unchanged.
_CURRENT_ROTATE_METHOD: Optional[str] = None

_VALID_ROTATE_METHODS = ("allgather", "all_to_all")


def set_cp_rotate_method(rotate_method: Optional[str]) -> None:
    """Set torch's CP rotate method once (idempotent).

    ``rotate_method`` ∈ {"allgather", "all_to_all"}. ``None`` leaves torch's
    default untouched. Calling repeatedly with the same value is a no-op.
    """
    global _CURRENT_ROTATE_METHOD
    if rotate_method is None:
        return
    assert (
        rotate_method in _VALID_ROTATE_METHODS
    ), f"cp_rotate_method='{rotate_method}' invalid; must be one of {_VALID_ROTATE_METHODS}"
    if rotate_method == _CURRENT_ROTATE_METHOD:
        return
    set_rotate_method(rotate_method)
    _CURRENT_ROTATE_METHOD = rotate_method
    logger.info(f"[CP] set_rotate_method('{rotate_method}')")


def cp_context(
    cp_mesh,
    rotate_method: Optional[str],
    buffers: List[torch.Tensor],
    seq_dims: List[int],
    no_restore: Optional[Set[torch.Tensor]] = None,
):
    """Enter torch-native context parallel over ``cp_mesh``.

    Ported from NeMo-RL `dtensor_policy_worker.py:505-529`. Inside this context,
    the listed ``buffers`` are sharded along their ``seq_dims`` across the CP
    group (torch's built-in load balancer handles the per-rank zigzag offset),
    and SDPA attention dispatches to ring attention on ``cp_mesh``.

    Args:
        cp_mesh: the ``cp`` submesh (``device_mesh["cp"]``).
        rotate_method: "allgather" | "all_to_all" | None (set once, idempotent).
        buffers: sequence tensors to CP-shard (e.g. sequences, position_ids,
            attention_mask). Each is sharded in-place along its ``seq_dims`` entry.
        seq_dims: per-buffer sequence dimension index (parallel to ``buffers``).
        no_restore: subset of ``buffers`` NOT to restore to the unsharded layout
            on context exit (an optimization for buffers we discard afterward).

    Returns:
        The ``context_parallel`` context manager (caller uses it in a ``with``).
    """
    set_cp_rotate_method(rotate_method)
    no_restore_buffers = no_restore if no_restore is not None else set()
    return context_parallel(
        cp_mesh,
        buffers=buffers,
        buffer_seq_dims=seq_dims,
        no_restore_buffers=no_restore_buffers,
    )


def maybe_cp_context(
    cp_size: int,
    cp_mesh,
    rotate_method: Optional[str],
    buffers: Iterable[torch.Tensor],
    seq_dims: List[int],
    no_restore: Optional[Set[torch.Tensor]] = None,
):
    """Return the CP context when ``cp_size > 1``, else a literal no-op.

    When ``cp_size == 1`` (flag-off / default), this returns
    ``contextlib.nullcontext()`` — torch CP is never touched, ``buffers`` are not
    mutated, and the forward is byte-identical to today (G1). Otherwise it
    delegates to :func:`cp_context`.
    """
    if cp_size <= 1:
        return contextlib.nullcontext()
    assert cp_mesh is not None, "cp_size > 1 but cp_mesh is None (Stage-3 mesh not surfaced)"
    return cp_context(
        cp_mesh,
        rotate_method,
        buffers=list(buffers),
        seq_dims=seq_dims,
        no_restore=no_restore,
    )
