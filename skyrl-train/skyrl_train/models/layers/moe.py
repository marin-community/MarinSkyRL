"""Grouped-GEMM MoE block (EP=1, torch backend) — Stage 3b.

Lifted from prime-rl ``src/prime_rl/trainer/models/layers/moe.py`` (the torch
path) with the expert-parallel / DeepEP surface and the aux-loss / load-balance
machinery STRIPPED. This is the EP=1 substrate the Stage-3b grouped-GEMM swap
installs in place of HF's eager ``*SparseMoeBlock`` when ``moe_grouped_gemm`` is
on. Stage 4 re-adds the ``@expert_parallel`` decorators and DeepEP dispatch.

What was lifted / changed vs prime-rl:
  * ``GroupedExperts`` — kept; ``set_ep_comm_backend`` / ``_forward_deepep`` /
    ``ep_comm_backend`` dropped. The for-loop impl ``_run_experts_for_loop`` is
    the EP=1 PARITY DEFAULT (fp32-capable, matches HF eager exactly).
    ``torch._grouped_mm`` is a bf16/SM90-only perf path kept behind
    ``use_grouped_mm`` (validated separately, not the parity oracle). The
    ``@expert_parallel`` decorator is NOT applied here (Stage 4).
  * ``TokenChoiceTopKRouter`` — kept as-is. Its native ``routed_experts`` arg
    (gather ``top_scores = scores.gather(1, routed_experts)``) IS the R3 replay
    hook: it re-gathers weights from the LIVE ``self.gate(x)`` softmax, exactly
    the Stage-2 monkeypatch semantics. ``expert_bias`` / ``force_balanced`` kept
    for API compatibility but unused on the swap path.
  * ``TokenReorderer`` — kept verbatim.
  * ``MoE`` — adapted: ``MoEArgs`` / ``ep_comm_backend`` / DeepEP /
    aux-loss-free ``expert_bias`` / ``tokens_per_expert`` / ``routing_confidence``
    bookkeeping all dropped. The shared expert is OPTIONAL (vanilla Qwen3-MoE
    has none); when present it is a gated ``FeedForward`` plus, for Qwen3-Next,
    a sigmoid ``shared_expert_gate: Linear(hidden, 1)`` —
    ``F.sigmoid(shared_expert_gate(h)) * shared_expert(h)`` (prime-rl's MoE omits
    this sigmoid gate; it is the REQUIRED Qwen3-Next adaptation).

Replay transport: ``MoE.forward`` reads the Stage-2 ``RouterReplay`` singleton
via ``get_active_replay()`` and pulls its per-layer ``[N, K]`` target slice,
threading it into the native router's ``routed_experts`` arg. The
``model_wrapper.forward`` replay-install seam is therefore UNCHANGED between the
eager (3a) and grouped (3b) paths.
"""

from __future__ import annotations

import os
from typing import Literal, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor

# Expert-parallel communication backend. "torch" = torchtitan ExpertParallel
# all_to_all (Stage 4); "deepep" = DeepEP fused dispatch/combine (Stage 5, lazy-imported).
EPCommBackend = Literal["torch", "deepep"]


# --------------------------------------------------------------------------- #
# [EPDIAG] EP residual-desync diagnostic probe (env-gated, cheap, REMOVABLE)   #
# --------------------------------------------------------------------------- #
# References the 2026-06-23 correction: the EP/CP-aware dispatch fix (11556c4)
# replicated input data across EP-group ranks (EPPROBE 947848: 32->2 unique
# shards) but the SeqNum=145 ALLTOALL_BASE[128] deadlock in torchtitan's
# _token_dispatch STILL recurred at ~2:58h (run 948592). That all-to-all is the
# fixed-size [128] num_tokens_per_expert metadata exchange — it can't desync on
# shape, only on per-rank ARRIVAL (a straggler) or via a routing/count DIVERGENCE
# that desyncs the SUBSEQUENT ragged token all-to-all. This probe logs, PER RANK,
# RIGHT BEFORE self.experts(...) (so it prints even if the all-to-all then hangs):
#   - a wall-clock time.time() timestamp (to measure the ~382s arrival spread),
#   - global rank + WORLD_SIZE + decoded (ddp,fsdp,cp,ep) mesh coords,
#   - the full num_tokens_per_expert vector + a stable hash/sum/min/max/argmax,
#   - a hash of the routing selected_experts_indices (the histc INPUT),
#   - whether R3 router-replay was active on this forward + the per-rank fwd index.
# Reading it across an EP group (same ddp,fsdp,cp; varying ep):
#   (a) ntpe_hash + sel_hash MATCH but timestamps spread ~382s  -> COMPUTE/arrival
#       straggler (one rank does more work upstream); NOT a routing problem.
#   (b) ntpe_hash / sel_hash DIVERGE -> routing produces per-rank-different expert
#       assignment despite replicated input -> implicates R3 router-replay (does
#       replay reconstruct identical routing on every rank?) or the CP token-split.
# Gate: EPDIAG=1. Coord decode reads optional EPDIAG_CP / EPDIAG_EP hints (the
# FSDP2 mesh is ["ddp","fsdp","cp","ep"], ep-fastest; dp = rank // (sp*cp*ep)).
# Cheap: a couple of host syncs on a small int vector, only when EPDIAG=1; remove
# the gate-block + this helper once the mechanism is pinned.

_EPDIAG_FWD_COUNT = 0


def _epdiag_enabled() -> bool:
    return os.environ.get("EPDIAG", "0") in ("1", "true", "True")


def _epdiag_decode_coords(rank: int, world: int) -> str:
    """Decode (ddp,fsdp,cp,ep) for a global rank given EPDIAG_CP/EPDIAG_EP hints.

    Mirrors worker.py's flat decomposition: inner = sp*cp*ep (sp=1 on the cp/ep
    config), ep-fastest. dp = rank // inner is the (ddp,fsdp) data-parallel group.
    Falls back to "ep_group=?" if hints are absent (raw rank still logged)."""
    try:
        cp = int(os.environ.get("EPDIAG_CP", "0"))
        ep = int(os.environ.get("EPDIAG_EP", "0"))
        if cp <= 0 or ep <= 0:
            return "coords=?(set EPDIAG_CP,EPDIAG_EP)"
        sp = int(os.environ.get("EPDIAG_SP", "1")) or 1
        inner = sp * cp * ep
        dp = rank // inner  # (ddp,fsdp) data-parallel group index
        rep = rank % inner  # position within the replicated (sp,cp,ep) block
        ep_coord = rep % ep
        cp_coord = (rep // ep) % cp
        # remaining higher dims (sp / fsdp / ddp) collapse into dp here; dp itself
        # identifies the (ddp,fsdp) group, which is what an EP group shares.
        return f"dp_group={dp} cp={cp_coord} ep={ep_coord} (inner={inner})"
    except Exception as e:  # never let the probe break the forward
        return f"coords=err({e})"


def _epdiag_probe(
    num_tokens_per_expert: torch.Tensor,
    selected_experts_indices: torch.Tensor,
    routed_experts: Optional[torch.Tensor],
) -> None:
    """Emit one [EPDIAG] line per rank right before the EP all_to_all. Best-effort."""
    global _EPDIAG_FWD_COUNT
    _EPDIAG_FWD_COUNT += 1
    fwd_idx = _EPDIAG_FWD_COUNT
    try:
        import time

        rank = int(os.environ.get("RANK", "-1"))
        world = int(os.environ.get("WORLD_SIZE", "-1"))
        coords = _epdiag_decode_coords(rank, world)
        ts = time.time()

        # Cheap host syncs on small int tensors (only when EPDIAG=1).
        ntpe = num_tokens_per_expert.detach().to(torch.int64).reshape(-1)
        ntpe_list = ntpe.tolist()
        ntpe_sum = int(ntpe.sum().item())
        ntpe_min = int(ntpe.min().item()) if ntpe.numel() else 0
        ntpe_max = int(ntpe.max().item()) if ntpe.numel() else 0
        ntpe_argmax = int(ntpe.argmax().item()) if ntpe.numel() else -1
        # Order-INDEPENDENT-free stable hash of the count VECTOR (position matters).
        ntpe_hash = hash(tuple(ntpe_list)) & 0xFFFFFFFF

        # Hash of the routing indices themselves (the histc input) — tests whether
        # the ROUTING diverges per rank, not just the resulting counts. Sum the raw
        # int indices into a single scalar fingerprint (cheap, one reduction).
        sel = selected_experts_indices.detach().reshape(-1).to(torch.int64)
        sel_fp = int(sel.sum().item())
        sel_n = int(sel.numel())

        r3_active = "R3replay" if routed_experts is not None else "natural"

        # [EPDIAG2] Degenerate-routing localization (A vs B). The (dp0,cp1) group
        # collapses all its tokens onto a fixed 8-expert set; the existing fields
        # only fingerprint that (ntpe_hash / max=63088), they do NOT say WHICH 8
        # experts nor show the raw forced rows. Add both, cheaply:
        #   - loaded_experts: the SORTED list of expert IDs that received tokens
        #     (the non-zero positions of the [128] global histogram). This is the
        #     A-vs-B tell: {0..7} / sentinel-adjacent (incl. expert 0) => candidate
        #     A (sentinel/mask collapse); an arbitrary content-dependent 8-set =>
        #     candidate B (capture/align degenerate row). n_loaded == 8 confirms the
        #     collapse; >8 = distributed (the non-degenerate groups).
        #   - sel_rows: a few RAW forced routed_experts rows (the [top_k] sets fed
        #     into histc) — head rows + a mid row + tail row of this rank's local
        #     token batch. If every dumped row is the IDENTICAL 8-set => "single
        #     repeated row" confirmed (and we can read its content); if they vary
        #     the collapse is statistical not a single replicated row.
        nz = (ntpe > 0).nonzero().reshape(-1)
        loaded_experts = nz.tolist()
        n_loaded = int(nz.numel())
        sel_rows = "n/a"
        if routed_experts is not None:
            re2d = routed_experts.detach().reshape(-1, routed_experts.shape[-1]).to(torch.int64)
            n_rows = re2d.shape[0]
            if n_rows > 0:
                probe_idx = sorted(set([0, 1, 2, n_rows // 2, n_rows - 1] if n_rows > 1 else [0]))
                probe_idx = [i for i in probe_idx if 0 <= i < n_rows]
                sel_rows = {i: re2d[i].tolist() for i in probe_idx}

        print(
            f"[EPDIAG] ts={ts:.3f} rank={rank}/{world} {coords} fwd={fwd_idx} "
            f"r3={r3_active} ntpe_hash={ntpe_hash} ntpe_sum={ntpe_sum} "
            f"ntpe_min={ntpe_min} ntpe_max={ntpe_max} ntpe_argmax={ntpe_argmax} "
            f"sel_fp={sel_fp} sel_n={sel_n} "
            f"n_loaded={n_loaded} loaded_experts={loaded_experts} sel_rows={sel_rows} "
            f"ntpe={ntpe_list}",
            flush=True,
        )
    except Exception as e:  # never let the probe break the forward
        print(f"[EPDIAG] probe error: {e}", flush=True)


# --------------------------------------------------------------------------- #
# Expert compute kernels (bare impls, no @expert_parallel — Stage 4 re-adds)   #
# --------------------------------------------------------------------------- #


def _run_experts_for_loop(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    """EP=1 parity default: per-expert gated-MLP via a Python for-loop.

    fp32-capable; numerically matches HF eager ``down(silu(gate(x)) * up(x))``.
    """
    # NOTE: incurs a device/host sync (tolist) — acceptable on the parity path.
    # histc returns float counts; split/sum need ints.
    counts = num_tokens_per_expert.to(torch.int64).tolist()
    num_padding = x.shape[0] - sum(counts)

    x_splits = torch.split(x[: sum(counts)], split_size_or_sections=counts, dim=0)
    out_experts_splits = []
    for expert_idx, x_expert in enumerate(x_splits):
        h = F.silu(torch.matmul(x_expert, w1[expert_idx].transpose(-2, -1)))
        h = h * torch.matmul(x_expert, w3[expert_idx].transpose(-2, -1))
        h = torch.matmul(h, w2[expert_idx].transpose(-2, -1))
        out_experts_splits.append(h)
    out = torch.cat(out_experts_splits, dim=0)
    out = torch.vstack((out, out.new_zeros((num_padding, out.shape[-1]))))
    return out


def _run_experts_grouped_mm(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    """bf16/SM90 perf path via ``torch._grouped_mm`` (NOT the parity oracle)."""
    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
    assert x.dim() == 2
    h = F.silu(torch._grouped_mm(x.bfloat16(), w1.bfloat16().transpose(-2, -1), offs=offsets))
    h = h * torch._grouped_mm(x.bfloat16(), w3.bfloat16().transpose(-2, -1), offs=offsets)
    out = torch._grouped_mm(h, w2.bfloat16().transpose(-2, -1), offs=offsets).type_as(x)
    return out


# Stage 4a: the EP grouped-mm compute path.
#
# torchtitan API note (0.2.2): the permute/pad that the OLD ``@expert_parallel``
# decorator performed has moved INTO ``ExpertParallel._token_dispatch`` (the
# distribute_module input_fn): when EP is active that hook all_to_all's the tokens
# AND runs ``generate_permute_indices`` to re-shuffle the cross-rank interleaved
# ``num_tokens_per_expert_group`` into local-expert order + pad each group to
# ALIGN_SIZE_M, and ``_token_combine`` (output_fn) unpermutes. So the EP COMPUTE
# below must run the BARE grouped-mm over the already-dispatched/padded local
# tokens — wrapping it (double-permute/pad) is wrong on the EP path. The standalone
# padding helper was also renamed ``expert_parallel`` -> ``indices_padding_wrapper``
# and moved to ``torchtitan.models.moe.utils``; it is the NON-EP grouped-mm padder
# only (a single rank with no dispatch), which the EP=1 oracle path does not need.
# Mirrors torchtitan's own ``GroupedExperts.forward`` (models/moe/moe.py).


class GroupedExperts(nn.Module):
    """Stacked per-expert gated-MLP weights, run grouped over tokens.

    Parameter layout matches the prime-rl converter target:
        w1: (num_experts, hidden_dim, dim)  — gate_proj
        w3: (num_experts, hidden_dim, dim)  — up_proj
        w2: (num_experts, dim, hidden_dim)  — down_proj
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int,
        use_grouped_mm: bool = False,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.w1 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
        self.w2 = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))
        self.w3 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
        self.use_grouped_mm = use_grouped_mm
        self.ep_comm_backend: EPCommBackend = "torch"

    def set_ep_comm_backend(self, backend: EPCommBackend) -> None:
        self.ep_comm_backend = backend

    def _forward_deepep(self, x: torch.Tensor, num_tokens_per_expert: torch.Tensor) -> torch.Tensor:
        """DeepEP local-expert compute (Stage 5).

        DeepEP's dispatch has already routed each token to its target expert's rank
        and ``MoE._run_deepep_routed_experts`` has permuted the received rows into
        local-expert order; here we just run the LOCAL experts (``.to_local()`` drops
        the ep ``Shard(0)``) over the per-local-expert ``num_tokens_per_expert``.
        """
        w1 = self.w1.to_local()
        w2 = self.w2.to_local()
        w3 = self.w3.to_local()
        if self.use_grouped_mm:
            return _run_experts_grouped_mm(w1, w2, w3, x, num_tokens_per_expert)
        return _run_experts_for_loop(w1, w2, w3, x, num_tokens_per_expert)

    def forward(self, x: torch.Tensor, num_tokens_per_expert: torch.Tensor) -> torch.Tensor:
        # DeepEP backend: dispatch/permute happened upstream in MoE; run local experts.
        if self.ep_comm_backend == "deepep":
            return self._forward_deepep(x, num_tokens_per_expert)
        # EP active ⇒ params are DTensors (Shard(0) on the ep submesh). torchtitan's
        # ExpertParallel._token_dispatch hook has ALREADY all_to_all'd `x` and run
        # generate_permute_indices to re-permute into local-expert order + pad each
        # group to ALIGN_SIZE_M; _token_combine unpermutes after. So here we just drop
        # the ep Shard(0) (`.to_local()`) and run the BARE grouped-mm over the local
        # experts — NO padding wrapper (that would double-pad). Mirrors torchtitan's
        # own GroupedExperts.forward. The for-loop is EP=1-only.
        if isinstance(self.w1, DTensor):
            w1 = self.w1.to_local()
            w2 = self.w2.to_local()
            w3 = self.w3.to_local()
            return _run_experts_grouped_mm(w1, w2, w3, x, num_tokens_per_expert)
        if self.use_grouped_mm:
            return _run_experts_grouped_mm(self.w1, self.w2, self.w3, x, num_tokens_per_expert)
        return _run_experts_for_loop(self.w1, self.w2, self.w3, x, num_tokens_per_expert)

    def init_weights(self, init_std: float = 0.02):
        nn.init.trunc_normal_(self.w1, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.w2, mean=0.0, std=init_std)
        nn.init.trunc_normal_(self.w3, mean=0.0, std=init_std)


class FeedForward(nn.Module):
    """Gated MLP used as the (optional) shared expert. SwiGLU: ``w2(silu(w1 x) * w3 x)``."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # gate_proj
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)  # down_proj
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)  # up_proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# --------------------------------------------------------------------------- #
# Router (native routed_experts arg = the R3 replay hook)                      #
# --------------------------------------------------------------------------- #


class TokenChoiceTopKRouter(nn.Module):
    """Token-choice top-K router. The ``routed_experts`` arg forces the top-k
    expert selection while re-gathering ``top_scores`` from the LIVE softmax —
    the R3 crux (gradients flow through ``self.gate``)."""

    def __init__(
        self,
        dim: int,
        num_experts: int,
        top_k: int,
        score_func: Literal["softmax", "sigmoid"] = "softmax",
        route_norm: bool = False,
        route_scale: float = 1.0,
    ):
        super().__init__()
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.num_experts = num_experts
        self.top_k = top_k
        self.score_func = score_func
        self.route_norm = route_norm
        self.route_scale = route_scale

    def forward(
        self,
        x: torch.Tensor,
        routed_experts: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (bs*slen, dim)
            routed_experts: optional (bs*slen, top_k) forced expert indices.
        Returns:
            top_scores: (bs*slen, top_k)
            selected_experts_indices: (bs*slen, top_k)
            num_tokens_per_expert: (num_experts,)
        """
        assert routed_experts is None or routed_experts.shape[-1] == self.top_k, (
            f"routed_experts shape: {routed_experts.shape}, top_k: {self.top_k}"
        )
        scores = self.gate(x)

        # softmax/sigmoid in float32 to match HF and avoid loss explosion.
        if self.score_func == "sigmoid":
            scores = torch.sigmoid(scores.to(torch.float32))
        elif self.score_func == "softmax":
            scores = F.softmax(scores.to(torch.float32), dim=1)
        else:
            raise NotImplementedError(f"Unknown score function {self.score_func}")

        if routed_experts is not None:
            # R3 replay: indices forced; weights re-gathered from the LIVE softmax.
            top_scores = scores.gather(dim=1, index=routed_experts)
            selected_experts_indices = routed_experts
        else:
            top_scores, selected_experts_indices = torch.topk(scores, k=self.top_k, dim=1)

        if self.route_norm:
            denominator = top_scores.sum(dim=-1, keepdim=True) + 1e-20
            top_scores = top_scores / denominator
        top_scores = top_scores * self.route_scale

        num_tokens_per_expert = torch.histc(
            selected_experts_indices.reshape(-1).float(),
            bins=self.num_experts,
            min=0,
            max=self.num_experts,
        ).to(torch.int64)

        return top_scores, selected_experts_indices, num_tokens_per_expert

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.gate.weight, mean=0.0, std=init_std)


class TokenReorderer(nn.Module):
    """Reorder token indices to match expert ordering for grouped expert compute."""

    def __init__(self, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(
        self,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        selected_experts_indices = selected_experts_indices.reshape(-1)
        # int64 counts: the for-loop path needs ints (tolist→split), and the EP
        # all_to_all dispatch (torchtitan _token_dispatch) requires INTEGER split
        # sizes — a float histc here makes NCCL alltoall_base reject the splits.
        num_tokens_per_expert = torch.histc(
            selected_experts_indices.float(),
            bins=self.num_experts,
            min=0,
            max=self.num_experts,
        ).to(torch.int64)
        token_indices_experts_sorted = torch.argsort(selected_experts_indices, stable=True)
        top_scores_experts_sorted = top_scores.view(-1)[token_indices_experts_sorted]
        token_indices_experts_sorted = token_indices_experts_sorted // self.top_k
        return top_scores_experts_sorted, token_indices_experts_sorted, num_tokens_per_expert


# --------------------------------------------------------------------------- #
# MoE orchestrator (EP=1; optional shared expert with Qwen3-Next sigmoid gate) #
# --------------------------------------------------------------------------- #


class MoE(nn.Module):
    """Grouped-GEMM MoE block, EP=1.

    Args:
        dim: hidden size.
        hidden_dim: routed-expert intermediate size (``moe_intermediate_size``).
        num_experts: number of routed experts.
        top_k: experts per token.
        route_norm: normalize top-k scores (HF ``norm_topk_prob``).
        score_func: "softmax" (Qwen) or "sigmoid".
        use_grouped_mm: bf16 perf path (default False → for-loop parity).
        shared_expert_dim: when not None, add a gated shared-expert FeedForward
            with this intermediate size (Qwen3-Next ``shared_expert_intermediate_size``).
        shared_expert_gated: when True, sigmoid-gate the shared expert output
            via a ``Linear(dim, 1)`` (the REQUIRED Qwen3-Next adaptation).
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int,
        top_k: int,
        route_norm: bool,
        score_func: Literal["softmax", "sigmoid"] = "softmax",
        use_grouped_mm: bool = False,
        shared_expert_dim: Optional[int] = None,
        shared_expert_gated: bool = False,
        score_before_experts: bool = True,
    ):
        super().__init__()
        self.experts = GroupedExperts(
            dim=dim,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            use_grouped_mm=use_grouped_mm,
        )
        self.ep_comm_backend: EPCommBackend = "torch"
        self.experts.set_ep_comm_backend(self.ep_comm_backend)
        # DeepEP scores tokens BEFORE the experts (the DeepEP path applies the
        # routing weight to the dispatched activation pre-matmul). The torch /
        # for-loop path keeps SkyRL's score-after-experts (matches HF eager).
        self.score_before_experts = score_before_experts
        self.deepep_token_chunk_size: Optional[int] = None
        self.router = TokenChoiceTopKRouter(
            dim=dim,
            num_experts=num_experts,
            top_k=top_k,
            score_func=score_func,
            route_norm=route_norm,
            route_scale=1.0,
        )
        self.reorderer = TokenReorderer(num_experts=num_experts, top_k=top_k)
        self.top_k = top_k

        if shared_expert_dim is not None:
            self.shared_expert = FeedForward(dim=dim, hidden_dim=shared_expert_dim)
            # Qwen3-Next: F.sigmoid(shared_expert_gate(h)) * shared_expert(h).
            self.shared_expert_gate = nn.Linear(dim, 1, bias=False) if shared_expert_gated else None
        else:
            self.shared_expert = None
            self.shared_expert_gate = None

    def set_ep_comm_backend(self, backend: EPCommBackend) -> None:
        self.ep_comm_backend = backend
        self.experts.set_ep_comm_backend(backend)

    def set_deepep_token_chunk_size(self, chunk_size: Optional[int]) -> None:
        self.deepep_token_chunk_size = chunk_size

    def _run_local_routed_experts(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        return self.experts(x, num_tokens_per_expert)

    def _run_deepep_routed_experts(
        self,
        x: torch.Tensor,
        selected_experts_indices: torch.Tensor,
        top_scores: torch.Tensor,
    ) -> torch.Tensor:
        """DeepEP routed-expert compute (Stage 5).

        Dispatches each token to its target expert's rank via DeepEP's fused
        all_to_all, runs the LOCAL experts, then combines back (which unpermutes →
        token i returns to row i, preserving router replay; scope §3). Chunked to
        overlap dispatch of chunk k+1 with the compute of chunk k. The combine
        already unpermutes, so NO scatter_add here (unlike the torch path).
        """
        from skyrl_train.distributed.deepep import (
            combine_tokens,
            dispatch_tokens_async,
            finalize_dispatch_tokens,
            sync_combine,
        )
        from skyrl_train.distributed.expert_parallel import get_ep_group

        if x.shape[0] == 0:
            shared_output = self.shared_expert(x) if self.shared_expert is not None else None
            return x.new_zeros(x.shape) if shared_output is None else shared_output

        group = get_ep_group(self.experts)
        chunk_size = min(self.deepep_token_chunk_size or x.shape[0], x.shape[0])

        def dispatch_chunk(start: int, end: int):
            return dispatch_tokens_async(
                x[start:end],
                selected_experts_indices[start:end],
                top_scores[start:end],
                num_experts=self.experts.num_experts,
                group=group,
                score_before_experts=self.score_before_experts,
            )

        def run_pending_chunk(pending_state):
            hidden_states, num_tokens_per_expert, dispatch_state = finalize_dispatch_tokens(pending_state)
            routed_output = self._run_local_routed_experts(hidden_states, num_tokens_per_expert)
            # Keep combine outside the checkpointed routed-expert region so
            # selective AC only recomputes local expert matmuls.
            return combine_tokens(routed_output, dispatch_state)

        pending_state = dispatch_chunk(0, chunk_size)
        routed_outputs: list[torch.Tensor] = []

        for chunk_start in range(chunk_size, x.shape[0], chunk_size):
            chunk_end = min(chunk_start + chunk_size, x.shape[0])
            next_pending_state = dispatch_chunk(chunk_start, chunk_end)
            routed_outputs.append(run_pending_chunk(pending_state))
            pending_state = next_pending_state

        routed_outputs.append(run_pending_chunk(pending_state))

        # Qwen3-(Next) sigmoid-gated shared expert applied on the combined per-token
        # output (NOT prime-rl's BCFeedForward); added to the routed combine result.
        if self.shared_expert is not None:
            shared_output = self.shared_expert(x)
            if self.shared_expert_gate is not None:
                shared_output = F.sigmoid(self.shared_expert_gate(x)) * shared_output
        else:
            shared_output = None
        sync_combine()
        routed_output = routed_outputs[0] if len(routed_outputs) == 1 else torch.cat(routed_outputs, dim=0)
        return routed_output if shared_output is None else shared_output + routed_output

    def _run_routed_experts(
        self,
        x: torch.Tensor,
        token_indices_experts_sorted: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
        top_scores_experts_sorted: torch.Tensor,
    ) -> torch.Tensor:
        dim = x.shape[-1]
        routed_indices = token_indices_experts_sorted.reshape(-1, 1).expand(-1, dim)
        routed_input = torch.gather(x, dim=0, index=routed_indices)
        routed_output = self.experts(routed_input, num_tokens_per_expert)
        # Scale AFTER experts (HF eager multiplies the expert output by routing_weights).
        routed_output = (routed_output.to(torch.float32) * top_scores_experts_sorted.reshape(-1, 1)).to(x.dtype)
        return routed_output

    def forward(
        self,
        x: torch.Tensor,
        routed_experts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (bs, slen, dim).
            routed_experts: optional (bs, slen, top_k) forced expert indices.
        Returns:
            (bs, slen, dim).
        """
        bs, slen, dim = x.shape
        x = x.view(-1, dim)

        if routed_experts is not None:
            _, _, top_k = routed_experts.shape
            # Reshape here because the source [bs, slen, top_k] is non-contiguous.
            routed_experts = routed_experts.reshape(-1, top_k)

        top_scores, selected_experts_indices, _ = self.router(x, routed_experts=routed_experts)

        if self.ep_comm_backend == "deepep":
            # DeepEP drives dispatch→local-experts→combine; combine already
            # unpermutes (token i → row i), so no reorderer/scatter_add here.
            routed_output = self._run_deepep_routed_experts(x, selected_experts_indices, top_scores)
            return routed_output.reshape(bs, slen, dim)

        (
            top_scores_experts_sorted,
            token_indices_experts_sorted,
            num_tokens_per_expert,
        ) = self.reorderer(top_scores, selected_experts_indices)

        # [EPDIAG] EP residual-desync probe: log per-rank num_tokens_per_expert +
        # routing fingerprint + arrival timestamp RIGHT BEFORE the EP all_to_all
        # (torchtitan _token_dispatch fires inside self.experts on the EP/torch
        # path). Env-gated (EPDIAG=1), cheap, removable. See helper above.
        if _epdiag_enabled():
            _epdiag_probe(num_tokens_per_expert, selected_experts_indices, routed_experts)

        routed_output = self._run_routed_experts(
            x,
            token_indices_experts_sorted,
            num_tokens_per_expert,
            top_scores_experts_sorted,
        )

        if self.shared_expert is not None:
            out = self.shared_expert(x)
            if self.shared_expert_gate is not None:
                out = F.sigmoid(self.shared_expert_gate(x)) * out
        else:
            out = torch.zeros_like(x)

        routed_indices = token_indices_experts_sorted.reshape(-1, 1).expand(-1, dim)
        out = out.scatter_add(dim=0, index=routed_indices, src=routed_output)
        out = out.reshape(bs, slen, dim)
        return out

    def init_weights(self, init_std: float):
        self.experts.init_weights(init_std)
        self.router.init_weights(init_std)
        if self.shared_expert is not None:
            for linear in (self.shared_expert.w1, self.shared_expert.w2, self.shared_expert.w3):
                nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)
            if self.shared_expert_gate is not None:
                nn.init.trunc_normal_(self.shared_expert_gate.weight, mean=0.0, std=init_std)
