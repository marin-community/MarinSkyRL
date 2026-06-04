"""MoE router replay (R3) on the eager HF MoE path — single-GPU subset.

Ported from verl's ``verl/utils/veomni/router_replay.py`` (the
``VeOmniRouterReplay`` controller) with the Ulysses SP all-gather / slice
helpers DROPPED (single GPU, no EP — Stage 2). SkyRL loads stock HF
``AutoModelForCausalLM`` whose ``Qwen3MoeSparseMoeBlock.forward(self,
hidden_states)`` takes only hidden states, so we cannot thread
``routed_experts`` as an explicit forward kwarg the way prime-rl does. Instead
we follow the veRL pattern: a module-level controller singleton plus a
monkeypatched ``*SparseMoeBlock.forward`` that reads per-layer targets from the
controller.

Lifecycle (per ``training_step`` micro-batch, REPLAY only — Stage 2 has no
RECORD path)::

    set_active_replay(ctrl)                          # install singleton
    ctrl.begin_replay()
    ctrl.set_microbatch_targets(per_layer_targets, replay_mask)
    out = model(...)                                 # patched forward fires
    ctrl.clear()
    set_active_replay(None)                          # uninstall

Layer indexing is **id-keyed positional**: the first time each MoE router fires
we assign the next position (``len(_id_to_pos)``) to ``id(module)``; every
later call for that module reuses the same position. This is recompute-safe
under activation checkpointing (backward recompute fires the same modules in
any order; the id lookup lands on the same position).

R3 crux (lives in the patched forward, not here): indices come from the
rollout (the routing decision); ``routing_weights`` are re-gathered from the
**live** trainer softmax so gradients flow through ``self.gate`` and the
experts. The controller is indices-only and model-agnostic.
"""

from __future__ import annotations

import os
from enum import Enum
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    import torch.nn as nn


__all__ = [
    "RouterReplayAction",
    "RouterReplay",
    "set_active_replay",
    "get_active_replay",
    "install_router_replay_patch",
    "count_moe_layers",
]


# Sentinel expert id written by the Stage-1 capture rail for unmatched /
# non-generated token rows (generators/utils.py:627). Rows whose captured
# targets are all this value fall through to native routing.
SENTINEL_EXPERT_ID = 0


# --------------------------------------------------------------------------- #
# Module-level controller singleton                                           #
# --------------------------------------------------------------------------- #

_active: Optional["RouterReplay"] = None


def set_active_replay(controller: Optional["RouterReplay"]) -> None:
    """Install (or clear with ``None``) the active replay controller.

    The patched ``SparseMoeBlock.forward`` reads this slot so it can find the
    controller without a forward kwarg. ``None`` ⇒ patched forward behaves
    exactly like stock HF (it computes natural topk and skips substitution).
    """
    global _active
    _active = controller


def get_active_replay() -> Optional["RouterReplay"]:
    return _active


class RouterReplayAction(Enum):
    DISABLED = "disabled"
    REPLAY = "replay"


class RouterReplay:
    """Single-GPU router replay controller (REPLAY-only subset of veRL's
    ``VeOmniRouterReplay``; RECORD + Ulysses SP helpers dropped — Stage 2)."""

    def __init__(self) -> None:
        self._action: RouterReplayAction = RouterReplayAction.DISABLED
        # id(router_module) -> position. Populated lazily on first sight of
        # each router; stable across the controller lifetime. Re-discovered
        # between phases (cleared by begin_replay); the same model produces
        # the same id table.
        self._id_to_pos: dict[int, int] = {}
        # REPLAY: positional list of [num_tokens, top_k] target index tensors
        # for the current micro-batch, ordered by layer position.
        self._targets: list[torch.Tensor] = []
        # REPLAY: optional [num_tokens] bool mask. True ⇒ substitute the
        # recorded target; False ⇒ fall through to native routing. R3 needs
        # this to skip prompt / pad / sentinel rows.
        self._replay_mask: Optional[torch.Tensor] = None
        self._debug: bool = os.environ.get("SKYRL_ROUTER_REPLAY_DEBUG") == "1"

    @property
    def action(self) -> RouterReplayAction:
        return self._action

    @property
    def num_layers(self) -> int:
        """Number of MoE layers discovered so far. In REPLAY this is known
        directly from the target list length once set_microbatch_targets ran;
        otherwise it reflects the id table populated as routers fire."""
        return max(len(self._id_to_pos), len(self._targets))

    # ------------------------------------------------------------- drivers

    def begin_replay(self) -> None:
        """Enter REPLAY mode. Must be called before the forward."""
        self._action = RouterReplayAction.REPLAY
        self._targets = []
        self._replay_mask = None
        self._id_to_pos = {}

    def set_microbatch_targets(
        self,
        per_layer_targets: list[torch.Tensor],
        replay_mask: Optional[torch.Tensor] = None,
    ) -> None:
        """Load per-layer target indices for the upcoming forward.

        ``per_layer_targets[i]`` is ``[num_tokens, top_k]`` int64 on device,
        ordered by layer position (matches the order routers fire). ``L`` is
        taken from the list length — no prior id discovery required.

        ``replay_mask`` (optional, ``[num_tokens]`` bool): per-token gate.
        True ⇒ substitute with the recorded target; False ⇒ native routing.
        """
        if self._action is not RouterReplayAction.REPLAY:
            raise RuntimeError(f"set_microbatch_targets requires REPLAY action, got {self._action}")
        self._targets = list(per_layer_targets)
        self._replay_mask = replay_mask.bool() if replay_mask is not None else None

    def clear(self) -> None:
        """Reset the state machine between micro-batches / steps."""
        self._action = RouterReplayAction.DISABLED
        self._targets = []
        self._replay_mask = None
        self._id_to_pos = {}

    def assert_layer_count(self, expected: int) -> None:
        if self.num_layers != expected:
            raise AssertionError(
                f"router_replay discovered {self.num_layers} MoE layers, "
                f"model config says {expected}. Layer discovery is broken."
            )

    # ------------------------------------------------------ router-side entry

    def on_router_forward(
        self,
        module: "nn.Module",
        routing_scores: torch.Tensor,
        top_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Called from each patched MoE router forward.

        Indices-only: returns the substituted target indices in REPLAY mode,
        else returns ``top_indices`` unchanged (DISABLED / no targets). All
        weight math (gather, renorm, dtype cast) lives in the patched
        ``SparseMoeBlock.forward`` so the controller stays model-agnostic.
        ``routing_scores`` is accepted for optional debug only.

        Position is keyed on ``id(module)`` so activation-checkpoint recompute
        (which fires routers again, possibly out of order) lands on the same
        position as the original forward.
        """
        mid = id(module)
        if mid in self._id_to_pos:
            pos = self._id_to_pos[mid]
        else:
            pos = len(self._id_to_pos)
            self._id_to_pos[mid] = pos

        if self._action is not RouterReplayAction.REPLAY:
            return top_indices

        if self._debug:
            if routing_scores.dim() != 2 or top_indices.dim() != 2:
                raise AssertionError(
                    f"router_replay: expected 2D tensors, got routing_scores "
                    f"{tuple(routing_scores.shape)} and top_indices {tuple(top_indices.shape)}."
                )
            if routing_scores.shape[0] != top_indices.shape[0]:
                raise AssertionError(
                    f"router_replay: routing_scores / top_indices row count mismatch: "
                    f"{routing_scores.shape[0]} vs {top_indices.shape[0]}."
                )

        # Strict: every layer position must have a target. A missing target is
        # a real plumbing bug (layer count mismatch, or set_microbatch_targets
        # not called before forward).
        if pos >= len(self._targets):
            raise RuntimeError(
                f"router_replay REPLAY: layer pos={pos} has no target "
                f"(only {len(self._targets)} targets set for this micro-batch). "
                "Likely cause: model has more MoE layers than the recorded "
                "routed_experts tensor describes, or set_microbatch_targets was "
                "not called before forward."
            )
        target = self._targets[pos].to(top_indices.device)
        if target.shape[0] != top_indices.shape[0]:
            raise RuntimeError(
                f"router_replay REPLAY: target at pos={pos} has {target.shape[0]} rows "
                f"but top_indices has {top_indices.shape[0]}."
            )

        if self._replay_mask is None:
            substituted = target
        else:
            mask = self._replay_mask.to(top_indices.device)
            if mask.shape[0] != top_indices.shape[0]:
                raise RuntimeError(
                    f"router_replay REPLAY: replay_mask has {mask.shape[0]} rows "
                    f"but top_indices has {top_indices.shape[0]}."
                )
            # Per-token gated substitution: True ⇒ recorded target, False ⇒
            # native routing (prompt / pad / sentinel rows).
            substituted = torch.where(mask.unsqueeze(-1), target, top_indices)

        # Defensive duplicate-detection: a forced row whose top-k contains a
        # duplicate expert (e.g. a sentinel row that slipped past the mask)
        # would corrupt the expert dispatch. Native indices are always
        # distinct, so revert any duplicate row to native routing.
        sorted_sub, _ = substituted.sort(dim=-1)
        has_duplicate = (sorted_sub[:, 1:] == sorted_sub[:, :-1]).any(dim=-1)
        return torch.where(has_duplicate.unsqueeze(-1), top_indices, substituted)


# --------------------------------------------------------------------------- #
# Generic monkeypatch for *SparseMoeBlock.forward                             #
# --------------------------------------------------------------------------- #

# Track patched classes so install is idempotent (the same block class is
# shared by every decoder layer; we patch the class once).
_patched_classes: set[type] = set()


def _make_replay_forward(orig_forward):
    """Build the patched ``SparseMoeBlock.forward``.

    Arch-generic for Qwen3-MoE and Qwen3-Next (byte-identical through the
    routing decision). Keys off the block's own ``top_k`` / ``norm_topk_prob``
    / softmax gating. Computes the LIVE softmax over ``self.gate(h)``, forces
    ``selected_experts`` from the controller, then RE-GATHERS ``routing_weights``
    from the live softmax (the R3 crux — NOT the rollout's weights) so router +
    expert grads flow. The expert loop and any shared-expert tail are unchanged.
    """
    import torch.nn.functional as F

    def _replay_forward(self, hidden_states: torch.Tensor):
        controller = get_active_replay()
        # Flag-off / no controller installed ⇒ exact stock HF behaviour.
        if controller is None or controller.action is not RouterReplayAction.REPLAY:
            return orig_forward(self, hidden_states)

        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        # LIVE softmax over the full expert set (full routing distribution).
        routing_weights_full = F.softmax(router_logits, dim=1, dtype=torch.float)
        # Natural topk (the fallback path for masked / duplicate rows).
        natural_weights, natural_experts = torch.topk(routing_weights_full, self.top_k, dim=-1)

        # Force selected experts from the rollout via the controller.
        selected_experts = controller.on_router_forward(self, routing_weights_full, natural_experts)
        # R3 crux: re-gather weights from the LIVE softmax for the forced set.
        routing_weights = routing_weights_full.gather(1, selected_experts)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # cast back to input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits

    return _replay_forward


def install_router_replay_patch(model) -> int:
    """Monkeypatch the MoE block class(es) in ``model`` for router replay.

    Discovers ``*SparseMoeBlock`` submodules (class name ends with
    ``SparseMoeBlock``), patches each distinct class's ``forward`` once, and
    returns the number of MoE block INSTANCES found (== number of MoE layers).
    Idempotent per class.
    """
    moe_block_count = 0
    classes_seen: set[type] = set()
    for module in model.modules():
        cls = type(module)
        if cls.__name__.endswith("SparseMoeBlock"):
            moe_block_count += 1
            classes_seen.add(cls)
    for cls in classes_seen:
        if cls in _patched_classes:
            continue
        orig_forward = cls.forward
        cls.forward = _make_replay_forward(orig_forward)
        _patched_classes.add(cls)
    return moe_block_count


def count_moe_layers(hf_config) -> int:
    """Count MoE layers in a model config (mirrors vLLM's _count_moe_layers).

    - Nemotron-style: explicit ``layers_block_type`` list with "moe" entries.
    - Qwen3-MoE / DeepSeek sparse: ``decoder_sparse_step > 1`` with optional
      ``mlp_only_layers`` exclusions.
    - Default: every layer is MoE except those in ``mlp_only_layers``.
    """
    layers_block_type = getattr(hf_config, "layers_block_type", None)
    if layers_block_type is not None:
        return list(layers_block_type).count("moe")
    n = hf_config.num_hidden_layers
    mlp_only = getattr(hf_config, "mlp_only_layers", None) or []
    step = getattr(hf_config, "decoder_sparse_step", 1) or 1
    if step > 1:
        return sum(1 for i in range(n) if (i + 1) % step == 0 and i not in mlp_only)
    return n - sum(1 for i in mlp_only if 0 <= i < n)
