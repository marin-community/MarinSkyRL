"""Swap HF eager ``*SparseMoeBlock`` → grouped-GEMM ``MoE`` — Stage 3b.

Gated on the ``moe_grouped_gemm`` flag in ``model_wrapper.__init__`` (default
False → no swap, HF eager block untouched, byte-identical to today). When on,
each HF ``Qwen3MoeSparseMoeBlock`` / ``Qwen3NextSparseMoeBlock`` instance is
replaced (in the parent decoder layer's ``.mlp`` attribute) by a thin
``GroupedMoEShim`` wrapping a freshly-built ``MoE`` whose weights are remapped
from the HF block. The shim returns the HF decoder-layer 2-tuple
``(out, router_logits_or_None)`` so the unchanged HF decoder code
(``hidden_states = self.mlp(hidden_states)`` / 2-tuple unpack) keeps working.

Replay transport: the shim reads the Stage-2 ``RouterReplay`` singleton via
``get_active_replay()`` and, when active, threads the controller's per-layer
forced-index slice into the native ``MoE`` router's ``routed_experts`` arg. The
``model_wrapper.forward`` replay-install seam is therefore UNCHANGED between the
eager (3a) and grouped (3b) paths — same singleton, same per-layer targets, same
``[N, K]`` contract.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from skyrl_train.models.layers.moe import MoE
from skyrl_train.models.layers.moe_weight_remap import remap_hf_block_to_moe
from skyrl_train.models.router_replay import RouterReplayAction, get_active_replay


class GroupedMoEShim(nn.Module):
    """Wraps a grouped ``MoE`` to satisfy the HF decoder-layer ``self.mlp(h)``
    2-tuple contract and to apply router replay via the native ``routed_experts``
    arg.

    HF ``*SparseMoeBlock.forward`` returns ``(final_hidden_states, router_logits)``;
    the decoder layer either ignores the second element or threads it to the
    aux-loss. Stage 3b drops aux-loss, so we return ``router_logits=None`` (the
    decoder code tolerates None for the dropped element).
    """

    def __init__(self, moe: MoE):
        super().__init__()
        self.moe = moe

    def _replay_indices(self, hidden_flat: torch.Tensor) -> Optional[torch.Tensor]:
        """Resolve forced top-k indices for the current rows via the controller.

        Computes the LIVE softmax + natural topk (so the controller's per-token
        replay-mask / duplicate fallback reverts to native routing where needed),
        calls ``on_router_forward`` to obtain the substituted indices, and returns
        them as a ``(N, top_k)`` tensor for the native router's ``routed_experts``
        arg. Returns None when no controller is active (→ natural routing).
        """
        controller = get_active_replay()
        if controller is None or controller.action is not RouterReplayAction.REPLAY:
            return None
        router = self.moe.router
        scores = router.gate(hidden_flat)
        if router.score_func == "sigmoid":
            scores = torch.sigmoid(scores.to(torch.float32))
        else:
            scores = F.softmax(scores.to(torch.float32), dim=1)
        _, natural_experts = torch.topk(scores, router.top_k, dim=-1)
        # Controller applies the per-token replay mask + duplicate fallback; the
        # native router re-gathers live scores from these forced indices.
        return controller.on_router_forward(self.moe, scores, natural_experts)

    def forward(self, hidden_states: torch.Tensor):
        bs, slen, dim = hidden_states.shape
        forced = self._replay_indices(hidden_states.view(-1, dim))
        if forced is not None:
            routed_experts = forced.view(bs, slen, -1)
            out = self.moe(hidden_states, routed_experts=routed_experts)
        else:
            out = self.moe(hidden_states)
        # 2-tuple contract; aux-loss dropped → router_logits is None.
        return out, None


def _build_moe_for_block(hf_block, hf_config) -> MoE:
    """Construct a grouped ``MoE`` mirroring the dims of an HF ``*SparseMoeBlock``."""
    dim = hf_block.gate.weight.shape[1]
    num_experts = hf_block.num_experts
    top_k = hf_block.top_k
    route_norm = bool(hf_block.norm_topk_prob)

    # Routed-expert intermediate size.
    if hasattr(hf_block.experts, "gate_up_proj"):
        hidden_dim = hf_block.experts.gate_up_proj.shape[1] // 2
    else:
        hidden_dim = hf_block.experts[0].gate_proj.weight.shape[0]

    shared = getattr(hf_block, "shared_expert", None)
    shared_dim = None
    shared_gated = False
    if shared is not None:
        shared_dim = shared.gate_proj.weight.shape[0]
        shared_gated = getattr(hf_block, "shared_expert_gate", None) is not None

    moe = MoE(
        dim=dim,
        hidden_dim=hidden_dim,
        num_experts=num_experts,
        top_k=top_k,
        route_norm=route_norm,
        score_func="softmax",
        use_grouped_mm=False,  # EP=1 for-loop parity default; grouped_mm = Stage-4 perf path
        shared_expert_dim=shared_dim,
        shared_expert_gated=shared_gated,
    )
    return moe


def swap_moe_blocks_to_grouped(model) -> int:
    """Replace every HF ``*SparseMoeBlock`` in ``model`` with a ``GroupedMoEShim``.

    Walks the module tree, and for each parent whose ``.mlp`` is a
    ``*SparseMoeBlock``, builds a grouped ``MoE`` (matching dims + device/dtype),
    remaps the HF weights into it, wraps it in a ``GroupedMoEShim``, and assigns
    it back to ``parent.mlp``. Returns the number of blocks swapped.

    Must run AFTER model load and BEFORE FSDP2 wrap (see model_wrapper).
    """
    hf_config = model.config
    swapped = 0
    for parent in model.modules():
        block = getattr(parent, "mlp", None)
        if block is None or not type(block).__name__.endswith("SparseMoeBlock"):
            continue
        moe = _build_moe_for_block(block, hf_config)
        # Match the source block's device/dtype before copying weights.
        ref_param = block.gate.weight
        moe = moe.to(device=ref_param.device, dtype=ref_param.dtype)
        remap_hf_block_to_moe(block, moe)
        parent.mlp = GroupedMoEShim(moe)
        swapped += 1
    return swapped
