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

ARCH SUPPORT MATRIX (what swap_moe_blocks_to_grouped can handle today):
  * ``Qwen3MoeSparseMoeBlock``  — supported (bare-tensor return).
  * ``Qwen3NextSparseMoeBlock`` — supported (2-tuple return).
  * ``OlmoeSparseMoeBlock`` (allenai/OLMoE-1B-7B-*, transformers 5.10.1) —
    supported (bare-tensor return; see _BARE_TENSOR_BLOCKS). Router is
    ``block.gate`` (``OlmoeTopKRouter``: plain softmax top-k + optional
    norm_topk_prob, weight at ``block.gate.weight`` [num_experts, hidden] →
    softmax-faithful to the native MoE router). Experts are FUSED nn.Parameter
    (``experts.gate_up_proj``/``experts.down_proj``) → fused remap branch. No
    shared expert. Topology: 64 experts, top_k 8, norm_topk_prob False.

NOT SUPPORTED — ``phimoe`` (microsoft/Phi-3.5-MoE-instruct, transformers 5.10.1).
PhimoeSparseMoeBlock's class name endswith "SparseMoeBlock" (so the scan would
match) BUT it is structurally incompatible: (1) its router is ``block.router``
(``PhimoeTopKRouter``), not ``block.gate`` → _build_moe_for_block's
``hf_block.gate.weight`` read AttributeErrors; (2) routing uses ``sparsemixer``
(custom mask-based top-2 with jitter), NOT plain softmax/sigmoid top-k, so the
native softmax MoE router would be numerically WRONG (same class of gap as
gemma4's per-expert scale); (3) experts are fused nn.Parameter (that part the
fused branch could handle, but 1+2 are blockers). Supporting phimoe needs a
sparsemixer-aware router + ``block.router`` detection — structural work, not a
flag flip. Use OLMoE as the small-MoE non-Qwen counterpart instead.

NOT SUPPORTED — ``gemma4`` (google/gemma-4-26B-A4B-it, transformers 5.10.1).
Gemma4's MoE is STRUCTURALLY different from both Qwen3 variants and this shim
CANNOT swap it without new code. Verified against the SIF modeling file
(transformers/models/gemma4/modeling_gemma4.py). The gaps:
  1. No ``*SparseMoeBlock`` wrapper exists. The MoE is composed DIRECTLY into
     ``Gemma4TextDecoderLayer`` as TWO separate attributes — ``self.router``
     (``Gemma4TextRouter``) and ``self.experts`` (``Gemma4TextExperts``) — gated
     by ``self.enable_moe_block``. swap_moe_blocks_to_grouped scans for
     ``parent.mlp`` whose class endswith "SparseMoeBlock" → matches NOTHING for
     gemma4 (returns 0 swaps; the model trains on the slow HF for-loop experts).
  2. Each gemma4 decoder layer ALSO has a DENSE ``self.mlp`` (``Gemma4TextMLP``)
     that runs IN PARALLEL with the MoE; the layer combines them as
     ``post_ffn_ln_1(mlp_out) + post_ffn_ln_2(moe_out)``. Any future swapper
     must target ``self.experts``+``self.router``, NOT ``self.mlp`` (dense).
  3. Router differs: ``Gemma4TextRouter`` uses ``self.proj`` (not ``self.gate``),
     applies an RMSNorm + ``self.scale``*scalar_root + a learned
     ``self.per_expert_scale[idx]`` gather on the top-k weights, and returns a
     3-tuple ``(router_probabilities, top_k_weights, top_k_index)``. The native
     ``MoE`` router (gate-only softmax/sigmoid) does not model per-expert scale.
  4. Expert weights are FUSED ``nn.Parameter`` tensors on ``Gemma4TextExperts``
     (``gate_up_proj`` ``[E, 2*moe_inter, hidden]``, ``down_proj``
     ``[E, hidden, moe_inter]``) — no ``hf_block.gate.weight`` and no
     per-expert ``gate_proj/up_proj/down_proj`` submodules, so both
     ``_build_moe_for_block`` (reads ``hf_block.gate.weight`` / ``hf_block.experts``)
     and ``remap_hf_block_to_moe`` would AttributeError/KeyError.
  5. The decoder applies SEPARATE pre/post RMSNorms around the MoE branch
     (``pre_feedforward_layernorm_2`` / ``post_feedforward_layernorm_2``) that
     live on the layer, not inside any swappable block — the grouped ``MoE``
     would need these threaded in to stay numerically faithful.
Supporting gemma4 requires a NEW block-detection + build + weight-remap +
(optionally) router-replay path keyed on ``Gemma4TextExperts``, plus handling
the parallel dense MLP and the per-expert router scale. This is structural work,
NOT a ``returns_tuple`` flag flip — DO NOT attempt a blind swap. Topology:
128 experts, top_k 8, moe_intermediate_size 704, hidden 2816, 30 layers.
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
    contract and to apply router replay via the native ``routed_experts`` arg.

    The return contract is arch-dependent and set at construction by
    ``swap_moe_blocks_to_grouped`` from the class of the HF block being replaced:

    * ``Qwen3NextSparseMoeBlock`` (``returns_tuple=True``, DEFAULT): the
      Qwen3-Next decoder unpacks a 2-tuple
      (``hidden_states, _ = self.mlp(...)`` / ``isinstance(..., tuple)`` guard),
      so the shim returns ``(out, router_logits_or_None)``. Aux-loss is dropped
      in Stage 3b, so ``router_logits`` is None. This is the validated 80B path
      and is the default to keep it byte-identical.
    * ``Qwen3MoeSparseMoeBlock`` (``returns_tuple=False``): the stock
      ``Qwen3MoeSparseMoeBlock.forward`` returns a BARE TENSOR, and the
      Qwen3-MoE decoder consumes it directly (``hidden_states = self.mlp(h)``
      then ``hidden_states = residual + hidden_states`` — no tuple unpack).
      Returning a 2-tuple here raised
      ``TypeError: unsupported operand type(s) for +: 'Tensor' and 'tuple'`` at
      ``modeling_qwen3_moe.py`` (residual add). So the shim must return the bare
      tensor for this arch.
    """

    def __init__(self, moe: MoE, returns_tuple: bool = True):
        super().__init__()
        self.moe = moe
        # Arch-dependent return contract; default True = Qwen3-Next 2-tuple
        # (preserves the validated 80B path byte-identically).
        self.returns_tuple = returns_tuple

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
        if self.returns_tuple:
            # Qwen3-Next decoder unpacks a 2-tuple; aux-loss dropped → None.
            return out, None
        # Qwen3-MoE decoder consumes a bare tensor (residual + self.mlp(h)).
        return out


def _moe_attr(hf_block, hf_config, name):
    """Resolve a MoE topology attribute (num_experts / top_k / norm_topk_prob)
    across HF arch variants. Qwen3-MoE exposes these directly on the
    ``*SparseMoeBlock``; Qwen3-Next moves them onto its router submodule
    (``block.gate`` = ``Qwen3NextTopKRouter``, which carries ``top_k`` /
    ``num_experts`` / ``norm_topk_prob``). Fall back to the HF config last
    (config field names: num_experts, num_experts_per_tok, norm_topk_prob).

    For ``norm_topk_prob`` specifically, if it is exposed nowhere (block / gate /
    config) default to ``True`` — the Qwen3.5/3.6 family
    (``Qwen3_5MoeForConditionalGeneration``) DROPPED ``norm_topk_prob`` as a
    config field (it lives on neither the ``Qwen3_5MoeSparseMoeBlock``, its
    ``.gate``, nor the top-level/text config), yet its native router renormalizes
    the top-k weights. This matches vLLM's own resolution for the family
    (``renormalize=getattr(config, "norm_topk_prob", True)`` in qwen3_next.py), so
    the grouped MoE stays bit-faithful. Topology attrs (num_experts / top_k) still
    hard-fail if unresolvable — a missing expert count is a real error, not a
    defaultable one."""
    if hasattr(hf_block, name):
        return getattr(hf_block, name)
    gate = getattr(hf_block, "gate", None)
    if gate is not None and hasattr(gate, name):
        return getattr(gate, name)
    cfg_alias = {"top_k": "num_experts_per_tok"}.get(name, name)
    if hasattr(hf_config, cfg_alias):
        return getattr(hf_config, cfg_alias)
    if name == "norm_topk_prob":
        # Faithful default for families that drop the field (Qwen3.5/3.6); mirrors
        # vLLM's getattr(config, "norm_topk_prob", True). Only norm_topk_prob is
        # defaulted — topology attrs below still raise.
        return True
    raise AttributeError(
        f"could not resolve MoE attribute '{name}' on {type(hf_block).__name__}, "
        f"its .gate, or the HF config"
    )


# Blocks whose native router UNCONDITIONALLY renormalizes the top-k weights
# (top_scores /= top_scores.sum) but expose NO ``norm_topk_prob`` on the block,
# its router, or the HF config. For these, route_norm must be forced True (the
# native softmax MoE router with route_norm=True is then bit-faithful). Keyed by
# the HF block class name.
#   * MixtralSparseMoeBlock — MixtralTopKRouter always does
#       ``router_top_value /= router_top_value.sum(dim=-1, keepdim=True)``; the
#       Mixtral config has no ``norm_topk_prob`` field, so _moe_attr would raise.
_ROUTE_NORM_ALWAYS_BLOCKS = {"MixtralSparseMoeBlock"}


def _build_moe_for_block(hf_block, hf_config) -> MoE:
    """Construct a grouped ``MoE`` mirroring the dims of an HF ``*SparseMoeBlock``."""
    dim = hf_block.gate.weight.shape[1]
    num_experts = _moe_attr(hf_block, hf_config, "num_experts")
    top_k = _moe_attr(hf_block, hf_config, "top_k")
    if type(hf_block).__name__ in _ROUTE_NORM_ALWAYS_BLOCKS:
        # Native router always renormalizes top-k weights but exposes no
        # ``norm_topk_prob`` to introspect → force True (see set docstring).
        route_norm = True
    else:
        route_norm = bool(_moe_attr(hf_block, hf_config, "norm_topk_prob"))

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

    Fails fast on known-unsupported MoE arches (gemma4) instead of silently
    swapping 0 blocks and training on the slow HF for-loop experts — see the
    module docstring "ARCH SUPPORT MATRIX" for why gemma4 needs new code.
    """
    hf_config = model.config
    # Guard: gemma4 composes its MoE as separate Gemma4TextRouter +
    # Gemma4TextExperts on the decoder layer (no *SparseMoeBlock), so the scan
    # below would no-op silently. Refuse rather than mislead. See module
    # docstring for the structural-shim work this needs.
    _model_type = getattr(hf_config, "model_type", "") or ""
    _text_type = getattr(getattr(hf_config, "text_config", None), "model_type", "") or ""
    if "gemma4" in _model_type or "gemma4" in _text_type:
        raise NotImplementedError(
            "moe_grouped_gemm is not supported for gemma4 (model_type="
            f"{_model_type!r}). gemma4's MoE is composed directly into "
            "Gemma4TextDecoderLayer as separate Gemma4TextRouter + "
            "Gemma4TextExperts (no *SparseMoeBlock), with a parallel dense MLP, "
            "fused expert nn.Parameters, and a per-expert router scale — see "
            "moe_swap.py ARCH SUPPORT MATRIX. Set moe_grouped_gemm=false (HF "
            "eager experts) until a gemma4 swap path is implemented."
        )
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
        # The shim must mirror the replaced HF block's return contract. Some HF
        # MoE decoders consume a BARE TENSOR (residual + self.mlp(h)); others
        # unpack a 2-tuple (hidden_states, router_logits). The default (no match)
        # is the 2-tuple, preserving the validated Qwen3-Next 80B path.
        #
        # BARE-TENSOR blocks (return final_hidden_states; decoder does
        #   `hidden_states = self.mlp(h)` then `residual + hidden_states`):
        #   * Qwen3MoeSparseMoeBlock — verified (TypeError otherwise, see shim docstring).
        #   * OlmoeSparseMoeBlock    — transformers 5.10.1: OlmoeSparseMoeBlock.forward
        #       returns a bare tensor; OlmoeDecoderLayer does `residual + self.mlp(h)`.
        #       Router is `block.gate` (OlmoeTopKRouter, plain softmax-topk + optional
        #       norm_topk_prob → faithful to the native MoE softmax router), weight at
        #       `block.gate.weight` [num_experts, hidden]; experts are FUSED nn.Parameter
        #       (`experts.gate_up_proj`/`experts.down_proj`) → handled by the fused branch
        #       in _build_moe_for_block / remap_hf_block_to_moe. No shared expert.
        #   * MixtralSparseMoeBlock  — transformers 5.10.1: MixtralSparseMoeBlock.forward
        #       returns a BARE TENSOR (`return hidden_states`, despite the `-> tuple`
        #       annotation — the 5.x refactor moved routing into MixtralTopKRouter +
        #       MixtralExperts); MixtralDecoderLayer does `hidden_states = self.mlp(h)`
        #       then `residual + hidden_states`. Router is `block.gate`
        #       (MixtralTopKRouter: F.linear(hidden, weight) → softmax(float32) → topk →
        #       `router_top_value /= sum` renorm), weight at `block.gate.weight`
        #       [num_experts, hidden] → faithful to the native MoE softmax router with
        #       route_norm=True. Experts are FUSED nn.Parameter
        #       (`experts.gate_up_proj` [E, 2*inter, hidden] / `experts.down_proj`
        #       [E, hidden, inter]) → handled by the fused branch in
        #       _build_moe_for_block / remap_hf_block_to_moe. No shared expert.
        #       Topology: 8 experts, top_k 2, intermediate_size 14336, hidden 4096.
        #       NOTE: Mixtral always renormalizes the top-k weights but exposes NO
        #       `norm_topk_prob` (config or router attr) → _build_moe_for_block must
        #       default route_norm True for it (see _MoE_ROUTE_NORM_ALWAYS below).
        # 2-TUPLE blocks: Qwen3NextSparseMoeBlock (default), etc.
        _BARE_TENSOR_BLOCKS = {"Qwen3MoeSparseMoeBlock", "OlmoeSparseMoeBlock", "MixtralSparseMoeBlock"}
        returns_tuple = type(block).__name__ not in _BARE_TENSOR_BLOCKS
        parent.mlp = GroupedMoEShim(moe, returns_tuple=returns_tuple)
        swapped += 1
    return swapped
