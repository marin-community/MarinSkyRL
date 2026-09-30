"""Compiled vLLM's Grug layer on one GPU: archived Inductor kernels, FA3 prefill and Triton fused MoE.

The Inductor pieces run from the archived output code (``replay.run_call``). The two ops that sit
outside them are called directly with vLLM's own kernels: attention through ``flash_attn_varlen_func``
with a paged bf16 KV cache as ``FlashAttentionImpl`` passes it, and the routed experts through
``TritonExperts`` on each expert-parallel rank's experts, whose bf16 partial sums are then added as a
reduce-scatter model says (``expert_parallel.reduce_partials``). Nothing here changes vLLM's code or
settings; eager vLLM is never run.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG,
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.expert_map_manager import determine_expert_map
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.models.grugmoe import GrugMoeRouter
from vllm.utils.torch_utils import canonicalize_singleton_dim_strides
from vllm.vllm_flash_attn import flash_attn_varlen_func

from skyrl_train.mismatch_harness.expert_parallel import ReduceOrder, reduce_partials
from skyrl_train.models.grug_moe import grug_long_layer_flags

FLASH_ATTENTION_VERSION = 3
KV_BLOCK_SIZE = 16


@dataclass(frozen=True)
class GrugShape:
    """The Grug dimensions both engines read from the Hugging Face config."""

    hidden: int
    heads: int
    kv_heads: int
    head_dim: int
    experts: int
    top_k: int
    intermediate: int
    layers: int
    sliding_window: int
    rope_theta: float
    max_seq_len: int

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> GrugShape:
        return cls(
            hidden=int(config["hidden_size"]),
            heads=int(config["num_attention_heads"]),
            kv_heads=int(config["num_key_value_heads"]),
            head_dim=int(config["head_dim"]),
            experts=int(config["num_local_experts"]),
            top_k=int(config["num_experts_per_tok"]),
            intermediate=int(config["intermediate_size"]),
            layers=int(config["num_hidden_layers"]),
            sliding_window=int(config["sliding_window"]),
            rope_theta=float(config["rope_theta"]),
            max_seq_len=int(config["max_position_embeddings"]),
        )

    def is_long(self, layer: int) -> bool:
        return grug_long_layer_flags(self.layers)[layer]


def vllm_config_context():
    """vLLM's custom ops and MoE kernels read the current config; a default one fixes no numerics here."""
    return set_current_vllm_config(VllmConfig())


def cos_sin_cache(shape: GrugShape) -> torch.Tensor:
    """The bf16 rotary table vLLM builds on the GPU at model init (half-RoPE, NeoX pairing)."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("cuda"):
            rotary = get_rope(
                shape.head_dim,
                max_position=shape.max_seq_len,
                rope_parameters={"rope_theta": shape.rope_theta, "partial_rotary_factor": 0.5},
                is_neox_style=True,
            )
    finally:
        torch.set_default_dtype(previous)
    return rotary.cos_sin_cache


# vLLM's CUDA-graph capture sizes for the probe engine (engine log ``cudagraph_capture_sizes``): a step of
# at most 512 tokens runs the compiled pieces padded up to the next size.
CUDA_GRAPH_CAPTURE_SIZES = (1, 2, 4, *range(8, 257, 8), *range(272, 513, 16))
MAX_CUDA_GRAPH_CAPTURE_SIZE = CUDA_GRAPH_CAPTURE_SIZES[-1]
# ``attention_config.flash_attn_max_num_splits_for_cuda_graph``: the split cap FA3 gets on steps a CUDA
# graph could hold.
FA3_MAX_NUM_SPLITS_FOR_CUDA_GRAPH = 32


def cuda_graph_padded_tokens(tokens: int) -> int:
    """The token count vLLM runs its compiled pieces with for a step of ``tokens`` scheduled tokens."""
    return next((size for size in CUDA_GRAPH_CAPTURE_SIZES if size >= tokens), tokens)


def fa3_num_splits(step_tokens: int) -> int:
    """The ``num_splits`` vLLM passes FA3 for a step of ``step_tokens`` scheduled tokens.

    The probe's engines run FULL_AND_PIECEWISE CUDA graphs, so ``FlashAttentionMetadataBuilder`` caps
    the split count of every step of at most the largest capture size, prefill steps included, and
    passes 0 (FA3's own heuristic) for larger steps. The count of splits FA3 then uses is chosen per
    request on the device, and it decides how the key blocks are summed.
    """
    return FA3_MAX_NUM_SPLITS_FOR_CUDA_GRAPH if step_tokens <= MAX_CUDA_GRAPH_CAPTURE_SIZE else 0


@dataclass(frozen=True)
class Requests:
    """Token rows of the prefill batch: one request per trainer sequence, flattened request-major.

    The compiled pieces and the MoE run on ``rows`` rows (the step's padded token count); attention
    reads only the ``tokens`` real rows, as vLLM's attention op does.
    """

    lengths: tuple[int, ...]
    padded: int = 0

    @property
    def tokens(self) -> int:
        return sum(self.lengths)

    @property
    def rows(self) -> int:
        return max(self.padded, self.tokens)

    def positions(self) -> torch.Tensor:
        real = torch.cat([torch.arange(length, dtype=torch.int64) for length in self.lengths])
        return pad_rows(real, self.rows).cuda()

    def cu_seqlens(self) -> torch.Tensor:
        return torch.tensor([0, *torch.tensor(self.lengths).cumsum(0).tolist()], dtype=torch.int32).cuda()


def pad_rows(tensor: torch.Tensor, rows: int) -> torch.Tensor:
    """Zero rows appended so the first dimension is ``rows``."""
    if tensor.shape[0] >= rows:
        return tensor
    padding = torch.zeros(rows - tensor.shape[0], *tensor.shape[1:], dtype=tensor.dtype, device=tensor.device)
    return torch.cat((tensor, padding))


def paged_kv_cache(
    key: torch.Tensor, value: torch.Tensor, lengths: tuple[int, ...], block_size: int = KV_BLOCK_SIZE
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A bf16 paged KV cache holding each request's keys and values in whole blocks of its own.

    The storage is vLLM's LBNHC layout (block, token, KV head, key and value channels), and the key and
    value caches are its two channel halves with singleton strides canonicalized, as
    ``FlashAttentionImpl.forward`` derives them from the ``[blocks, kv_heads, block_size, 2 * head_dim]``
    tensor it is given. Returns the key cache, the value cache and the block table.
    """
    kv_heads, head_dim = key.shape[1], key.shape[2]
    blocks_per_request = [-(-length // block_size) for length in lengths]
    storage = torch.zeros(
        sum(blocks_per_request), block_size, kv_heads, 2 * head_dim, dtype=key.dtype, device=key.device
    )
    key_cache, value_cache = (canonicalize_singleton_dim_strides(half) for half in storage.split(head_dim, dim=-1))
    block_table = torch.zeros(len(lengths), max(blocks_per_request), dtype=torch.int32, device=key.device)
    start_block = start_token = 0
    for index, (length, blocks) in enumerate(zip(lengths, blocks_per_request, strict=True)):
        block_table[index, :blocks] = torch.arange(start_block, start_block + blocks, dtype=torch.int32)
        slots = torch.arange(length, device=key.device)
        key_cache[start_block + slots // block_size, slots % block_size] = key[start_token : start_token + length]
        value_cache[start_block + slots // block_size, slots % block_size] = value[start_token : start_token + length]
        start_block += blocks
        start_token += length
    return key_cache, value_cache, block_table


def flash_attention_prefill(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    requests: Requests,
    *,
    window: int | None,
    scale: float,
    num_splits: int,
    block_size: int = KV_BLOCK_SIZE,
) -> torch.Tensor:
    """FA3 varlen prefill over a paged bf16 KV cache, with the arguments ``FlashAttentionImpl`` passes.

    ``query`` is ``[tokens, heads, head_dim]``; ``key``/``value`` are ``[tokens, kv_heads, head_dim]``.
    Grug mixes windowed and full layers, so vLLM runs without an ahead-of-time schedule;
    ``num_splits`` is the step's split cap (``fa3_num_splits``).
    """
    key_cache, value_cache, block_table = paged_kv_cache(key, value, requests.lengths, block_size)
    output = torch.empty_like(query)
    descale = torch.ones(len(requests.lengths), key.shape[1], dtype=torch.float32, device=key.device)
    flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        out=output,
        cu_seqlens_q=requests.cu_seqlens(),
        max_seqlen_q=max(requests.lengths),
        seqused_k=torch.tensor(requests.lengths, dtype=torch.int32, device=key.device),
        max_seqlen_k=max(requests.lengths),
        softmax_scale=scale,
        causal=True,
        alibi_slopes=None,
        window_size=[window - 1, 0] if window is not None else None,
        block_table=block_table,
        softcap=0,
        scheduler_metadata=None,
        fa_version=FLASH_ATTENTION_VERSION,
        q_descale=None,
        k_descale=descale,
        v_descale=descale,
        num_splits=num_splits,
    )
    return output


@dataclass(frozen=True)
class MoeRecord:
    """What the MoE op computed for one call."""

    vllm_ids: torch.Tensor
    vllm_weights: torch.Tensor
    """vLLM's own routing of the router logits it received."""
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    """The routing the experts ran with (vLLM's, or the trainer's when given)."""
    slots: torch.Tensor
    """Per-slot expert outputs ``[tokens, top_k, hidden]``, route weight applied, from the owning rank."""
    partials: torch.Tensor
    output: torch.Tensor


@dataclass
class VllmMoe:
    """``vllm::moe_forward`` for one Grug layer, emulating DP/EP over ``ep_size`` ranks on one GPU.

    ``routing`` optionally replaces vLLM's routing with given ``(topk_ids, topk_weights)`` so the experts
    see the trainer's routes; ``config_tokens`` is the gathered token count the Triton config is
    chosen for (vLLM chooses it from the all-gathered batch, not from this rank's tokens).
    """

    bias: torch.Tensor
    w13: torch.Tensor
    w2: torch.Tensor
    top_k: int
    ep_size: int
    home_rank: int
    order: ReduceOrder
    config_tokens: int | None = None
    routing: tuple[torch.Tensor, torch.Tensor] | None = None
    records: list[MoeRecord] = field(default_factory=list)

    def route(self, hidden_states: torch.Tensor, router_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        router = GrugMoeRouter(top_k=self.top_k, global_num_experts=self.w13.shape[0], bias=self.bias)
        weights, ids = router._compute_routing(hidden_states, router_logits, None)
        return weights, ids

    def experts(
        self, hidden_states: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-rank bf16 partial sums ``[ep_size, tokens, hidden]`` and per-slot outputs ``[tokens, top_k, hidden]``.

        Rows past the routed tokens pad the batch to ``config_tokens``; they route to experts
        ``0..top_k-1`` and are dropped.
        """
        experts, intermediate2, hidden = self.w13.shape
        tokens = hidden_states.shape[0]
        padded = self.config_tokens or tokens
        if padded < tokens:
            raise ValueError(f"config token count {padded} is below the {tokens} tokens routed")
        inputs = torch.zeros(padded, hidden, dtype=hidden_states.dtype, device=hidden_states.device)
        inputs[:tokens] = hidden_states
        ids = torch.arange(self.top_k, dtype=topk_ids.dtype, device=topk_ids.device).repeat(padded, 1)
        ids[:tokens] = topk_ids
        weights = torch.zeros(padded, self.top_k, dtype=topk_weights.dtype, device=topk_weights.device)
        weights[:tokens] = topk_weights
        partials = []
        slots = torch.zeros(tokens, self.top_k, hidden, dtype=hidden_states.dtype, device=hidden_states.device)
        for rank in range(self.ep_size):
            local, expert_map, _ = determine_expert_map(self.ep_size, rank, experts)
            if expert_map is not None:
                expert_map = expert_map.to(hidden_states.device)
                owned = (expert_map >= 0).nonzero().flatten()
                w13, w2 = self.w13[owned].contiguous(), self.w2[owned].contiguous()
                mine = expert_map[topk_ids.long()] >= 0
            else:
                w13, w2 = self.w13, self.w2
                mine = torch.ones_like(topk_ids, dtype=torch.bool)
            kernel = TritonExperts(
                _moe_config(experts, local, self.top_k, hidden, intermediate2 // 2), FUSED_MOE_UNQUANTIZED_CONFIG
            )
            workspace13_shape, workspace2_shape, output_shape = kernel.workspace_shapes(
                padded, intermediate2, hidden, self.top_k, experts, local, None, MoEActivation.SILU
            )
            workspace2 = torch.empty(workspace2_shape, dtype=inputs.dtype, device=inputs.device)
            output = torch.empty(output_shape, dtype=inputs.dtype, device=inputs.device)
            kernel.apply(
                output,
                inputs,
                w13,
                w2,
                weights,
                ids,
                MoEActivation.SILU,
                experts,
                expert_map,
                None,
                None,
                torch.empty(workspace13_shape, dtype=inputs.dtype, device=inputs.device),
                workspace2,
                None,
                False,
            )
            # TritonExperts keeps the weighted per-slot outputs (intermediate_cache3) at the start of workspace2.
            cache3 = workspace2.flatten()[: padded * self.top_k * hidden].view(padded, self.top_k, hidden)
            slots = torch.where(mine.unsqueeze(-1), cache3[:tokens], slots)
            partials.append(output[:tokens].clone())
        return torch.stack(partials), slots

    def __call__(self, hidden_states, router_logits, shared_experts_input, input_ids, layer_name, hidden_dim_unpadded):
        del shared_experts_input, input_ids, layer_name, hidden_dim_unpadded
        vllm_weights, vllm_ids = self.route(hidden_states, router_logits)
        topk_ids, topk_weights = vllm_ids, vllm_weights
        if self.routing is not None:
            # The given routing covers the real tokens; padding rows keep vLLM's own routing.
            ids, weights = self.routing
            topk_ids = torch.cat((ids.to(vllm_ids.dtype), vllm_ids[ids.shape[0] :]))
            topk_weights = torch.cat((weights.to(vllm_weights.dtype), vllm_weights[weights.shape[0] :]))
        partials, slots = self.experts(hidden_states, topk_weights, topk_ids)
        output = reduce_partials(partials, self.order, self.home_rank)
        self.records.append(MoeRecord(vllm_ids, vllm_weights, topk_ids, topk_weights, slots, partials, output))
        return output


def _moe_config(experts: int, local: int, top_k: int, hidden: int, intermediate: int) -> FusedMoEConfig:
    return FusedMoEConfig(
        num_experts=experts,
        experts_per_token=top_k,
        hidden_dim=hidden,
        intermediate_size=intermediate,
        num_local_experts=local,
        num_logical_experts=experts,
        activation=MoEActivation.SILU,
        device="cuda",
        routing_method=RoutingMethodType.Custom,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        in_dtype=torch.bfloat16,
        router_logits_dtype=torch.float32,
    )


MoeOp = Callable[..., torch.Tensor]
