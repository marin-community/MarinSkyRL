"""Megatron-Core modules for training the Grug MoE policy with ``trainer.strategy=megatron``.

Grug differs from a stock Megatron GPT model in a handful of places that the
layer spec cannot express through configuration alone:

* every norm (embedding, per-layer, final) is followed by a low-rank sigmoid
  gate (``GrugGatedRMSNorm``);
* queries and keys use a weightless RMS norm and a per-layer query scale;
* attention output is projected away from the value direction (XSA) and
  scaled by a per-head sigmoid gate computed from the attention input;
* the router selects the top-(k+1) experts on biased logits, drops the last
  one, and renormalizes sigmoid weights of the survivors.

Everything else (sliding window on local layers, RoPE skipped on long layers,
half-RoPE, grouped-GEMM experts with a shared expert, GQA) maps onto stock
Megatron-Core settings chosen by ``GrugModelProvider`` in
``grug_megatron_bridge``.
"""

import torch
import torch.nn.functional as F
from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TENorm
from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec_for_backend
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import (
    TransformerBlock,
    TransformerBlockSubmodules,
    get_num_layers_to_build,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.typed_torch import apply_module
from torch import nn

from skyrl_train.models.grug_moe import (
    GRUG_GATED_NORM_RANK,
    GRUG_ROUTER_RENORM_EPS,
    GRUG_ROUTING_RENORM_SUM,
    GRUG_XSA_EPS,
    grug_long_layer_flags,
    grug_rms_norm_no_weight,
    jax_top_k,
)


def _first_present(preferred: torch.Tensor | None, fallback: torch.Tensor | None) -> torch.Tensor | None:
    return fallback if preferred is None else preferred


class GrugGatedRMSNorm(nn.Module):
    """RMSNorm followed by Grug's low-rank sigmoid gate: ``norm(x) * sigmoid(up(silu(down(norm(x)))))``."""

    def __init__(self, config: TransformerConfig, hidden_size: int, eps: float):
        super().__init__()
        self.norm = TENorm(config=config, hidden_size=hidden_size, eps=eps)
        device = torch.cuda.current_device()
        self.down_proj = nn.Linear(
            hidden_size, GRUG_GATED_NORM_RANK, bias=False, device=device, dtype=config.params_dtype
        )
        self.up_proj = nn.Linear(
            GRUG_GATED_NORM_RANK, hidden_size, bias=False, device=device, dtype=config.params_dtype
        )
        # Replicated across TP ranks; the attribute makes Megatron all-reduce their grads under SP.
        for param in (self.down_proj.weight, self.up_proj.weight):
            param.sequence_parallel = config.sequence_parallel

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(hidden_states)
        gate = torch.sigmoid(self.up_proj(F.silu(self.down_proj(normalized))))
        return normalized * gate


class GrugQKNorm(nn.Module):
    """Weightless RMS norm applied to each query and key head."""

    def __init__(self, config: TransformerConfig, hidden_size: int, eps: float):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return grug_rms_norm_no_weight(hidden_states)


class GrugSelfAttention(SelfAttention):
    """Grug attention on top of Megatron's fused-QKV self attention.

    Compared with ``SelfAttention.forward`` this drops inference support and
    adds the per-layer query scale, XSA, and the per-head output gate. RoPE
    is skipped entirely on long layers via ``config.no_rope_freq``.
    """

    def __init__(self, config: TransformerConfig, submodules: SelfAttentionSubmodules, layer_number: int, **kwargs):
        super().__init__(config, submodules, layer_number, **kwargs)
        self.attn_gate = TEColumnParallelLinear(
            config.hidden_size,
            config.num_attention_heads,
            config=config,
            init_method=config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="attn_gate",
            tp_group=self.pg_collection.tp,
        )
        is_long = grug_long_layer_flags(config.num_layers)[layer_number - 1]
        self.query_scale = config.grug_qk_mult * (config.grug_qk_mult_long_scale if is_long else 1.0)
        self.skip_rope = bool(config.no_rope_freq[layer_number - 1])

    def forward(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_context=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        rotary_pos_cos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        *,
        inference_params=None,
    ):
        if inference_context is not None or inference_params is not None:
            raise NotImplementedError("GrugSelfAttention only supports training forwards")
        if rotary_pos_cos is not None or rotary_pos_sin is not None or rotary_pos_cos_sin is not None:
            raise NotImplementedError("GrugSelfAttention applies RoPE from rotary_pos_emb only")

        query, key, value = self.get_query_key_value_tensors(hidden_states, key_value_states)

        is_thd = packed_seq_params is not None and packed_seq_params.qkv_format == "thd"
        if is_thd:
            query, key, value = query.squeeze(1), key.squeeze(1), value.squeeze(1)

        if rotary_pos_emb is not None and not self.skip_rope:
            if not isinstance(rotary_pos_emb, tuple):
                rotary_pos_emb = (rotary_pos_emb,) * 2
            q_pos_emb, k_pos_emb = rotary_pos_emb
            cu_seqlens_q = cu_seqlens_kv = None
            if is_thd:
                cu_seqlens_q = _first_present(packed_seq_params.cu_seqlens_q_padded, packed_seq_params.cu_seqlens_q)
                cu_seqlens_kv = _first_present(packed_seq_params.cu_seqlens_kv_padded, packed_seq_params.cu_seqlens_kv)
            query = apply_rotary_pos_emb(
                query, q_pos_emb, config=self.config, cu_seqlens=cu_seqlens_q, cp_group=self.pg_collection.cp
            )
            key = apply_rotary_pos_emb(
                key, k_pos_emb, config=self.config, cu_seqlens=cu_seqlens_kv, cp_group=self.pg_collection.cp
            )

        query = query * self.query_scale

        # Grug is causal-only and the trainer removes left padding, so right-padded keys sit after
        # every valid query and the causal (or causal sliding-window) mask alone is exact.
        attention_mask = None
        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=self.attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )
        else:
            core_attn_out = self._run_core_attention(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=self.attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )

        core_attn_out = self._apply_xsa(core_attn_out, value)
        if is_thd:
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

        gate, _ = self.attn_gate(hidden_states)
        core_attn_out = self._apply_head_gate(core_attn_out, gate)
        output, bias = apply_module(self.linear_proj)(core_attn_out)
        return output, bias

    def _apply_xsa(self, core_attn_out: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Remove each head's component along its (GQA-expanded) value vector."""

        heads = core_attn_out.view(
            *value.shape[:-2], self.num_attention_heads_per_partition, self.hidden_size_per_attention_head
        )
        expanded_value = value.repeat_interleave(
            self.num_attention_heads_per_partition // self.num_query_groups_per_partition, dim=-2
        )
        out = heads.float()
        v = expanded_value.float()
        dot = (out * v).sum(dim=-1, keepdim=True)
        v_norm = v.square().sum(dim=-1, keepdim=True)
        out = out - (dot / (v_norm + GRUG_XSA_EPS)) * v
        return out.to(core_attn_out.dtype).reshape(core_attn_out.shape)

    def _apply_head_gate(self, core_attn_out: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """Scale every head by ``2 * sigmoid(attn_gate(x))``."""

        heads = core_attn_out.view(*gate.shape, self.hidden_size_per_attention_head)
        gated = heads * (2.0 * torch.sigmoid(gate.float())).unsqueeze(-1).to(heads.dtype)
        return gated.reshape(core_attn_out.shape)


class GrugTopKRouter(TopKRouter):
    """Grug routing: biased top-(k+1) selection with sigmoid weights renormalized to a fixed sum.

    The persistent fp32 ``expert_bias`` buffer is Grug's frozen query bias. It
    only steers expert selection; the combine weights come from the unbiased
    logits of the first ``k`` selected experts.
    """

    def __init__(self, config: TransformerConfig, pg_collection=None, is_mtp_layer: bool = False):
        super().__init__(config=config, pg_collection=pg_collection, is_mtp_layer=is_mtp_layer)
        assert self.enable_expert_bias, "GrugTopKRouter requires moe_router_enable_expert_bias=True"

    def routing(self, logits: torch.Tensor, padding_mask: torch.Tensor | None = None):
        logits = logits.view(-1, self.config.num_moe_experts).float()
        biased_logits = logits + self.expert_bias
        _, topk_indices = jax_top_k(biased_logits, self.topk + 1)
        selected = topk_indices[:, : self.topk]
        combine = torch.sigmoid(torch.gather(logits, dim=-1, index=selected))
        combine = combine * (GRUG_ROUTING_RENORM_SUM / (combine.sum(dim=-1, keepdim=True) + GRUG_ROUTER_RENORM_EPS))
        probs = torch.zeros_like(logits).scatter(1, selected, combine)
        routing_map = torch.zeros_like(logits, dtype=torch.bool).scatter(1, selected, True)
        return probs, routing_map

    def forward(self, input: torch.Tensor, padding_mask: torch.Tensor | None = None):
        self._maintain_float32_expert_bias()
        logits = self.gating(input)
        return self.routing(logits, padding_mask)


class GrugGPTModel(GPTModel):
    """GPTModel with Grug's gated embedding norm on the first pipeline stage."""

    def __init__(self, config: TransformerConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        if self.pre_process:
            self.embed_norm = GrugGatedRMSNorm(
                config=config, hidden_size=config.hidden_size, eps=config.layernorm_epsilon
            )

    def _preprocess(
        self,
        input_ids,
        position_ids,
        decoder_input=None,
        inference_context=None,
        packed_seq_params=None,
        padding_mask=None,
    ):
        apply_embed_norm = self.pre_process and decoder_input is None
        outputs = super()._preprocess(
            input_ids,
            position_ids,
            decoder_input=decoder_input,
            inference_context=inference_context,
            packed_seq_params=packed_seq_params,
            padding_mask=padding_mask,
        )
        if not apply_embed_norm:
            return outputs
        decoder_input, *rest = outputs
        return (self.embed_norm(decoder_input), *rest)


def grug_layer_spec(config: TransformerConfig) -> ModuleSpec:
    """Build the Transformer Engine layer spec for one Grug decoder layer."""

    backend = TESpecProvider()
    moe = get_moe_module_spec_for_backend(backend, num_experts=config.num_moe_experts, moe_grouped_gemm=True)
    moe.keywords["submodules"].router = GrugTopKRouter
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=GrugGatedRMSNorm,
            self_attention=ModuleSpec(
                module=GrugSelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=backend.column_parallel_linear(),
                    core_attention=backend.core_attention(),
                    linear_proj=backend.row_parallel_linear(),
                    q_layernorm=GrugQKNorm,
                    k_layernorm=GrugQKNorm,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=GrugGatedRMSNorm,
            mlp=moe,
            mlp_bda=get_bias_dropout_add,
        ),
    )


def grug_block_spec(config: TransformerConfig, vp_stage: int | None, pp_rank: int) -> ModuleSpec:
    """Build this pipeline stage's decoder block spec with Grug's gated final norm."""

    layer_spec = grug_layer_spec(config)
    num_layers = get_num_layers_to_build(config, vp_stage=vp_stage, pp_rank=pp_rank)
    return ModuleSpec(
        module=TransformerBlock,
        submodules=TransformerBlockSubmodules(layer_specs=[layer_spec] * num_layers, layer_norm=GrugGatedRMSNorm),
    )
