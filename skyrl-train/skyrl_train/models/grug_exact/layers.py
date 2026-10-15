"""The Megatron layer-spec modules of the exact Grug trainer.

Each module computes its values with the decode-invariant engine's kernels (``inductor_regions``, ``vllm_kernels``,
``ep_combine``, the invariant GEMM) and gives them the gradient of the trainer's own chain for the same value
(``rounding``, ``reference_kernels``). The exact model (``model``) calls them with the forward's ``ExactForward``.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from megatron.core.transformer.moe.experts import TEGroupedMLP
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer

from skyrl_train.config.numerics import ALLTOALL_DISPATCHER
from skyrl_train.inference_engines.vllm.decode_invariant.gemm import invariant_gemm
from skyrl_train.models.grug_exact import inductor_regions as regions
from skyrl_train.models.grug_exact.context import ExactForward, ResidualSum
from skyrl_train.models.grug_exact.ep_combine import ep_combine
from skyrl_train.models.grug_exact.projections import qkv_projection, split_qkv
from skyrl_train.models.grug_exact.reference_kernels import (
    gated_product_value,
    hybrid_input_norm_value,
    logits_value,
    query_key_values,
    swiglu_value,
    xsa_head_gate_value,
)
from skyrl_train.models.grug_exact.rounding import kernel_value, rms_norm_single_rounding
from skyrl_train.models.grug_exact.vllm_kernels import expert_outputs, fa3_attention_sbhd, topk_experts
from skyrl_train.models.grug_megatron import GrugGatedRMSNorm, GrugSelfAttention, GrugTopKRouter


class _ResidualSumGradient(torch.autograd.Function):
    """``value``, the bf16 ``residual + mlp`` summed in fp32, differentiated as that expression."""

    @staticmethod
    def forward(ctx, value, residual, mlp):
        return value

    @staticmethod
    def backward(ctx, grad):
        return None, grad, grad


class _ExactGatedNorm(GrugGatedRMSNorm):
    """Grug's gated RMSNorm with the gate's product from the engine's kernel; subclasses compute the norm."""

    def __init__(self, config: TransformerConfig, hidden_size: int, eps: float):
        super().__init__(config, hidden_size, eps)
        self.eps = eps

    def gate_logits(self, normalized: torch.Tensor) -> torch.Tensor:
        """The gate's logits with the engine's SiLU kernel, differentiated as ``F.silu``."""
        hidden = self.down_proj(normalized)
        return self.up_proj(kernel_value(regions.gate_silu(hidden), lambda: F.silu(hidden)))

    def gated(self, normalized: torch.Tensor) -> torch.Tensor:
        """``normalized * sigmoid(gate)`` from the engine's kernel, differentiated as the fp32 product rounded once."""
        gate = self.gate_logits(normalized)
        product = regions.gated_product(normalized, gate).view_as(normalized)
        return gated_product_value(product, normalized, gate)


class ExactGatedNorm(_ExactGatedNorm):
    """The gated norm after the attention: the engine's RMSNorm of the stored bf16 hidden states."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        value = regions.rms_norm(hidden_states, self.norm.weight).view_as(hidden_states)
        return self.gated(kernel_value(value, lambda: self.norm(hidden_states)))


class ExactInputGatedNorm(_ExactGatedNorm):
    """A layer's input gated norm: the engine normalizes the stored bf16 sum by the statistic of the unrounded sum
    that formed it, which the previous layer or stage hands on."""

    def forward(self, hidden_states: torch.Tensor, statistic: torch.Tensor) -> torch.Tensor:
        weight = self.norm.weight
        value = regions.rms_norm_from_square_sum(hidden_states, statistic, weight).view_as(hidden_states)
        variance = (statistic / hidden_states.shape[-1]).view(*hidden_states.shape[:-1], 1)
        return self.gated(hybrid_input_norm_value(value, hidden_states, variance, weight, self.eps))


class ExactFinalGatedNorm(_ExactGatedNorm):
    """The final gated norm: the engine normalizes the last layer's unrounded sum."""

    def forward(self, hidden_states: torch.Tensor, parts: ResidualSum) -> torch.Tensor:
        weight = self.norm.weight
        # The gradient flows through the bf16 sum's exactly-zero ``residual - residual.detach()``.
        residual = hidden_states.float()

        def reference() -> torch.Tensor:
            unrounded = (parts.residual.float() + parts.mlp.float()) + (residual - residual.detach())
            return rms_norm_single_rounding(unrounded, weight, self.eps)

        value = regions.final_norm(parts.residual, parts.mlp, weight).view_as(hidden_states)
        return self.gated(kernel_value(value, reference))


class ExactEmbeddingGatedNorm(_ExactGatedNorm):
    """The embedding gated norm; returns its output and layer 0's input-norm statistic of the unrounded product."""

    def forward(self, embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        value = regions.embedding_norm(embeddings, self.norm.weight).view_as(embeddings)
        normalized = kernel_value(value, lambda: self.norm(embeddings))
        gate = self.gate_logits(normalized)
        product, statistic = regions.gated_product_square_sum(normalized, gate)
        return gated_product_value(product.view_as(normalized), normalized, gate), statistic


class ExactSelfAttention(GrugSelfAttention):
    """Grug attention with the engine's q/k chain, FA3 attention, XSA with the head gate, and GEMM shapes."""

    def __init__(self, config: TransformerConfig, submodules, layer_number: int, **kwargs):
        super().__init__(config, submodules, layer_number, **kwargs)
        if any(
            linear.tp_size != 1 or linear.use_bias for linear in (self.linear_qkv, self.linear_proj, self.attn_gate)
        ):
            raise NotImplementedError("exact attention needs unsharded, bias-free projections")
        if self.sconv_k is not None or self.sconv_attn is not None or self.logical_kv_heads != config.num_query_groups:
            raise NotImplementedError("exact attention does not cover Hero's ShortConv or per-layer KV heads")
        is_long = self.skip_rope
        # The engine applies the two query factors one after the other.
        self.qk_mult = config.grug_qk_mult
        self.qk_mult_scale = config.grug_qk_mult_long_scale if is_long else 1.0
        # The engine's FA3 call for this layer: a sliding window of window_size[0] + 1 keys on local layers.
        self.fa3_window = None if is_long else config.window_size[0] + 1
        self.fa3_scale = (
            config.softmax_scale if config.softmax_scale is not None else 1.0 / math.sqrt(config.kv_channels)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        rotary_pos_emb,
        kept_attention: torch.Tensor | None,
        keep: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """``residual + attention(hidden_states)`` ``[S, B, H]``, the residual added in the output projection's
        epilogue as the engine adds it, and, when ``keep``, the FA3 output to keep for the recompute, which reuses it
        as ``kept_attention``."""
        groups = self.num_query_groups_per_partition
        head_dim = self.hidden_size_per_attention_head
        query_width = self.num_attention_heads_per_partition // groups * head_dim
        mixed = qkv_projection(hidden_states, self.linear_qkv.weight, groups, query_width, head_dim)
        query, key, value = split_qkv(mixed, groups, query_width, head_dim)
        query, key = self._query_key(query, key, rotary_pos_emb)
        if kept_attention is None:
            with torch.no_grad():
                attention = fa3_attention_sbhd(query, key, value, window=self.fa3_window, scale=self.fa3_scale)
        else:
            attention = kept_attention
        to_keep = attention if keep and kept_attention is None else None
        if torch.is_grad_enabled():
            attention = kernel_value(
                attention,
                lambda: self.core_attention(query, key, value, None, attn_mask_type=self.attn_mask_type),
            )
        rows = hidden_states.reshape(-1, hidden_states.shape[-1])
        # The engine reads the fp32 gate logits as they are: Inductor fuses the sigmoid into the XSA kernel.
        gate = invariant_gemm(rows, self.attn_gate.weight).view(*hidden_states.shape[:-1], -1)
        gate = logits_value(gate, hidden_states, self.attn_gate.weight)
        xsa = regions.xsa_head_gate(attention, value, gate).view_as(attention)
        gated = xsa_head_gate_value(xsa, attention, value, gate, head_dim)
        width = hidden_states.shape[-1]
        output = torch.addmm(residual.reshape(-1, width), gated.reshape(-1, width), self.linear_proj.weight.t())
        return output.view_as(hidden_states), to_keep

    def _query_key(self, query: torch.Tensor, key: torch.Tensor, rotary_pos_emb) -> tuple[torch.Tensor, torch.Tensor]:
        """The engine's q/k norm, RoPE and query scale on the raw ``[S, B, heads, dim]`` projections, differentiated
        as ``rounded_query_key(qk_norm_fp32(query), qk_norm_fp32(key), ...)``."""
        rotary = rotary_pos_emb is not None and not self.skip_rope
        if rotary:
            engine_query, engine_key = regions.query_key_rope(query, key, query.shape[0], self.config.rotary_base)
        else:
            engine_query, engine_key = regions.query_key_full(query, key)
        freqs = None
        if rotary:
            freqs = rotary_pos_emb if isinstance(rotary_pos_emb, tuple) else (rotary_pos_emb,) * 2
        return query_key_values(
            engine_query.view_as(query), engine_key.view_as(key), query, key, freqs, self.qk_mult, self.qk_mult_scale
        )


class ExactTopKRouter(GrugTopKRouter):
    """Grug's router with the engine's invariant GEMM for the logits and its tie-breaking for the selection; the
    forward also returns the selected experts in the engine's slot order."""

    def __init__(self, config: TransformerConfig, pg_collection=None, is_mtp_layer: bool = False):
        super().__init__(config=config, pg_collection=pg_collection, is_mtp_layer=is_mtp_layer)
        if self.weight.dtype != torch.bfloat16:
            raise ValueError("the invariant router GEMM multiplies the bf16 router input by the bf16 router weight")

    def gating(self, input: torch.Tensor) -> torch.Tensor:
        if input.dtype != torch.bfloat16:
            raise ValueError(f"the invariant router GEMM takes a bf16 router input, got {input.dtype}")
        value = invariant_gemm(input.reshape(-1, input.shape[-1]), self.weight)
        return logits_value(value.view(*input.shape[:-1], value.shape[-1]), input, self.weight)

    def select_topk(self, biased_logits: torch.Tensor) -> torch.Tensor:
        return topk_experts(biased_logits, self.topk)

    def forward(self, input: torch.Tensor, padding_mask=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._maintain_float32_expert_bias()
        return self.route(self.gating(input))


class ExactSharedExpertMLP(SharedExpertMLP):
    """Grug's shared expert with the engine's GEMM shapes (gate and up as two GEMMs, the routed output added in the
    down projection's epilogue) and activation kernel."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.use_shared_expert_gate:
            raise NotImplementedError("the exact shared expert has no gate")
        if any(linear.tp_size != 1 or linear.use_bias for linear in (self.linear_fc1, self.linear_fc2)):
            raise NotImplementedError("the exact shared expert needs unsharded, bias-free projections")

    def forward(self, hidden_states: torch.Tensor, routed: torch.Tensor) -> torch.Tensor:
        """``routed + shared_expert(hidden_states)``, the sum formed in the down projection's epilogue."""
        gate_weight, up_weight = self.linear_fc1.weight.chunk(2, dim=0)
        gate, up = F.linear(hidden_states, gate_weight), F.linear(hidden_states, up_weight)
        value = regions.shared_activation(gate, up).view_as(gate)
        activation = swiglu_value(value, torch.cat((gate, up), dim=-1))
        width = routed.shape[-1]
        output = torch.addmm(
            routed.reshape(-1, width), activation.reshape(-1, activation.shape[-1]), self.linear_fc2.weight.t()
        )
        return output.view_as(routed)


class ExactGroupedMLP(TEGroupedMLP):
    """Grug's routed experts with each dispatched slot's value from the engine's fused-MoE kernels, differentiated as
    the grouped GEMMs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if any(linear.tp_size != 1 or linear.use_bias for linear in (self.linear_fc1, self.linear_fc2)):
            raise NotImplementedError("exact routed experts need unsharded, bias-free projections")

    def engine_forward(self, permuted_local_hidden_states, tokens_per_expert, permuted_probs):
        """The experts' output with the fused-MoE kernels' values; ``forward`` gives the grouped GEMMs' values."""
        with torch.no_grad():
            value = expert_outputs(
                permuted_local_hidden_states,
                tokens_per_expert.tolist(),
                permuted_probs,
                [getattr(self.linear_fc1, f"weight{index}") for index in range(self.linear_fc1.num_gemms)],
                [getattr(self.linear_fc2, f"weight{index}") for index in range(self.linear_fc2.num_gemms)],
            )
        if not torch.is_grad_enabled():
            return value, None
        reference, bias = super().forward(permuted_local_hidden_states, tokens_per_expert, permuted_probs)
        return kernel_value(value, lambda: reference), bias


class ExactMoELayer(MoELayer):
    """Grug's MoE layer: the routed output added in the engine's expert-parallel order, then the shared expert's
    output added as the engine adds it."""

    def __init__(self, config: TransformerConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        if not self.use_shared_expert or self.shared_expert_overlap or config.moe_latent_size:
            raise NotImplementedError("the exact MoE layer needs one shared expert outside the dispatcher")
        if config.moe_token_dispatcher_type != ALLTOALL_DISPATCHER:
            raise NotImplementedError(f"the exact MoE layer runs on the {ALLTOALL_DISPATCHER} token dispatcher")

    def forward(self, hidden_states: torch.Tensor, context: ExactForward, recompute: bool) -> torch.Tensor:
        probs, routing_map, selected = self.router(hidden_states)
        dispatched, probs = self.preprocess(hidden_states, probs, routing_map)
        dispatched, probs = self.dispatch(dispatched, probs)
        dispatched, tokens_per_expert, permuted_probs = self.token_dispatcher.dispatch_postprocess(dispatched, probs)
        if recompute:
            # A recompute only rebuilds the gradient, which the grouped GEMMs' forward carries.
            output, _ = self.experts(dispatched, tokens_per_expert, permuted_probs)
        else:
            output, _ = self.experts.engine_forward(dispatched, tokens_per_expert, permuted_probs)
        output = self.combine(self.token_dispatcher.combine_preprocess(output))
        routed = self._combine_postprocess(output, selected, context)
        return self.shared_experts(hidden_states, routed)

    def _combine_postprocess(self, output: torch.Tensor, selected: torch.Tensor, context: ExactForward) -> torch.Tensor:
        """Each token's routed output added as the engine adds it, differentiated as the trainer's unpermute."""
        dispatcher = self.token_dispatcher
        ranks = context.row_serving_ranks(selected.shape[0])
        with torch.no_grad():
            combined = ep_combine(output, dispatcher.routing_map, selected, ranks, context.expert_parallel_size)
        combined = combined.view(dispatcher.hidden_shape)
        if not torch.is_grad_enabled():
            return combined
        return kernel_value(combined, lambda: dispatcher.combine_postprocess(output))


class ExactTransformerLayer(TransformerLayer):
    """A Grug decoder layer whose forward hands its input-norm statistic and residual sum to the next consumer."""

    def forward(
        self, hidden_states: torch.Tensor, statistic: torch.Tensor, rotary_pos_emb, context: ExactForward
    ) -> tuple[torch.Tensor, torch.Tensor, ResidualSum]:
        """The layer output, the next input norm's statistic of its unrounded sum, and the sum's parts."""
        kept = context.kept_attention(self.layer_number)
        recompute = kept is not None
        normalized = self.input_layernorm(hidden_states, statistic)
        hidden_states, to_keep = self.self_attention(
            normalized, hidden_states, rotary_pos_emb, kept, keep=not recompute
        )
        if to_keep is not None:
            context.keep(self.layer_number, to_keep)
        mlp = self.mlp(self.pre_mlp_layernorm(hidden_states), context, recompute)
        output, square_sum = regions.residual_square_sum(hidden_states, mlp)
        output = output.view_as(hidden_states)
        if torch.is_grad_enabled():
            output = _ResidualSumGradient.apply(output, hidden_states, mlp)
        return output, square_sum, ResidualSum(hidden_states.detach(), mlp.detach())
