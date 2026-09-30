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

import math
import weakref
from enum import StrEnum

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
from megatron.core.transformer.moe.experts import TEGroupedMLP
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
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

from skyrl_train.mismatch_probe.numerics import active_numerics
from skyrl_train.models.grug_rounding import (
    gated_norm_product_fp32,
    rms_norm_hybrid,
    rms_norm_single_rounding,
    weighted_down_projection_single_rounding,
    rotate_neox_fp32,
    swiglu_single_rounding,
    xsa_and_gate_single_rounding,
)
from skyrl_train.models.grug_vllm_kernels import (
    fa3_attention_sbhd,
    vllm_ep_combine,
    vllm_expert_outputs,
    vllm_qkv_projection,
)
from skyrl_train.models.grug_moe import (
    GRUG_ATTN_GATE_SCALE,
    GRUG_GATED_NORM_RANK,
    GRUG_QK_RMS_NORM_EPS,
    GRUG_ROUTER_RENORM_EPS,
    GRUG_ROUTING_RENORM_SUM,
    GRUG_XSA_EPS,
    grug_long_layer_flags,
    grug_rms_norm_no_weight,
    jax_top_k,
)


def _first_present(preferred: torch.Tensor | None, fallback: torch.Tensor | None) -> torch.Tensor | None:
    return fallback if preferred is None else preferred


# Unrounded residual sums (and the embedding's unrounded gated-norm product) awaiting the next norm,
# keyed by the bf16 tensor that norm receives. Each entry keeps a weak reference to that tensor: under
# pipeline parallelism several micro-batches are in flight and a stage's last residual has no reader, so a
# freed tensor's ``id()`` can come back for an unrelated tensor, which must not receive the entry.
_RESIDUAL_FP32: dict[int, tuple[weakref.ref, torch.Tensor]] = {}
# Compiled vLLM pads a GEMM's output width to a multiple of this many columns (the 20-head gate becomes 24).
VLLM_GEMM_OUTPUT_ALIGNMENT = 8


def _hand_off(receiver: torch.Tensor, unrounded: torch.Tensor) -> None:
    for key in [key for key, (ref, _) in _RESIDUAL_FP32.items() if ref() is None]:
        del _RESIDUAL_FP32[key]
    _RESIDUAL_FP32[id(receiver)] = weakref.ref(receiver), unrounded


def _take_hand_off(receiver: torch.Tensor) -> torch.Tensor | None:
    entry = _RESIDUAL_FP32.pop(id(receiver), None)
    if entry is None or entry[0]() is not receiver:
        return None
    return entry[1]


def _install_residual_hooks(layer: TransformerLayer) -> None:
    """Rebuild the MLP residual in fp32 for the ``mlp_residual`` and norm-input numerics."""
    stored: dict[str, torch.Tensor] = {}

    def active() -> bool:
        numerics = active_numerics()
        return numerics.mlp_residual or numerics.input_norm_variance or numerics.final_norm_fp32

    def keep_residual(module, args):
        if active():
            stored["residual"] = args[0]

    postprocess = layer.mlp.postprocess
    combine_postprocess = layer.mlp.token_dispatcher.combine_postprocess

    def keep_routed(output):
        routed = combine_postprocess(output)
        if active():
            stored["routed"] = routed
        return routed

    def keep_shared(output, shared_expert_output):
        if active():
            if shared_expert_output is None:
                raise RuntimeError("Grug residual numerics expect a shared expert")
            stored["shared"] = shared_expert_output
        return postprocess(output, shared_expert_output)

    def mlp_residual(module, args, output):
        if not active():
            return None
        numerics = active_numerics()
        routed, shared = stored.pop("routed"), stored.pop("shared")
        # Compiled vLLM: h + (routed + shared) in fp32; the trainer rounds routed + shared first.
        unrounded = stored["residual"].float() + (routed.float() + shared.float())
        if numerics.mlp_residual:
            hidden = unrounded.to(output[0].dtype)
        else:
            hidden = (stored["residual"].float() + (routed + shared).float()).to(output[0].dtype)
        stored.pop("residual")
        if numerics.input_norm_variance or numerics.final_norm_fp32:
            _hand_off(hidden, unrounded)
        return (hidden, *output[1:])

    layer.pre_mlp_layernorm.register_forward_pre_hook(keep_residual)
    layer.mlp.token_dispatcher.combine_postprocess = keep_routed
    layer.mlp.postprocess = keep_shared
    layer.register_forward_hook(mlp_residual)


def _install_route_weight_hooks(experts: TEGroupedMLP) -> None:
    """Apply route weights after the fp32 down projection when ``route_weight`` is active."""
    stored: dict[str, object] = {}

    def active() -> bool:
        # ``vllm_experts`` computes the weighted down projection in vLLM's kernel and replaces this flag.
        numerics = active_numerics()
        return numerics.route_weight and not numerics.vllm_experts

    def unit_probs(module, args, kwargs):
        if not active():
            return None
        if module._with_fused_impl:
            raise NotImplementedError("route_weight numerics require the unfused grouped-MLP path")
        hidden, tokens_per_expert, probs = args
        stored["probs"], stored["splits"] = probs, tokens_per_expert.tolist()
        return (hidden, tokens_per_expert, torch.ones_like(probs)), kwargs

    def weighted_output(module, args, output):
        if not active():
            return None
        activation = args[0]
        weights = [getattr(module, f"weight{index}") for index in range(module.num_gemms)]
        result = weighted_down_projection_single_rounding(
            activation, weights, stored.pop("splits"), stored.pop("probs").unsqueeze(-1)
        )
        return (result, output[1]) if isinstance(output, tuple) else result

    experts.register_forward_pre_hook(unit_probs, with_kwargs=True)
    experts.linear_fc2.register_forward_hook(weighted_output)


def _install_vllm_experts_hooks(experts: TEGroupedMLP) -> None:
    """Take the routed experts' values from vLLM's fused-MoE kernels when ``vllm_experts`` is active.

    The kernels compute each dispatched row (one token-expert slot) as vLLM does, with the route weight inside
    the down projection's fp32 accumulator. With gradients enabled the trainer's grouped-GEMM experts also run
    and carry the gradient under the kernels' bytes through the exact-zero ``x - x.detach()``; a scoring forward
    without gradients runs the kernels alone.
    """
    grouped_forward = experts.forward

    def forward(permuted_local_hidden_states, tokens_per_expert, permuted_probs):
        if not active_numerics().vllm_experts:
            return grouped_forward(permuted_local_hidden_states, tokens_per_expert, permuted_probs)
        if any(linear.tp_size != 1 or linear.use_bias for linear in (experts.linear_fc1, experts.linear_fc2)):
            raise NotImplementedError("vllm_experts numerics need unsharded, bias-free expert projections")
        with torch.no_grad():
            value = vllm_expert_outputs(
                permuted_local_hidden_states,
                tokens_per_expert.tolist(),
                permuted_probs,
                [getattr(experts.linear_fc1, f"weight{index}") for index in range(experts.linear_fc1.num_gemms)],
                [getattr(experts.linear_fc2, f"weight{index}") for index in range(experts.linear_fc2.num_gemms)],
            )
        if not torch.is_grad_enabled():
            return value, None
        reference, bias = grouped_forward(permuted_local_hidden_states, tokens_per_expert, permuted_probs)
        return value + (reference - reference.detach()), bias

    experts.forward = forward


def _vllm_gemm(linear: nn.Module, compute) -> None:
    """Run ``linear`` (a Transformer Engine linear) as ``compute(x)`` when ``vllm_gemm`` is active.

    ``compute`` issues ``torch.mm`` in compiled vLLM's shapes on the module's own weight, so autograd gives the
    weight its gradient; Megatron's gradient hooks add it to the parameter's main gradient as for any
    parameter Transformer Engine did not already accumulate.
    """
    te_forward = linear.forward

    def forward(x, *args, **kwargs):
        if not active_numerics().vllm_gemm:
            return te_forward(x, *args, **kwargs)
        if args or kwargs or linear.tp_size != 1 or linear.use_bias:
            raise NotImplementedError("vllm_gemm numerics need an unsharded, bias-free projection of one input")
        return compute(x), None

    linear.forward = forward


def _install_vllm_gemm_attention_hooks(attention: "GrugSelfAttention") -> None:
    """Compiled vLLM's attention projections: q, k and v as three GEMMs, the head gate padded to 24 outputs."""
    qkv = attention.linear_qkv
    groups = attention.num_query_groups_per_partition
    head_dim = attention.hidden_size_per_attention_head
    query_width = attention.num_attention_heads_per_partition // groups * head_dim
    gate = attention.attn_gate

    def head_gate(x: torch.Tensor) -> torch.Tensor:
        # Inductor pads the gate projection's 20 outputs to a multiple of 8 with zero rows (fusion map, M-rope:1119).
        heads = gate.weight.shape[0]
        padded = torch.cat(
            (gate.weight, gate.weight.new_zeros(-heads % VLLM_GEMM_OUTPUT_ALIGNMENT, gate.weight.shape[1]))
        )
        return F.linear(x, padded)[..., :heads].contiguous()

    _vllm_gemm(qkv, lambda x: vllm_qkv_projection(x, qkv.weight, groups, query_width, head_dim))
    _vllm_gemm(gate, head_gate)
    _vllm_gemm(attention.linear_proj, lambda x: F.linear(x, attention.linear_proj.weight))


def _install_vllm_gemm_shared_hooks(shared: SharedExpertMLP) -> None:
    """Compiled vLLM's shared expert: gate and up projections as two GEMMs, then the down projection."""
    fc1, fc2 = shared.linear_fc1, shared.linear_fc2

    def gate_and_up(x: torch.Tensor) -> torch.Tensor:
        gate_weight, up_weight = fc1.weight.chunk(2, dim=0)
        return torch.cat((F.linear(x, gate_weight), F.linear(x, up_weight)), dim=-1)

    _vllm_gemm(fc1, gate_and_up)
    _vllm_gemm(fc2, lambda x: F.linear(x, fc2.weight))


def _install_fa3_attention_hooks(attention: "GrugSelfAttention") -> None:
    """Take the attention value from vLLM's FA3 forward when ``fa3_attention`` is active.

    The gradient is the trainer's cuDNN attention backward at the same query, key and value: with
    gradients enabled the cuDNN forward also runs, and ``x - x.detach()`` (exactly zero) carries its
    graph under FA3's bytes. A scoring forward without gradients runs FA3 alone.
    """
    core = attention.core_attention
    cudnn_forward = core.forward

    def forward(
        query, key, value, attention_mask, attn_mask_type=None, attention_bias=None, packed_seq_params=None, **kwargs
    ):
        def cudnn():
            return cudnn_forward(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
                **kwargs,
            )

        if not active_numerics().fa3_attention:
            return cudnn()
        if packed_seq_params is not None or attention_bias is not None or kwargs:
            raise NotImplementedError("fa3_attention numerics support unpacked causal sequences only")
        output = fa3_attention_sbhd(query, key, value, window=attention.fa3_window, scale=attention.fa3_scale)
        if not torch.is_grad_enabled():
            return output
        reference = cudnn()
        return output.detach() + (reference - reference.detach())

    core.forward = forward


def _install_ep_combine_hooks(layer: TransformerLayer) -> None:
    """Add the routed expert outputs in vLLM's expert-parallel order when ``ep_sum`` is active.

    The value comes from ``vllm_ep_combine``; the gradient is the trainer's own unpermute (each slot's
    gradient is the token's output gradient in both), through the same exact-zero difference.
    """
    dispatcher = layer.mlp.token_dispatcher
    router = layer.mlp.router
    unpermute = dispatcher.combine_postprocess

    def combine_postprocess(permuted):
        output = unpermute(permuted)
        if not active_numerics().ep_sum:
            return output
        if dispatcher.shared_experts is not None:
            raise NotImplementedError("ep_sum numerics expect the shared expert outside the dispatcher")
        selected, expert_parallel = router.take_ep_route()
        combined = vllm_ep_combine(
            permuted, dispatcher.routing_map, selected, expert_parallel.dp_ranks, expert_parallel.ep_size
        ).view_as(output)
        return combined.detach() + (output - output.detach())

    dispatcher.combine_postprocess = combine_postprocess


def _install_shared_swiglu_hooks(shared: SharedExpertMLP) -> None:
    """Recompute the shared expert's activation with one rounding when ``shared_swiglu`` is active."""
    stored: dict[str, torch.Tensor] = {}

    def keep_fc1_output(module, args, output):
        if active_numerics().shared_swiglu:
            stored["fc1"] = output[0] if isinstance(output, tuple) else output

    def replace_fc2_input(module, args):
        if not active_numerics().shared_swiglu:
            return None
        fc1_output = stored.pop("fc1", None)
        if fc1_output is None:
            raise RuntimeError("shared_swiglu requires the shared expert's fc1 output")
        return (swiglu_single_rounding(fc1_output), *args[1:])

    shared.linear_fc1.register_forward_hook(keep_fc1_output)
    shared.linear_fc2.register_forward_pre_hook(replace_fc2_input)


def install_numerics_hooks(root: nn.Module) -> None:
    """Install the switchable-numerics hooks on every Grug decoder layer under ``root``."""
    for module in root.modules():
        if isinstance(module, SharedExpertMLP):
            _install_shared_swiglu_hooks(module)
            _install_vllm_gemm_shared_hooks(module)
        if isinstance(module, TEGroupedMLP):
            _install_route_weight_hooks(module)
            _install_vllm_experts_hooks(module)
        if isinstance(module, GrugSelfAttention):
            _install_fa3_attention_hooks(module)
            _install_vllm_gemm_attention_hooks(module)
        if isinstance(module, TransformerLayer) and isinstance(module.pre_mlp_layernorm, GrugGatedRMSNorm):
            # The residual hooks wrap the EP combine, so the fp32 residual reads the combine's value.
            _install_ep_combine_hooks(module)
            _install_residual_hooks(module)
            module.input_layernorm.role = NormRole.INPUT


def clear_numerics_handoffs() -> None:
    """Drop the fp32 tensors the numerics hooks hand from one module to the next.

    Each forward starts empty: a pipeline stage's last residual has no reader, and the entries are keyed
    by ``id()`` of tensors that a later forward may reuse.
    """
    _RESIDUAL_FP32.clear()


class NormRole(StrEnum):
    """Where a gated norm sits, which decides the residual it reads under the norm-input numerics."""

    INPUT = "input"
    POST_ATTENTION = "post_attention"
    FINAL = "final"
    EMBEDDING = "embedding"


class GrugGatedRMSNorm(nn.Module):
    """RMSNorm followed by Grug's low-rank sigmoid gate: ``norm(x) * sigmoid(up(silu(down(norm(x)))))``."""

    def __init__(self, config: TransformerConfig, hidden_size: int, eps: float):
        super().__init__()
        self.role = NormRole.POST_ATTENTION
        self.eps = eps
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
        numerics = active_numerics()
        unrounded = _take_hand_off(hidden_states)
        if unrounded is not None:
            # The hand-off may come from a no-grad checkpointed forward; keep its values and take the
            # gradient through the bf16 residual so the loss still reaches the layers before this norm.
            # ``x - x.detach()`` is exactly zero, so the values are unchanged bit for bit.
            residual = hidden_states.float()
            unrounded = unrounded.detach() + (residual - residual.detach())
        if unrounded is not None and self.role is NormRole.INPUT and numerics.input_norm_variance:
            normalized = rms_norm_hybrid(hidden_states, unrounded, self.norm.weight, self.eps)
        elif unrounded is not None and self.role is NormRole.FINAL and numerics.final_norm_fp32:
            normalized = rms_norm_single_rounding(unrounded, self.norm.weight, self.eps)
        else:
            normalized = self.norm(hidden_states)
        gate = self.up_proj(F.silu(self.down_proj(normalized)))
        product = gated_norm_product_fp32(normalized, gate)
        output = product.to(normalized.dtype) if numerics.gated_norm else normalized * torch.sigmoid(gate)
        if self.role is NormRole.EMBEDDING and numerics.input_norm_variance:
            # Layer 0's input norm takes its variance from the unrounded embedding gated-norm product.
            _hand_off(output, product)
        return output


class GrugQKNorm(nn.Module):
    """Weightless RMS norm applied to each query and key head."""

    def __init__(self, config: TransformerConfig, hidden_size: int, eps: float):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if active_numerics().qk_rope:
            # The attention forward rounds once after RoPE and the query scale.
            fp32 = hidden_states.float()
            return fp32 * torch.rsqrt(fp32.square().mean(dim=-1, keepdim=True) + GRUG_QK_RMS_NORM_EPS)
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
        # Compiled vLLM applies the two query factors one after the other.
        self.qk_mult = config.grug_qk_mult
        self.qk_mult_scale = config.grug_qk_mult_long_scale if is_long else 1.0
        self.skip_rope = bool(config.no_rope_freq[layer_number - 1])
        # vLLM's FA3 call for this layer: a sliding window of window_size[0] + 1 keys on local layers.
        self.fa3_window = None if is_long else config.window_size[0] + 1
        self.fa3_scale = (
            config.softmax_scale if config.softmax_scale is not None else 1.0 / math.sqrt(config.kv_channels)
        )

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

        if active_numerics().qk_rope:
            if is_thd:
                raise NotImplementedError("qk_rope numerics support unpacked sequences only")
            if rotary_pos_emb is not None and not self.skip_rope:
                q_pos_emb, k_pos_emb = rotary_pos_emb if isinstance(rotary_pos_emb, tuple) else (rotary_pos_emb,) * 2
                rotary_dim = q_pos_emb.shape[-1]
                query = rotate_neox_fp32(query, q_pos_emb)
                # The rotated half of q is stored once before the scale; the pass-through half is not.
                query = torch.cat((query[..., :rotary_dim].to(value.dtype).float(), query[..., rotary_dim:]), dim=-1)
                key = rotate_neox_fp32(key, k_pos_emb)
            query = (query.float() * self.qk_mult * self.qk_mult_scale).to(value.dtype)
            key = key.to(value.dtype)
        else:
            if rotary_pos_emb is not None and not self.skip_rope:
                if not isinstance(rotary_pos_emb, tuple):
                    rotary_pos_emb = (rotary_pos_emb,) * 2
                q_pos_emb, k_pos_emb = rotary_pos_emb
                cu_seqlens_q = cu_seqlens_kv = None
                if is_thd:
                    cu_seqlens_q = _first_present(packed_seq_params.cu_seqlens_q_padded, packed_seq_params.cu_seqlens_q)
                    cu_seqlens_kv = _first_present(
                        packed_seq_params.cu_seqlens_kv_padded, packed_seq_params.cu_seqlens_kv
                    )
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

        gate, _ = self.attn_gate(hidden_states)
        if active_numerics().xsa_gate:
            if is_thd:
                raise NotImplementedError("xsa_gate numerics support unpacked sequences only")
            core_attn_out = xsa_and_gate_single_rounding(
                core_attn_out, value, gate, self.hidden_size_per_attention_head
            )
        else:
            core_attn_out = self._apply_xsa(core_attn_out, value)
            if is_thd:
                core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)
            core_attn_out = self._apply_head_gate(core_attn_out, gate)
        output, bias = apply_module(self.linear_proj)(core_attn_out)
        return output, bias

    def _apply_xsa(self, core_attn_out: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Remove each head's component along its (GQA-expanded) value vector."""

        # CoreAttention returns TP-local heads. Derive that count from its
        # output: the inherited head-count field can still be global here.
        local_heads = core_attn_out.shape[-1] // self.hidden_size_per_attention_head
        local_query_groups = value.shape[-2]
        if local_heads * self.hidden_size_per_attention_head != core_attn_out.shape[-1]:
            raise ValueError("Grug XSA received a partial attention head")
        if local_heads % local_query_groups:
            raise ValueError("Grug XSA local heads are not divisible by local query groups")
        heads = core_attn_out.view(*value.shape[:-2], local_heads, self.hidden_size_per_attention_head)
        expanded_value = value.repeat_interleave(local_heads // local_query_groups, dim=-2)
        out = heads.float()
        v = expanded_value.float()
        dot = (out * v).sum(dim=-1, keepdim=True)
        v_norm = v.square().sum(dim=-1, keepdim=True)
        out = out - (dot / (v_norm + GRUG_XSA_EPS)) * v
        return out.to(core_attn_out.dtype).reshape(core_attn_out.shape)

    def _apply_head_gate(self, core_attn_out: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """Scale every head by ``2 * sigmoid(attn_gate(x))``."""

        heads = core_attn_out.view(*gate.shape, self.hidden_size_per_attention_head)
        gated = heads * (GRUG_ATTN_GATE_SCALE * torch.sigmoid(gate.float())).unsqueeze(-1).to(heads.dtype)
        return gated.reshape(core_attn_out.shape)


def _grug_topk_indices(biased_logits: torch.Tensor, topk: int) -> torch.Tensor:
    """Return the first K indices from Grug's biased top-(K+1) selection."""
    _, indices = jax_top_k(biased_logits, topk + 1)
    return indices[:, :topk]


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
        if self.router_replay is None:
            selected = _grug_topk_indices(biased_logits, self.topk)
        else:
            # Grug overrides MCore's routing method, so its replay hook must be
            # called here. Selection uses biased logits; combine weights below
            # still use the original, unbiased logits.
            def native_topk(scores, topk, num_groups=None, group_topk=None):
                if num_groups is not None or group_topk is not None:
                    raise ValueError("Grug router replay does not support grouped top-k")
                indices = _grug_topk_indices(scores, topk)
                return scores.gather(1, indices), indices

            _, selected = self.router_replay.get_replay_topk(biased_logits, self.topk, None, None, native_topk)
        if active_numerics().ep_sum:
            # The combine adds the slots in vLLM's order: the selection order, per vLLM EP rank.
            expert_parallel = None if self.router_replay is None else self.router_replay.take_vllm_expert_parallel()
            if expert_parallel is None:
                raise ValueError("ep_sum numerics need each token's vLLM data-parallel rank from router replay")
            self._ep_route = (selected, expert_parallel)
        combine = torch.sigmoid(torch.gather(logits, dim=-1, index=selected))
        combine = combine * (GRUG_ROUTING_RENORM_SUM / (combine.sum(dim=-1, keepdim=True) + GRUG_ROUTER_RENORM_EPS))
        probs = torch.zeros_like(logits).scatter(1, selected, combine)
        routing_map = torch.zeros_like(logits, dtype=torch.bool).scatter(1, selected, True)
        return probs, routing_map

    def take_ep_route(self):
        """The slot order and vLLM placement of this router's last routing, for the ``ep_sum`` combine."""
        route = self.__dict__.pop("_ep_route", None)
        if route is None:
            raise RuntimeError("ep_sum combine ran without a routing from this layer's router")
        return route

    def forward(self, input: torch.Tensor, padding_mask: torch.Tensor | None = None):
        self._maintain_float32_expert_bias()
        if active_numerics().router_gemm:
            # Compiled vLLM runs an fp32 GEMM on the stored bf16 input and fp32 weights holding the bf16 values.
            logits = F.linear(input.float(), self.weight.float())
        else:
            logits = self.gating(input)
        return self.routing(logits, padding_mask)


class GrugGPTModel(GPTModel):
    """GPTModel with Grug's gated embedding norm on the first pipeline stage."""

    def __init__(self, config: TransformerConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        install_numerics_hooks(self)
        if self.post_process and isinstance(self.decoder.final_layernorm, GrugGatedRMSNorm):
            self.decoder.final_layernorm.role = NormRole.FINAL
        if self.pre_process:
            self.embed_norm = GrugGatedRMSNorm(
                config=config, hidden_size=config.hidden_size, eps=config.layernorm_epsilon
            )
            self.embed_norm.role = NormRole.EMBEDDING

    def forward(self, *args, **kwargs):
        clear_numerics_handoffs()
        return super().forward(*args, **kwargs)

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
