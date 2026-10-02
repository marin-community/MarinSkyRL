"""Megatron-Core modules for training the Grug MoE policy with ``trainer.strategy=megatron``.

Grug differs from a stock Megatron GPT model in a handful of places that the
layer spec cannot express through configuration alone:

* every norm (embedding, per-layer, final) is followed by a low-rank sigmoid
  gate (``GrugGatedRMSNorm``);
* queries and keys use a weightless RMS norm and a per-layer query scale;
* attention output is projected away from the value direction (XSA) and
  scaled by a per-head sigmoid gate computed from the attention input;
* the router selects the top-(k+1) experts on biased logits, drops the last
  one, and renormalizes sigmoid weights of the survivors;
* Hero adds latent expert projections, separate shared experts, causal ShortConv,
  and different key/value head counts on local and long-attention layers.

Everything else (sliding window on local layers, RoPE skipped on long layers,
half-RoPE, grouped-GEMM experts, GQA) maps onto stock
Megatron-Core settings chosen by ``GrugModelProvider`` in
``grug_megatron_bridge``.

Under ``Numerics.EXACT`` the forward computes the bytes a decode-invariant vLLM engine
(``inference_engines.vllm.decode_invariant``) computes for every token, region by region: compiled vLLM's Inductor
kernels for the norms, the q/k chain, XSA with the head gate and the shared expert's activation
(``grug_inductor_kernels``), vLLM's FA3, fused-MoE experts, dense GEMM shapes and log-probability kernel
(``grug_vllm_kernels``), the engine's row-invariant router GEMM, and vLLM's expert-parallel addition order, which
follows each sequence's serving engine (``grug_vllm_kernels.serving_engine_ranks``). Each region's gradient is that of the trainer's
chain for the same value (``grug_rounding``, computed by ``grug_reference_kernels``), or of the Megatron computation
it replaces.
"""

import math
import weakref
from dataclasses import dataclass, replace
from enum import StrEnum
from functools import partial
from typing import NamedTuple

import torch
import torch.nn.functional as F
from megatron.core import tensor_parallel
from megatron.core.tensor_parallel.mappings import all_gather_last_dim_from_tensor_parallel_region
from megatron.core.tensor_parallel.random import is_checkpointing
from megatron.core.transformer.moe.experts import TEGroupedMLP
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
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
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint, sharded_state_dict_default
from megatron.core.typed_torch import apply_module
from torch import nn

from skyrl_train.config import grug_vllm_shapes as vllm_shapes
from skyrl_train.config.numerics import ALLTOALL_DISPATCHER, MAX_PIPELINE_STAGES, Numerics, one_layer_recompute_units
from skyrl_train.models import grug_inductor_kernels as vllm_inductor
from skyrl_train.models.grug_handoffs import clear_hand_offs, hand_off, same_storage, take_hand_off
from skyrl_train.models.grug_invariant_kernels import invariant_router_logits
from skyrl_train.models.grug_reference_kernels import (
    gated_product_value,
    hybrid_input_norm_value,
    query_key_values,
    router_logits_value,
    swiglu_value,
    xsa_head_gate_value,
)
from skyrl_train.models.grug_rounding import (
    STAGE_STATISTIC_COLUMNS,
    append_stage_statistic,
    rms_norm_single_rounding,
    split_stage_statistic,
    vllm_value,
)
from skyrl_train.models.grug_vllm_kernels import (
    fa3_attention_sbhd,
    serving_row_ranks,
    vllm_ep_combine,
    vllm_expert_outputs,
    vllm_qkv_projection,
    vllm_topk_experts,
)
from skyrl_train.models.megatron_router_replay import RouterScoreType
from skyrl_train.models.grug_shortconv import causal_short_conv
from skyrl_train.models.grug_moe import (
    GRUG_ATTN_GATE_SCALE,
    GRUG_GATED_NORM_RANK,
    GRUG_ROUTER_RENORM_EPS,
    GRUG_ROUTING_RENORM_SUM,
    GRUG_XSA_EPS,
    grug_long_layer_flags,
    grug_rms_norm_no_weight,
    jax_top_k,
)

# Compiled vLLM pads a GEMM's output width to a multiple of this many columns (the 20-head gate becomes 24).
VLLM_GEMM_OUTPUT_ALIGNMENT = 8


def _first_present(preferred: torch.Tensor | None, fallback: torch.Tensor | None) -> torch.Tensor | None:
    return fallback if preferred is None else preferred


def _vllm_numerics(config: TransformerConfig) -> bool:
    return config.grug_numerics is Numerics.EXACT


def validate_vllm_numerics_model(config: TransformerConfig) -> None:
    """Refuse a model whose shapes or parts the vendored vLLM kernels do not compute: they are compiled for Snowball."""
    expected = {
        "hidden_size": vllm_shapes.HIDDEN,
        "num_attention_heads": vllm_shapes.HEADS,
        "num_query_groups": vllm_shapes.KV_HEADS,
        "kv_channels": vllm_shapes.HEAD_DIM,
        "moe_shared_expert_intermediate_size": vllm_shapes.SHARED_WIDTH,
        "grug_qk_mult": vllm_shapes.QUERY_FACTORS[0],
        "grug_qk_mult_long_scale": vllm_shapes.QUERY_FACTORS[1],
        "tensor_model_parallel_size": 1,
        "context_parallel_size": 1,
        "moe_token_dispatcher_type": ALLTOALL_DISPATCHER,
        "params_dtype": torch.bfloat16,
    }
    found = {name: getattr(config, name) for name in expected}
    if found != expected:
        raise ValueError(f"{Numerics.EXACT} numerics are compiled for {expected}, got {found}")
    if config.pipeline_model_parallel_size > MAX_PIPELINE_STAGES:
        raise ValueError(f"{Numerics.EXACT} numerics are verified on at most {MAX_PIPELINE_STAGES} pipeline stages")
    if config.grug_hero or config.grug_sconv_sites or config.moe_latent_size:
        raise ValueError(f"{Numerics.EXACT} numerics do not cover Hero, ShortConv or latent experts")


@dataclass(frozen=True)
class ResidualSum:
    """The bf16 tensors a layer adds in fp32 to form its output, ``residual + (routed + shared)``, before rounding.

    ``square_sum`` is the per-row sum of squares of the unrounded sum when compiled vLLM's fused residual-add and norm
    kernel already formed the layer output.
    """

    residual: torch.Tensor
    routed: torch.Tensor
    shared: torch.Tensor
    square_sum: torch.Tensor | None = None

    def statistic(self) -> torch.Tensor:
        """The next input norm's statistic of the unrounded sum: compiled vLLM's per-row sum of squares (``[rows]``)."""
        if self.square_sum is not None:
            return self.square_sum
        return vllm_inductor.residual_square_sum(self.residual, self.routed, self.shared)[1]


@dataclass(frozen=True)
class GatedProduct:
    """The embedding gated norm's bf16 norm output and gate logits; its output is ``normalized * sigmoid(gate)``."""

    normalized: torch.Tensor
    gate: torch.Tensor

    def statistic(self) -> torch.Tensor:
        """Layer 0's input-norm statistic of the unrounded product: compiled vLLM's per-row sum of squares from its fused
        product and norm kernel (``[rows]``)."""
        return vllm_inductor.gated_product_square_sum(self.normalized, self.gate)[1]


@dataclass(frozen=True)
class StageStatistic:
    """The input-norm statistic of a pipeline stage's first layer, computed on the previous stage from the unrounded
    residual sum that formed the layer's input (``ResidualSum.statistic``) and received with the hidden states.

    ``value`` is ``[S, B, 1]`` fp32.
    """

    value: torch.Tensor

    def statistic(self) -> torch.Tensor:
        return self.value.reshape(-1)


# Per module, what it computed in a checkpoint unit's first forward for the unit's recompute, with a weak reference to
# the tensor that identifies the unit (an input norm's input, or the decoder layer's input). Full activation recompute
# reruns the unit inside the backward on ``detach()`` copies of its inputs, after the first forward took the unit's
# hand-offs, so the recompute reads what it needs here by storage. Only a forward whose backward will run keeps
# entries, and its backward takes every one of them (``assert_recompute_drained``).
_RECOMPUTE_STATISTICS: dict[int, list[tuple[weakref.ref, torch.Tensor]]] = {}
# True while a forward runs with gradients enabled, so that a backward, and the recompute of its checkpoint units,
# follows.
_BACKWARD_FOLLOWS = False
# The input of each decoder layer whose forward is running, innermost last: the key of its checkpoint unit.
_LAYER_INPUTS: list[torch.Tensor] = []


class CheckpointPass(StrEnum):
    """Which forward of Megatron's full activation recompute is running, if any."""

    NONE = "none"
    FIRST = "first"
    RECOMPUTE = "recompute"


def checkpoint_pass() -> CheckpointPass:
    """The first forward of a checkpoint unit runs without gradients; its recompute inside the backward with them."""
    if not is_checkpointing():
        return CheckpointPass.NONE
    return CheckpointPass.RECOMPUTE if torch.is_grad_enabled() else CheckpointPass.FIRST


def _one_layer_units(config: TransformerConfig) -> bool:
    return one_layer_recompute_units(config.recompute_granularity, config.recompute_method, config.recompute_num_layers)


def _recomputing_one_layer(config: TransformerConfig) -> bool:
    """True in full activation recompute's second forward of a checkpointed unit that holds one layer."""
    return _one_layer_units(config) and checkpoint_pass() is CheckpointPass.RECOMPUTE


def _keep_for_recompute(owner: nn.Module, receiver: torch.Tensor, value: torch.Tensor) -> None:
    """Keep ``value`` for the recompute of the checkpoint unit that ``receiver`` identifies; a forward without a
    backward recomputes nothing and keeps nothing."""
    if not _BACKWARD_FOLLOWS:
        return
    _RECOMPUTE_STATISTICS.setdefault(id(owner), []).append((weakref.ref(receiver), value))


def _take_for_recompute(owner: nn.Module, receiver: torch.Tensor) -> torch.Tensor:
    entries = _RECOMPUTE_STATISTICS.get(id(owner), [])
    for index, (ref, value) in enumerate(entries):
        original = ref()
        if original is not None and same_storage(original, receiver):
            del entries[index]
            return value
    raise RuntimeError("a checkpoint unit's recompute found nothing kept by the unit's first forward")


def assert_recompute_drained() -> None:
    """Raise unless every value a checkpoint unit's first forward kept was taken by the unit's recompute."""
    kept = sum(len(entries) for entries in _RECOMPUTE_STATISTICS.values())
    if kept:
        raise RuntimeError(f"{kept} values kept by checkpoint units' first forwards were not taken by their recompute")


def _install_layer_input_hooks(layer: TransformerLayer) -> None:
    """Record each decoder layer's input while its forward runs (``_LAYER_INPUTS``)."""

    def enter(module, args, kwargs):
        _LAYER_INPUTS.append(kwargs["hidden_states"] if "hidden_states" in kwargs else args[0])

    def leave(module, args, kwargs, output):
        _LAYER_INPUTS.pop()

    layer.register_forward_pre_hook(enter, with_kwargs=True)
    layer.register_forward_hook(leave, with_kwargs=True, always_call=True)


def _unit_key(config: TransformerConfig) -> torch.Tensor | None:
    """The running layer's input, when full recompute makes that layer a checkpoint unit of its own (else ``None``)."""
    return _LAYER_INPUTS[-1] if _LAYER_INPUTS and _one_layer_units(config) else None


class _ResidualSumGradient(torch.autograd.Function):
    """``value``, the bf16 ``residual + (routed + shared)`` summed in fp32, differentiated as that expression.

    Autograd of ``(residual.float() + (routed.float() + shared.float())).to(bf16)`` hands each of the three inputs
    ``grad.float().to(bf16)``: the fp32 sums pass the gradient through, and the casts convert it there and back.
    """

    @staticmethod
    def forward(ctx, value, residual, routed, shared):
        return value

    @staticmethod
    def backward(ctx, grad):
        converted = grad.float().to(grad.dtype)
        return None, converted, converted, converted


def _vllm_residual_sum(parts: ResidualSum, config: TransformerConfig) -> tuple[torch.Tensor, ResidualSum | None]:
    """The layer output ``residual + (routed + shared)`` from compiled vLLM's fused residual-add and norm kernel, which
    also forms the next input norm's sum of squares, and the hand-off for that norm (``None`` when no one reads it).

    The kernel adds in fp32 and rounds once, as ``(residual.float() + (routed.float() + shared.float())).to(bf16)``
    does. Full recompute's second forward of a one-layer unit only rebuilds the layer's graph for its backward, and the
    layer output is the unit's output, whose value the backward never reads, so that forward forms no value.
    """
    if _recomputing_one_layer(config):
        value, handed = torch.empty_like(parts.residual), None
    else:
        value, square_sum = vllm_inductor.residual_square_sum(parts.residual, parts.routed, parts.shared)
        value = value.view_as(parts.residual)
        handed = ResidualSum(parts.residual, parts.routed, parts.shared, square_sum)
    if torch.is_grad_enabled():
        value = _ResidualSumGradient.apply(value, parts.residual, parts.routed, parts.shared)
    return value, handed


def _install_residual_hooks(layer: TransformerLayer) -> None:
    """Form the layer output ``residual + (routed + shared)`` with compiled vLLM's fused residual-add kernel."""
    stored: dict[str, torch.Tensor] = {}

    def keep_residual(module, args):
        stored["residual"] = args[0]

    postprocess = layer.mlp.postprocess
    combine_postprocess = layer.mlp.token_dispatcher.combine_postprocess

    def keep_routed(output):
        routed = combine_postprocess(output)
        stored["routed"] = routed
        return routed

    def keep_shared(output, shared_expert_output):
        if shared_expert_output is None:
            raise RuntimeError("Grug's vLLM numerics expect a shared expert")
        stored["shared"] = shared_expert_output
        return postprocess(output, shared_expert_output)

    def mlp_residual(module, args, output):
        parts = ResidualSum(stored.pop("residual"), stored.pop("routed"), stored.pop("shared"))
        hidden, handed = _vllm_residual_sum(parts, layer.config)
        if handed is not None:
            hand_off(hidden, handed)
        return (hidden, *output[1:])

    layer.pre_mlp_layernorm.register_forward_pre_hook(keep_residual)
    layer.mlp.token_dispatcher.combine_postprocess = keep_routed
    layer.mlp.postprocess = keep_shared
    layer.register_forward_hook(mlp_residual)


def _install_vllm_experts_hooks(experts: TEGroupedMLP) -> None:
    """Take the routed experts' values from vLLM's fused-MoE kernels.

    The kernels compute each dispatched row (one token-expert slot) as vLLM does, with the route weight inside the down
    projection's fp32 accumulator. With gradients enabled the trainer's grouped-GEMM experts also run and give the
    kernels' bytes their gradient (``vllm_value``); a forward without gradients runs the kernels alone.

    Full recompute's second forward of a one-layer unit runs the grouped-GEMM experts alone: that forward only rebuilds
    the layer's graph for its backward, and the layer uses the experts' output only in sums (the combine, the shared
    expert and the residuals), whose gradients do not depend on the summands' values. The gradients equal those of a
    forward that also runs the kernels; the layer's output, the first forward's, keeps the kernels' bytes.
    """
    if any(linear.tp_size != 1 or linear.use_bias for linear in (experts.linear_fc1, experts.linear_fc2)):
        raise NotImplementedError("Grug's vLLM numerics need unsharded, bias-free expert projections")
    grouped_forward = experts.forward

    def forward(permuted_local_hidden_states, tokens_per_expert, permuted_probs):
        if _recomputing_one_layer(experts.config):
            return grouped_forward(permuted_local_hidden_states, tokens_per_expert, permuted_probs)
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
        return vllm_value(value, lambda: reference), bias

    experts.forward = forward


def _install_vllm_gemm(linear: nn.Module, compute) -> None:
    """Run ``linear`` (a Transformer Engine linear) as ``compute(x)``.

    ``compute`` issues ``torch.mm`` in compiled vLLM's shapes on the module's own weight, so autograd gives the weight
    its gradient; Megatron's gradient hooks add it to the parameter's main gradient as for any parameter Transformer
    Engine did not already accumulate.
    """
    if linear.tp_size != 1 or linear.use_bias:
        raise NotImplementedError("Grug's vLLM numerics need unsharded, bias-free projections")

    def forward(x, *args, **kwargs):
        if args or kwargs:
            raise NotImplementedError("Grug's vLLM numerics run each projection on one input")
        return compute(x), None

    linear.forward = forward


def _install_vllm_gemm_attention_hooks(attention: "GrugSelfAttention") -> None:
    """Compiled vLLM's attention projections: q, k and v as three GEMMs, and the head gate padded to a multiple of 8
    outputs, as the decode-invariant engine compiles it."""
    qkv = attention.linear_qkv
    groups = attention.num_query_groups_per_partition
    head_dim = attention.hidden_size_per_attention_head
    query_width = attention.num_attention_heads_per_partition // groups * head_dim
    gate = attention.attn_gate

    def head_gate(x: torch.Tensor) -> torch.Tensor:
        heads = gate.weight.shape[0]
        columns = heads + -heads % VLLM_GEMM_OUTPUT_ALIGNMENT
        weight = torch.cat((gate.weight, gate.weight.new_zeros(columns - heads, gate.weight.shape[1])))
        return F.linear(x, weight)[..., :heads].contiguous()

    _install_vllm_gemm(qkv, lambda x: vllm_qkv_projection(x, qkv.weight, groups, query_width, head_dim))
    _install_vllm_gemm(gate, head_gate)
    _install_vllm_gemm(attention.linear_proj, lambda x: F.linear(x, attention.linear_proj.weight))


def _install_vllm_gemm_shared_hooks(shared: SharedExpertMLP) -> None:
    """Compiled vLLM's shared expert: gate and up projections as two GEMMs, then the down projection."""
    fc1, fc2 = shared.linear_fc1, shared.linear_fc2

    def gate_and_up(x: torch.Tensor) -> torch.Tensor:
        gate_weight, up_weight = fc1.weight.chunk(2, dim=0)
        return torch.cat((F.linear(x, gate_weight), F.linear(x, up_weight)), dim=-1)

    _install_vllm_gemm(fc1, gate_and_up)
    _install_vllm_gemm(fc2, lambda x: F.linear(x, fc2.weight))


def _install_fa3_attention_hooks(attention: "GrugSelfAttention") -> None:
    """Take the attention value from vLLM's FA3 forward, computed as the decode-invariant engine computes each row.

    The gradient is the trainer's cuDNN attention backward at the same query, key and value: with gradients enabled the
    cuDNN forward also runs and gives FA3's bytes its gradient (``vllm_value``). A forward without gradients runs FA3
    alone.

    Under full recompute with one-layer units, a unit's first forward keeps FA3's output for the unit's recompute,
    which reproduces the first forward's query, key and value bit for bit and so would compute the same bytes again.
    """
    core = attention.core_attention
    cudnn_forward = core.forward

    def forward(
        query, key, value, attention_mask, attn_mask_type=None, attention_bias=None, packed_seq_params=None, **kwargs
    ):
        if packed_seq_params is not None or attention_bias is not None or kwargs:
            raise NotImplementedError("Grug's vLLM numerics support unpacked causal sequences only")
        phase, unit = checkpoint_pass(), _unit_key(attention.config)
        if phase is CheckpointPass.RECOMPUTE and unit is not None:
            output = _take_for_recompute(core, unit)
        else:
            output = fa3_attention_sbhd(query, key, value, window=attention.fa3_window, scale=attention.fa3_scale)
            if phase is CheckpointPass.FIRST and unit is not None:
                _keep_for_recompute(core, unit, output)
        if not torch.is_grad_enabled():
            return output
        reference = cudnn_forward(
            query,
            key,
            value,
            attention_mask,
            attn_mask_type=attn_mask_type,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
        )
        return vllm_value(output.detach(), lambda: reference)

    core.forward = forward


def _install_ep_combine_hooks(layer: TransformerLayer) -> None:
    """Add the routed expert outputs in vLLM's expert-parallel order (``vllm_ep_combine``).

    The gradient is the trainer's own unpermute (each slot's gradient is the token's output gradient in both), through
    ``vllm_value``. A forward without gradients skips the trainer's unpermute.

    Full recompute's second forward of a one-layer unit takes the trainer's unpermute alone: that forward only rebuilds
    the layer's graph for its backward, and the layer uses the routed output only in sums and casts (the shared expert
    and the residuals), whose gradients do not depend on the summands' values. The gradients equal those of a forward
    that also runs the combine; the layer's output, the first forward's, keeps the combine's bytes.
    """
    dispatcher = layer.mlp.token_dispatcher
    if dispatcher.shared_experts is not None:
        raise NotImplementedError("Grug's vLLM numerics expect the shared expert outside the dispatcher")
    router = layer.mlp.router
    unpermute = dispatcher.combine_postprocess

    def combine_postprocess(permuted):
        if _recomputing_one_layer(layer.config):
            return unpermute(permuted)
        route = router.take_ep_route()
        with torch.no_grad():
            combined = vllm_ep_combine(
                permuted, dispatcher.routing_map, route.selected, route.serving_ranks, route.expert_parallel_size
            ).view(dispatcher.hidden_shape)
        if not torch.is_grad_enabled():
            return combined
        output = unpermute(permuted)
        return vllm_value(combined, lambda: output)

    dispatcher.combine_postprocess = combine_postprocess


def _install_shared_swiglu_hooks(shared: SharedExpertMLP) -> None:
    """Take the shared expert's activation ``silu(gate) * up`` from compiled vLLM's activation kernel; the gradient is
    that of the activation computed in fp32 and rounded once."""
    stored: dict[str, torch.Tensor] = {}

    def keep_fc1_output(module, args, output):
        stored["fc1"] = output[0] if isinstance(output, tuple) else output

    def replace_fc2_input(module, args):
        fc1_output = stored.pop("fc1", None)
        if fc1_output is None:
            raise RuntimeError("Grug's vLLM numerics need the shared expert's fc1 output")
        gate, up = torch.chunk(fc1_output, 2, dim=-1)
        value = vllm_inductor.shared_activation(gate, up).view(gate.shape)
        # Differentiated as swiglu_single_rounding(fc1_output).
        return (swiglu_value(value, fc1_output), *args[1:])

    shared.linear_fc1.register_forward_hook(keep_fc1_output)
    shared.linear_fc2.register_forward_pre_hook(replace_fc2_input)


class NormRole(StrEnum):
    """Where a gated norm sits, which decides the residual it reads under the vLLM numerics."""

    INPUT = "input"
    POST_ATTENTION = "post_attention"
    FINAL = "final"
    EMBEDDING = "embedding"


def install_vllm_numerics(model: "GrugGPTModel") -> None:
    """Install the vLLM numerics on every Grug decoder layer, gated norm and attention module of ``model``."""
    validate_vllm_numerics_model(model.config)
    for module in model.modules():
        if isinstance(module, SharedExpertMLP):
            _install_shared_swiglu_hooks(module)
            _install_vllm_gemm_shared_hooks(module)
        if isinstance(module, TEGroupedMLP):
            _install_vllm_experts_hooks(module)
        if isinstance(module, GrugSelfAttention):
            _install_fa3_attention_hooks(module)
            _install_vllm_gemm_attention_hooks(module)
        if isinstance(module, TransformerLayer) and isinstance(module.pre_mlp_layernorm, GrugGatedRMSNorm):
            # The residual hooks wrap the EP combine, so the fp32 residual reads the combine's value.
            _install_ep_combine_hooks(module)
            _install_residual_hooks(module)
            _install_layer_input_hooks(module)
            module.input_layernorm.role = NormRole.INPUT
    if model.post_process:
        model.decoder.final_layernorm.role = NormRole.FINAL
    if model.pre_process:
        model.embed_norm.role = NormRole.EMBEDDING


class GrugGatedRMSNorm(nn.Module):
    """RMSNorm followed by Grug's low-rank sigmoid gate: ``norm(x) * sigmoid(up(silu(down(norm(x)))))``."""

    def __init__(self, config: TransformerConfig, hidden_size: int, eps: float):
        super().__init__()
        self.vllm_numerics = _vllm_numerics(config)
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
        if self.vllm_numerics:
            return self._vllm_forward(hidden_states)
        normalized = self.norm(hidden_states)
        gate = torch.sigmoid(self.up_proj(F.silu(self.down_proj(normalized))))
        return normalized * gate

    def _vllm_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """The norm and gated product from compiled vLLM's kernels, each norm reading the sum that formed its input."""
        weight = self.norm.weight
        if self.role is NormRole.INPUT:
            normalized = self._input_norm(hidden_states)
        elif self.role is NormRole.FINAL:
            normalized = self._final_norm(hidden_states)
        elif self.role is NormRole.EMBEDDING:
            value = vllm_inductor.embedding_norm(hidden_states, weight).view_as(hidden_states)
            normalized = vllm_value(value, lambda: self.norm(hidden_states))
        else:
            value = vllm_inductor.rms_norm(hidden_states, weight).view_as(hidden_states)
            normalized = vllm_value(value, lambda: self.norm(hidden_states))
        gate = self.up_proj(F.silu(self.down_proj(normalized)))
        value = vllm_inductor.gated_product(normalized, gate).view_as(normalized)
        # Differentiated as gated_norm_product_fp32(normalized, gate).to(normalized.dtype).
        output = gated_product_value(value, normalized, gate)
        if self.role is NormRole.EMBEDDING:
            # Layer 0's input norm takes its statistic from the unrounded embedding gated-norm product.
            hand_off(output, GatedProduct(normalized, gate))
        return output

    def _input_norm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """A layer's input norm: compiled vLLM takes the statistic from the unrounded sum that formed its input."""
        weight = self.norm.weight
        statistic = self._input_statistic(hidden_states)
        value = vllm_inductor.rms_norm_from_square_sum(hidden_states, statistic, weight).view_as(hidden_states)
        variance = (statistic / hidden_states.shape[-1]).view(*hidden_states.shape[:-1], 1)
        # Differentiated as rms_norm_hybrid(hidden_states, variance_with_gradient(variance, hidden_states), weight, eps).
        return hybrid_input_norm_value(value, hidden_states, variance, weight, self.eps)

    def _input_statistic(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """The input norm's statistic of the unrounded sum: compiled vLLM's per-row sum of squares.

        A checkpoint unit's first forward keeps what it computed for the recompute, which cannot reach the hand-off.
        """
        phase = checkpoint_pass()
        parts = take_hand_off(hidden_states)
        if parts is None:
            if phase is CheckpointPass.RECOMPUTE:
                return _take_for_recompute(self, hidden_states)
            raise RuntimeError(
                "a layer's input norm found no hand-off of the sum that formed its input: the embedding gated norm, "
                "the previous layer or the previous pipeline stage registered none"
            )
        with torch.no_grad():
            statistic = parts.statistic()
        if phase is CheckpointPass.FIRST:
            _keep_for_recompute(self, hidden_states, statistic)
        return statistic

    def _final_norm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """The final norm: compiled vLLM normalizes the last layer's unrounded sum."""
        parts = take_hand_off(hidden_states)
        if not isinstance(parts, ResidualSum):
            raise RuntimeError("the final norm found no residual sum handed on by the last decoder layer")
        # The hand-off may come from a no-grad checkpointed forward; keep its values and take the gradient through the
        # bf16 residual so the loss still reaches the layers before this norm. ``x - x.detach()`` is exactly zero, so
        # the values are unchanged bit for bit.
        residual = hidden_states.float()

        def reference() -> torch.Tensor:
            unrounded = (
                parts.residual.detach().float() + (parts.routed.detach().float() + parts.shared.detach().float())
            ) + (residual - residual.detach())
            return rms_norm_single_rounding(unrounded, self.norm.weight, self.eps)

        value = vllm_inductor.final_norm(parts.residual, parts.routed, parts.shared, self.norm.weight)
        return vllm_value(value.view_as(hidden_states), reference)


class GrugQKNorm(nn.Module):
    """Weightless RMS norm applied to each query and key head."""

    def __init__(self, config: TransformerConfig, hidden_size: int, eps: float):
        super().__init__()
        self.vllm_numerics = _vllm_numerics(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.vllm_numerics:
            # The attention forward runs compiled vLLM's q/k norm, RoPE and query scale on the raw projections.
            return hidden_states
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
        is_long = grug_long_layer_flags(config.num_layers, config.grug_global_every)[layer_number - 1]
        self.query_scale = config.grug_qk_mult * (config.grug_qk_mult_long_scale if is_long else 1.0)
        self.skip_rope = bool(config.no_rope_freq[layer_number - 1])
        self.vllm_numerics = _vllm_numerics(config)
        # Compiled vLLM applies the two query factors one after the other.
        self.qk_mult = config.grug_qk_mult
        self.qk_mult_scale = config.grug_qk_mult_long_scale if is_long else 1.0
        # vLLM's FA3 call for this layer: a sliding window of window_size[0] + 1 keys on local layers.
        self.fa3_window = None if is_long else config.window_size[0] + 1
        self.fa3_scale = (
            config.softmax_scale if config.softmax_scale is not None else 1.0 / math.sqrt(config.kv_channels)
        )
        self.logical_kv_heads = config.grug_global_kv_heads if is_long else config.grug_local_kv_heads
        self.sconv_k = None
        self.sconv_attn = None
        if "k" in config.grug_sconv_sites:
            self.sconv_k = GrugShortConv(
                config,
                self.num_query_groups_per_partition * config.kv_channels,
                self.pg_collection,
                channel_parallel=True,
            )
            # K convolution precedes the weightless head norm.
            self.k_layernorm = nn.Identity()
        if "attn" in config.grug_sconv_sites:
            self.sconv_attn = GrugShortConv(config, config.hidden_size, self.pg_collection)
        if self.logical_kv_heads != config.num_query_groups:
            attention_config = replace(config, num_query_groups=self.logical_kv_heads)
            self.core_attention = submodules.core_attention(
                config=attention_config,
                layer_number=layer_number,
                attn_mask_type=self.attn_mask_type,
                attention_type="self",
                cp_comm_type=kwargs.get("cp_comm_type"),
                softmax_scale=config.softmax_scale,
                pg_collection=self.pg_collection,
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
        if self.sconv_k is not None:
            key = self.sconv_k(key.flatten(-2), packed_seq_params).reshape(key.shape)
            key = grug_rms_norm_no_weight(key)
        if self.logical_kv_heads != self.config.num_query_groups:
            key, value = (self._logical_kv(tensor) for tensor in (key, value))

        is_thd = packed_seq_params is not None and packed_seq_params.qkv_format == "thd"
        if self.vllm_numerics:
            if is_thd:
                raise NotImplementedError("Grug's vLLM numerics support unpacked sequences only")
            return self._vllm_forward(hidden_states, query, key, value, rotary_pos_emb)
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
        if self.sconv_attn is not None:
            output = self.sconv_attn(output, packed_seq_params)
        return output, bias

    def _vllm_forward(self, hidden_states, query, key, value, rotary_pos_emb):
        """Compiled vLLM's q/k chain, FA3 attention, XSA with the head gate and output projection."""
        query, key = self._vllm_query_key(query, key, rotary_pos_emb)
        if self.checkpoint_core_attention and self.training:
            attention = self._checkpointed_attention_forward(
                query, key, value, None, attn_mask_type=self.attn_mask_type
            )
        else:
            attention = self._run_core_attention(query, key, value, None, attn_mask_type=self.attn_mask_type)
        gate, _ = self.attn_gate(hidden_states)
        xsa = vllm_inductor.xsa_head_gate(attention, value, gate).view_as(attention)
        # Differentiated as xsa_and_gate_single_rounding(attention, value, gate, head_dim).
        gated = xsa_head_gate_value(xsa, attention, value, gate, self.hidden_size_per_attention_head)
        return apply_module(self.linear_proj)(gated)

    def _vllm_query_key(
        self, query: torch.Tensor, key: torch.Tensor, rotary_pos_emb
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compiled vLLM's q/k norm, RoPE and query scale on the raw ``[S, B, heads, dim]`` projections.

        The values come from vLLM's Inductor kernels; with gradients enabled they are differentiated as
        ``rounded_query_key(qk_norm_fp32(query), qk_norm_fp32(key), ...)``, which computes the same values up to rounding.
        """
        sequence, batch = query.shape[:2]
        rotary = rotary_pos_emb is not None and not self.skip_rope
        if rotary:
            positions = torch.arange(sequence, device=query.device).repeat_interleave(batch)
            vllm_query, vllm_key = vllm_inductor.query_key_rope(query, key, positions, self.config.rotary_base)
        else:
            vllm_query, vllm_key = vllm_inductor.query_key_full(query, key)
        vllm_query, vllm_key = vllm_query.view_as(query), vllm_key.view_as(key)
        freqs = None
        if rotary:
            freqs = rotary_pos_emb if isinstance(rotary_pos_emb, tuple) else (rotary_pos_emb,) * 2
        return query_key_values(vllm_query, vllm_key, query, key, freqs, self.qk_mult, self.qk_mult_scale)

    def _logical_kv(self, tensor):
        # The checkpoint stores max(local, global) heads. Retain unused rows for
        # lossless export, but attend only to the leading logical KV heads.
        # A differentiable gather is needed: TP's stored and logical head owners differ.
        gathered = all_gather_last_dim_from_tensor_parallel_region(
            tensor.flatten(-2), group=self.pg_collection.tp
        ).reshape(*tensor.shape[:2], self.config.num_query_groups, self.config.kv_channels)
        count = self.logical_kv_heads // self.pg_collection.tp.size()
        start = self.pg_collection.tp.rank() * count
        return gathered[..., start : start + count, :].contiguous()

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


class EpRoute(NamedTuple):
    """A router's selection for vLLM's expert-parallel combine: each row's experts in slot order, each row's serving
    rank, and vLLM's expert-parallel size."""

    selected: torch.Tensor
    serving_ranks: torch.Tensor
    expert_parallel_size: int


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

    replay_score_type = RouterScoreType.LOGITS

    def __init__(self, config: TransformerConfig, pg_collection=None, is_mtp_layer: bool = False):
        super().__init__(config=config, pg_collection=pg_collection, is_mtp_layer=is_mtp_layer)
        assert self.enable_expert_bias, "GrugTopKRouter requires moe_router_enable_expert_bias=True"
        self.vllm_numerics = _vllm_numerics(config)
        self._ep_route: EpRoute | None = None

    def routing(self, logits: torch.Tensor, padding_mask: torch.Tensor | None = None):
        logits = logits.view(-1, self.config.num_moe_experts).float()
        biased_logits = logits + self.expert_bias
        # vLLM's selection breaks exact ties in its own order, which native routing must reproduce.
        select = vllm_topk_experts if self.vllm_numerics else _grug_topk_indices
        if self.router_replay is None:
            selected = select(biased_logits, self.topk)
        else:
            # Grug overrides MCore's routing method, so its replay hook must be
            # called here. Selection uses biased logits; combine weights below
            # still use the original, unbiased logits.
            def native_topk(scores, topk, num_groups=None, group_topk=None):
                if num_groups is not None or group_topk is not None:
                    raise ValueError("Grug router replay does not support grouped top-k")
                indices = select(scores, topk)
                return scores.gather(1, indices), indices

            _, selected = self.router_replay.get_replay_topk(biased_logits, self.topk, None, None, native_topk)
        if self.vllm_numerics and not _recomputing_one_layer(self.config):
            # The combine adds the slots in vLLM's order: the selection order, per vLLM EP rank.
            self._ep_route = EpRoute(selected, *serving_row_ranks(selected.shape[0]))
        combine = torch.sigmoid(torch.gather(logits, dim=-1, index=selected))
        combine = combine * (GRUG_ROUTING_RENORM_SUM / (combine.sum(dim=-1, keepdim=True) + GRUG_ROUTER_RENORM_EPS))
        probs = torch.zeros_like(logits).scatter(1, selected, combine)
        routing_map = torch.zeros_like(logits, dtype=torch.bool).scatter(1, selected, True)
        return probs, routing_map

    def take_ep_route(self) -> EpRoute:
        """The slot order and serving ranks of this router's last routing, for vLLM's expert-parallel combine."""
        route, self._ep_route = self._ep_route, None
        if route is None:
            raise RuntimeError("the expert-parallel combine ran without a routing from this layer's router")
        return route

    def forward(self, input: torch.Tensor, padding_mask: torch.Tensor | None = None):
        self._maintain_float32_expert_bias()
        logits = self._invariant_logits(input) if self.vllm_numerics else self.gating(input)
        return self.routing(logits, padding_mask)

    def _invariant_logits(self, input: torch.Tensor) -> torch.Tensor:
        """The router logits a decode-invariant vLLM engine computes: the row-invariant Triton GEMM on the bf16 input
        and the bf16 weight (exact products, fp32 sums), whatever rows share the call; the gradient is the fp32 GEMM's."""
        if self.weight.dtype != torch.bfloat16 or input.dtype != torch.bfloat16:
            raise ValueError("the invariant router GEMM multiplies the bf16 router input by the bf16 router weight")
        with torch.no_grad():
            value = invariant_router_logits(input.reshape(-1, input.shape[-1]), self.weight)
        value = value.view(*input.shape[:-1], value.shape[-1])
        # Differentiated as F.linear(input.float(), self.weight.float()).
        return router_logits_value(value, input, self.weight)


def _stage_output_statistic(output: torch.Tensor) -> torch.Tensor:
    """The next stage's first input-norm statistic, from the residual sum that formed this stage's last output."""
    parts = take_hand_off(output)
    if not isinstance(parts, ResidualSum):
        raise RuntimeError("a pipeline stage's output carries no residual sum to hand its input-norm statistic on")
    with torch.no_grad():
        return parts.statistic().reshape(*output.shape[:-1], 1)


class GrugGPTModel(GPTModel):
    """GPTModel with Grug's gated embedding norm on the first pipeline stage.

    Under the vLLM numerics a stage that is not the last appends each token's input-norm statistic (one fp32, as two
    bf16 words) to the hidden states it hands on, and the next stage splits it off for its first layer's input norm
    (``StageStatistic``). Megatron's pipeline exchanges each micro-batch's tensor shape, so the wider tensor rides the
    existing point-to-point transfer in the forward and its gradient in the backward.
    """

    def __init__(self, config: TransformerConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        if self.pre_process:
            self.embed_norm = GrugGatedRMSNorm(
                config=config, hidden_size=config.hidden_size, eps=config.layernorm_epsilon
            )
        self.vllm_numerics = _vllm_numerics(config)
        self._received: tuple[torch.Tensor, torch.Tensor] | None = None
        if self.vllm_numerics:
            install_vllm_numerics(self)

    def set_input_tensor(self, input_tensor) -> None:
        if not self.vllm_numerics:
            super().set_input_tensor(input_tensor)
            return
        tensors = input_tensor if isinstance(input_tensor, list) else [input_tensor]
        if len(tensors) != 1:
            raise ValueError("a Grug pipeline stage receives one tensor")
        received = tensors[0]
        self._received = None
        if received is not None:
            if received.shape[-1] != self.config.hidden_size + STAGE_STATISTIC_COLUMNS:
                raise ValueError("a Grug pipeline stage under the vLLM numerics receives its input-norm statistic")
            self._received = split_stage_statistic(received, self.config.hidden_size)
            received = self._received[0]
        super().set_input_tensor(received)

    def forward(self, *args, **kwargs):
        if not self.vllm_numerics:
            return super().forward(*args, **kwargs)
        global _BACKWARD_FOLLOWS
        _BACKWARD_FOLLOWS = torch.is_grad_enabled()
        # Each forward starts without hand-offs: a stage's last residual has no reader, and the entries are keyed by
        # tensors that a later forward may reuse. The statistics kept for recompute stay: under pipeline parallelism a
        # micro-batch's backward, and its recompute, runs after later micro-batches' forwards.
        clear_hand_offs()
        if self._received is not None:
            hidden, statistic = self._received
            self._received = None
            hand_off(hidden, StageStatistic(statistic))
        output = super().forward(*args, **kwargs)
        if self.post_process:
            return output
        return append_stage_statistic(output, _stage_output_statistic(output))

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


class GrugShortConv(nn.Module):
    """ShortConv on CP-local tokens with optional TP sequence or channel sharding."""

    def __init__(self, config, channels, pg_collection, channel_parallel=False):
        super().__init__()
        self.pg_collection = pg_collection
        self.channel_parallel = channel_parallel
        self.sequence_parallel = config.sequence_parallel and not channel_parallel
        self.weight = nn.Parameter(
            torch.zeros(
                config.grug_sconv_kernel, channels, device=torch.cuda.current_device(), dtype=config.params_dtype
            )
        )
        with torch.no_grad():
            self.weight[0].fill_(1)
        # Output convolutions see the full TP sequence on every rank, so their
        # weight gradients are already complete replicas, not SP partials.
        self.weight.sequence_parallel = False
        if channel_parallel:
            tensor_parallel.set_tensor_model_parallel_attributes(self.weight, True, 1, 1)

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        return make_sharded_tensors_for_checkpoint(
            self.state_dict(keep_vars=True),
            prefix,
            {"weight": 1} if self.channel_parallel else {},
            sharded_offsets=sharded_offsets,
            tp_group=self.pg_collection.tp,
            dp_cp_group=metadata["dp_cp_group"],
        )

    def forward(self, hidden_states, packed_seq_params=None):
        if self.sequence_parallel:
            hidden_states = tensor_parallel.gather_from_sequence_parallel_region(
                hidden_states, tensor_parallel_output_grad=False, group=self.pg_collection.tp
            )
        lengths = None
        if packed_seq_params is not None and packed_seq_params.qkv_format == "thd":
            cumulative = _first_present(packed_seq_params.cu_seqlens_q_padded, packed_seq_params.cu_seqlens_q)
            lengths = tuple((cumulative[1:] - cumulative[:-1]).tolist())
        result = causal_short_conv(hidden_states, self.weight, lengths, self.pg_collection.cp)
        if self.sequence_parallel:
            result = tensor_parallel.scatter_to_sequence_parallel_region(result, group=self.pg_collection.tp)
        return result


class GrugSharedExperts(nn.Module):
    """Keep Hero's shared experts separate, including their ordered BF16 additions."""

    def __init__(self, config, submodules, pg_collection, gate=False, name=None):
        super().__init__()
        self.experts = nn.ModuleList(
            [
                SharedExpertMLP(config=config, submodules=submodules, pg_collection=pg_collection, gate=gate)
                for _ in range(config.grug_num_shared_experts)
            ]
        )

    def forward(self, hidden_states):
        return tuple(expert(hidden_states) for expert in self.experts)

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        state = {}
        for index, expert in enumerate(self.experts):
            state.update(sharded_state_dict_default(expert, f"{prefix}experts.{index}.", sharded_offsets, metadata))
        return state

    def backward_dw(self):
        for expert in self.experts:
            expert.backward_dw()


class GrugMoELayer(MoELayer):
    """Native Megatron latent dispatch, with Hero's pre-dispatch latent RMSNorm."""

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        # MCore 0.18 marks expert parameters as dense when EP=1. With a
        # smaller expert TP group, those replicas instead belong to expert DP
        # (which includes the extra attention-TP ranks).
        if (
            config.expert_model_parallel_size == 1
            and config.tensor_model_parallel_size != config.expert_tensor_parallel_size
        ):
            for param in self.experts.parameters():
                param.allreduce = False
        if config.moe_latent_size:
            self.latent_norm = TENorm(config=config, hidden_size=config.moe_latent_size, eps=config.layernorm_epsilon)

    def preprocess(self, hidden_states, probs, routing_map):
        if not self.config.moe_latent_size:
            return super().preprocess(hidden_states, probs, routing_map)
        hidden_states, _ = self.fc1_latent_proj(hidden_states)
        hidden_states = self.latent_norm(hidden_states)
        return self.token_dispatcher.dispatch_preprocess(hidden_states, routing_map, probs)

    def postprocess(self, output, shared_expert_output):
        output = super().postprocess(output, None)
        if shared_expert_output is not None:
            for shared in shared_expert_output:
                output = output + shared
        return output


class GrugTransformerLayer(TransformerLayer):
    """Pass document metadata explicitly to the MLP branch's ShortConv."""

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.sconv_mlp = GrugShortConv(config, config.hidden_size, self.pg_collection)

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        hidden_states, context = self._forward_attention(hidden_states, attention_mask, **kwargs)
        normalized = self._forward_pre_mlp_layernorm(hidden_states)
        padding_mask = kwargs.get("padding_mask")
        packed_seq_params = kwargs.get("packed_seq_params")

        def mlp_forward(x):
            return self.mlp(x, padding_mask=padding_mask)

        if self.recompute_mlp:
            output, bias = tensor_parallel.checkpoint(mlp_forward, False, normalized)
        else:
            output, bias = mlp_forward(normalized)
        output = self.sconv_mlp(output, packed_seq_params)
        return self._forward_post_mlp((output, bias), hidden_states), context


def grug_layer_spec(config: TransformerConfig) -> ModuleSpec:
    """Build the Transformer Engine layer spec for one Grug decoder layer."""

    backend = TESpecProvider()
    moe = get_moe_module_spec_for_backend(backend, num_experts=config.num_moe_experts, moe_grouped_gemm=True)
    moe.keywords["submodules"].router = GrugTopKRouter
    if config.grug_hero:
        shared = moe.keywords["submodules"].shared_experts
        moe.keywords["submodules"].shared_experts = partial(GrugSharedExperts, **shared.keywords)
        moe = partial(GrugMoELayer, **moe.keywords)
    return ModuleSpec(
        module=GrugTransformerLayer if "mlp" in config.grug_sconv_sites else TransformerLayer,
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
