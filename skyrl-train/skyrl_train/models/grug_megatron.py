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
from dataclasses import dataclass
from enum import StrEnum

import torch
import torch.nn.functional as F
from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TENorm
from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec_for_backend
from megatron.core.tensor_parallel.random import is_checkpointing
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
from skyrl_train.models import grug_inductor_kernels as vllm_inductor
from skyrl_train.models.grug_handoffs import clear_hand_offs, hand_off, same_storage, take_hand_off
from skyrl_train.models.grug_rounding import (
    STAGE_STATISTIC_COLUMNS,
    append_stage_statistic,
    gated_norm_product_fp32,
    rms_norm_hybrid,
    rms_norm_single_rounding,
    weighted_down_projection_single_rounding,
    rotate_neox_fp32,
    split_stage_statistic,
    swiglu_single_rounding,
    vllm_value,
    xsa_and_gate_single_rounding,
)
from skyrl_train.models.grug_vllm_kernels import (
    VLLM_MAX_BATCHED_TOKENS,
    fa3_attention_sbhd,
    fixed_rows_linear,
    planned_vllm_steps,
    step_lm_head_logits,
    step_rows_linear,
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


@dataclass(frozen=True)
class ResidualSum:
    """The bf16 tensors a layer adds in fp32 to form its output, ``residual + (routed + shared)``, before rounding."""

    residual: torch.Tensor
    routed: torch.Tensor
    shared: torch.Tensor

    def unrounded(self) -> torch.Tensor:
        return self.residual.detach().float() + (self.routed.detach().float() + self.shared.detach().float())

    def statistic(self, numerics) -> torch.Tensor:
        """The next input norm's statistic of the unrounded sum: compiled vLLM's per-row sum of squares from its fused
        residual-add and norm kernel (``vllm_norms``, ``[rows]``), else the sum's variance (``[..., 1]``)."""
        if numerics.vllm_norms:
            return vllm_inductor.residual_square_sum(self.residual, self.routed, self.shared)[1]
        return self.unrounded().pow(2).mean(dim=-1, keepdim=True)


@dataclass(frozen=True)
class GatedProduct:
    """The embedding gated norm's bf16 norm output and gate logits; its output is ``normalized * sigmoid(gate)``."""

    normalized: torch.Tensor
    gate: torch.Tensor

    def unrounded(self) -> torch.Tensor:
        return gated_norm_product_fp32(self.normalized.detach(), self.gate.detach())

    def statistic(self, numerics) -> torch.Tensor:
        """Layer 0's input-norm statistic of the unrounded product: compiled vLLM's per-row sum of squares from its fused
        product and norm kernel (``vllm_norms``, ``[rows]``), else the product's variance (``[..., 1]``)."""
        if numerics.vllm_norms:
            return vllm_inductor.gated_product_square_sum(self.normalized, self.gate)[1]
        return self.unrounded().pow(2).mean(dim=-1, keepdim=True)


@dataclass(frozen=True)
class StageStatistic:
    """The input-norm statistic of a pipeline stage's first layer, computed on the previous stage from the unrounded
    residual sum that formed the layer's input (``ResidualSum.statistic``) and received with the hidden states.

    ``value`` is ``[S, B, 1]`` fp32; both stages run the same numerics, so it is the statistic this stage's norm needs.
    """

    value: torch.Tensor

    def statistic(self, numerics) -> torch.Tensor:
        return self.value.reshape(-1) if numerics.vllm_norms else self.value


HandOff = ResidualSum | GatedProduct | StageStatistic

# Per input norm, the statistic it took from its hand-off in a checkpoint unit's first forward, with a weak reference
# to the norm's input. Full activation recompute reruns the unit inside the backward on ``detach()`` copies of its
# inputs, after the first forward took the unit's hand-offs, so the recompute reads the statistic here by storage.
_RECOMPUTE_STATISTICS: dict[int, list[tuple[weakref.ref, torch.Tensor | None]]] = {}
# Compiled vLLM pads a GEMM's output width to a multiple of this many columns (the 20-head gate becomes 24).
VLLM_GEMM_OUTPUT_ALIGNMENT = 8


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


def _keep_for_recompute(norm: nn.Module, receiver: torch.Tensor, statistic: torch.Tensor | None) -> None:
    entries = [(ref, kept) for ref, kept in _RECOMPUTE_STATISTICS.get(id(norm), []) if ref() is not None]
    entries.append((weakref.ref(receiver), statistic))
    _RECOMPUTE_STATISTICS[id(norm)] = entries


def _take_for_recompute(norm: nn.Module, receiver: torch.Tensor) -> torch.Tensor | None:
    entries = _RECOMPUTE_STATISTICS.get(id(norm), [])
    for index, (ref, statistic) in enumerate(entries):
        original = ref()
        if original is not None and same_storage(original, receiver):
            del entries[index]
            return statistic
    raise RuntimeError("an input norm's recompute found no statistic from its checkpoint unit's first forward")


def _variance_with_gradient(variance: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
    """The unrounded sum's ``variance``, differentiated as the variance of the bf16 input it was rounded to."""
    return vllm_value(variance, lambda: hidden_states.float().pow(2).mean(dim=-1, keepdim=True))


def _install_residual_hooks(layer: TransformerLayer) -> None:
    """Rebuild the MLP residual in fp32 for the ``mlp_residual`` and norm-input numerics."""
    stored: dict[str, torch.Tensor] = {}

    def active() -> bool:
        numerics = active_numerics()
        return numerics.mlp_residual or numerics.input_norm_variance or numerics.final_norm_fp32 or numerics.vllm_norms

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
        parts = ResidualSum(stored.pop("residual"), stored.pop("routed"), stored.pop("shared"))
        # Compiled vLLM: h + (routed + shared) in fp32; the trainer rounds routed + shared first.
        if numerics.mlp_residual or numerics.vllm_norms:
            hidden = (parts.residual.float() + (parts.routed.float() + parts.shared.float())).to(output[0].dtype)
        else:
            hidden = (parts.residual.float() + (parts.routed + parts.shared).float()).to(output[0].dtype)
        if numerics.input_norm_variance or numerics.final_norm_fp32 or numerics.vllm_norms:
            hand_off(hidden, parts)
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


def _recomputing_one_layer(config: TransformerConfig) -> bool:
    """True in full activation recompute's second forward of a checkpointed unit that holds one layer.

    Megatron re-runs each checkpointed unit's forward inside the autograd engine's backward, where the engine's
    graph task is set. A unit is one layer under ``recompute_method`` ``block``, or ``uniform`` with one layer.
    """
    one_layer_units = config.recompute_granularity == "full" and (
        config.recompute_method == "block" or config.recompute_num_layers == 1
    )
    return one_layer_units and torch.is_grad_enabled() and torch._C._current_graph_task_id() != -1


def _install_vllm_experts_hooks(experts: TEGroupedMLP) -> None:
    """Take the routed experts' values from vLLM's fused-MoE kernels when ``vllm_experts`` is active.

    The kernels compute each dispatched row (one token-expert slot) as vLLM does, with the route weight inside
    the down projection's fp32 accumulator. With gradients enabled the trainer's grouped-GEMM experts also run
    and carry the gradient under the kernels' bytes through the exact-zero ``x - x.detach()``; a scoring forward
    without gradients runs the kernels alone.

    Full recompute's second forward of a one-layer unit runs the grouped-GEMM experts alone: that forward only
    rebuilds the layer's graph for its backward, and the layer uses the experts' output only in sums (the
    combine, the shared expert and the residuals), whose gradients do not depend on the summands' values. The
    gradients equal those of a forward that also runs the kernels; the layer's output, the first forward's,
    keeps the kernels' bytes.
    """
    grouped_forward = experts.forward

    def forward(permuted_local_hidden_states, tokens_per_expert, permuted_probs):
        if not active_numerics().vllm_experts or _recomputing_one_layer(experts.config):
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
    """Compiled vLLM's attention projections: q, k and v as three GEMMs, and the head gate as the engine's GEMM.

    Inductor's ``pad_mm`` pass times the 20-output gate projection against a copy padded with zero rows to a multiple
    of 8 when it compiles a graph, so an engine's graph holds one GEMM or the other: the width the probe recorded for
    this layer (``recorded_gate_columns``), else the padded one (fusion map, M-rope:1119). Under ``vllm_steps`` each
    sequence's gate rows run at its logged step's row count.
    """
    qkv = attention.linear_qkv
    groups = attention.num_query_groups_per_partition
    head_dim = attention.hidden_size_per_attention_head
    query_width = attention.num_attention_heads_per_partition // groups * head_dim
    gate = attention.attn_gate
    layer = attention.layer_number - 1

    def head_gate(x: torch.Tensor) -> torch.Tensor:
        heads = gate.weight.shape[0]
        columns = vllm_inductor.recorded_gate_columns(layer) or heads + -heads % VLLM_GEMM_OUTPUT_ALIGNMENT
        weight = torch.cat((gate.weight, gate.weight.new_zeros(columns - heads, gate.weight.shape[1])))
        if active_numerics().vllm_steps:
            if x.ndim != 3:
                raise NotImplementedError("vllm_steps numerics run the head gate on [S, B, H] inputs")
            return step_rows_linear(x, weight, planned_vllm_steps(x.shape[1]))[..., :heads].contiguous()
        return F.linear(x, weight)[..., :heads].contiguous()

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

        numerics = active_numerics()
        if not (numerics.fa3_attention or numerics.vllm_steps):
            return cudnn()
        if packed_seq_params is not None or attention_bias is not None or kwargs:
            raise NotImplementedError("fa3_attention numerics support unpacked causal sequences only")
        steps = planned_vllm_steps(query.shape[1]) if numerics.vllm_steps else None
        output = fa3_attention_sbhd(
            query,
            key,
            value,
            window=attention.fa3_window,
            scale=attention.fa3_scale,
            steps=steps,
            window_rows=numerics.fa3_window_rows,
        )
        if not torch.is_grad_enabled():
            return output
        reference = cudnn()
        return output.detach() + (reference - reference.detach())

    core.forward = forward


def _install_ep_combine_hooks(layer: TransformerLayer) -> None:
    """Add the routed expert outputs in vLLM's expert-parallel order when ``ep_sum`` is active.

    The value comes from ``vllm_ep_combine``; the gradient is the trainer's own unpermute (each slot's
    gradient is the token's output gradient in both), through the exact-zero ``x - x.detach()``. A forward
    without gradients skips the trainer's unpermute.

    Full recompute's second forward of a one-layer unit takes the trainer's unpermute alone: that forward
    only rebuilds the layer's graph for its backward, and the layer uses the routed output only in sums and
    casts (the shared expert and the residuals), whose gradients do not depend on the summands' values. The
    gradients equal those of a forward that also runs the combine; the layer's output, the first forward's,
    keeps the combine's bytes.
    """
    dispatcher = layer.mlp.token_dispatcher
    router = layer.mlp.router
    unpermute = dispatcher.combine_postprocess

    def combine_postprocess(permuted):
        if not active_numerics().ep_sum:
            return unpermute(permuted)
        if dispatcher.shared_experts is not None:
            raise NotImplementedError("ep_sum numerics expect the shared expert outside the dispatcher")
        selected, expert_parallel = router.take_ep_route()
        if _recomputing_one_layer(layer.config):
            return unpermute(permuted)
        with torch.no_grad():
            combined = vllm_ep_combine(
                permuted, dispatcher.routing_map, selected, expert_parallel.dp_ranks, expert_parallel.ep_size
            ).view(dispatcher.hidden_shape)
        if not torch.is_grad_enabled():
            return combined
        output = unpermute(permuted)
        return combined + (output - output.detach())

    dispatcher.combine_postprocess = combine_postprocess


def _install_shared_swiglu_hooks(shared: SharedExpertMLP) -> None:
    """Recompute the shared expert's activation with one rounding under ``shared_swiglu`` or ``vllm_swiglu``.

    Under ``vllm_swiglu`` the value comes from compiled vLLM's activation kernel and the gradient from
    ``shared_swiglu``'s computation.
    """
    stored: dict[str, torch.Tensor] = {}

    def recomputed() -> bool:
        numerics = active_numerics()
        return numerics.shared_swiglu or numerics.vllm_swiglu

    def keep_fc1_output(module, args, output):
        if recomputed():
            stored["fc1"] = output[0] if isinstance(output, tuple) else output

    def replace_fc2_input(module, args):
        if not recomputed():
            return None
        fc1_output = stored.pop("fc1", None)
        if fc1_output is None:
            raise RuntimeError("shared_swiglu and vllm_swiglu require the shared expert's fc1 output")
        if not active_numerics().vllm_swiglu:
            return (swiglu_single_rounding(fc1_output), *args[1:])
        gate, up = torch.chunk(fc1_output, 2, dim=-1)
        value = vllm_inductor.shared_activation(gate, up).view(gate.shape)
        return (vllm_value(value, lambda: swiglu_single_rounding(fc1_output)), *args[1:])

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
    """Drop the hand-offs the numerics hooks pass from one module to the next within a forward.

    Each forward starts empty: a pipeline stage's last residual has no reader, and the entries are keyed by tensors
    that a later forward may reuse. The statistics kept for recompute stay: under pipeline parallelism a micro-batch's
    backward, and its recompute, runs after later micro-batches' forwards.
    """
    clear_hand_offs()


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
        weight = self.norm.weight
        if self.role is NormRole.INPUT:
            normalized = self._input_norm(hidden_states, numerics)
        elif self.role is NormRole.FINAL:
            normalized = self._final_norm(hidden_states, numerics)
        elif numerics.vllm_norms and self.role is NormRole.EMBEDDING:
            value = vllm_inductor.embedding_norm(hidden_states, weight).view_as(hidden_states)
            normalized = vllm_value(value, lambda: self.norm(hidden_states))
        elif numerics.vllm_norms:
            value = vllm_inductor.rms_norm(hidden_states, weight).view_as(hidden_states)
            normalized = vllm_value(value, lambda: self.norm(hidden_states))
        else:
            normalized = self.norm(hidden_states)
        gate = self.up_proj(F.silu(self.down_proj(normalized)))
        if numerics.vllm_norms:
            value = vllm_inductor.gated_product(normalized, gate).view_as(normalized)
            output = vllm_value(value, lambda: gated_norm_product_fp32(normalized, gate).to(normalized.dtype))
        elif numerics.gated_norm:
            output = gated_norm_product_fp32(normalized, gate).to(normalized.dtype)
        else:
            output = normalized * torch.sigmoid(gate)
        if self.role is NormRole.EMBEDDING and (numerics.input_norm_variance or numerics.vllm_norms):
            # Layer 0's input norm takes its variance from the unrounded embedding gated-norm product.
            hand_off(output, GatedProduct(normalized, gate))
        return output

    def _input_norm(self, hidden_states: torch.Tensor, numerics) -> torch.Tensor:
        """A layer's input norm: compiled vLLM takes the variance from the unrounded sum that formed its input."""
        if not (numerics.input_norm_variance or numerics.vllm_norms):
            return self.norm(hidden_states)
        weight = self.norm.weight
        statistic = self._input_statistic(hidden_states, numerics)
        if statistic is None:
            # A layer run on its own, with no layer or pipeline stage handing its sum on: normalize the rounded input
            # by its own variance.
            if not numerics.vllm_norms:
                return self.norm(hidden_states)
            value = vllm_inductor.rms_norm(hidden_states, weight).view_as(hidden_states)
            return vllm_value(value, lambda: self.norm(hidden_states))
        if not numerics.vllm_norms:
            variance = _variance_with_gradient(statistic, hidden_states)
            return rms_norm_hybrid(hidden_states, variance, weight, self.eps)
        value = vllm_inductor.rms_norm_from_square_sum(hidden_states, statistic, weight).view_as(hidden_states)
        variance = (statistic / hidden_states.shape[-1]).view(*hidden_states.shape[:-1], 1)
        return vllm_value(
            value,
            lambda: rms_norm_hybrid(hidden_states, _variance_with_gradient(variance, hidden_states), weight, self.eps),
        )

    def _input_statistic(self, hidden_states: torch.Tensor, numerics) -> torch.Tensor | None:
        """The input norm's statistic of the unrounded sum: its variance, or compiled vLLM's sum of squares.

        A checkpoint unit's first forward keeps what it computed for the recompute, which cannot reach the hand-off.
        """
        phase = checkpoint_pass()
        parts = take_hand_off(hidden_states)
        if parts is None and phase is CheckpointPass.RECOMPUTE:
            return _take_for_recompute(self, hidden_states)
        statistic = None
        if parts is not None:
            with torch.no_grad():
                statistic = parts.statistic(numerics)
        if phase is CheckpointPass.FIRST:
            _keep_for_recompute(self, hidden_states, statistic)
        return statistic

    def _final_norm(self, hidden_states: torch.Tensor, numerics) -> torch.Tensor:
        """The final norm: compiled vLLM normalizes the last layer's unrounded sum."""
        parts = take_hand_off(hidden_states)
        if parts is None or not (numerics.final_norm_fp32 or numerics.vllm_norms):
            return self.norm(hidden_states)
        if not isinstance(parts, ResidualSum):
            raise RuntimeError("the final norm's input must come from a decoder layer's residual sum")
        # The hand-off may come from a no-grad checkpointed forward; keep its values and take the gradient
        # through the bf16 residual so the loss still reaches the layers before this norm. ``x - x.detach()``
        # is exactly zero, so the values are unchanged bit for bit.
        residual = hidden_states.float()

        def reference() -> torch.Tensor:
            unrounded = parts.unrounded() + (residual - residual.detach())
            return rms_norm_single_rounding(unrounded, self.norm.weight, self.eps)

        if not numerics.vllm_norms:
            return reference()
        value = vllm_inductor.final_norm(parts.residual, parts.routed, parts.shared, self.norm.weight)
        return vllm_value(value.view_as(hidden_states), reference)


def qk_norm_fp32(hidden_states: torch.Tensor) -> torch.Tensor:
    """Grug's weightless q/k RMS norm per head in fp32, unrounded."""
    fp32 = hidden_states.float()
    return fp32 * torch.rsqrt(fp32.square().mean(dim=-1, keepdim=True) + GRUG_QK_RMS_NORM_EPS)


class GrugQKNorm(nn.Module):
    """Weightless RMS norm applied to each query and key head."""

    def __init__(self, config: TransformerConfig, hidden_size: int, eps: float):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        numerics = active_numerics()
        if numerics.vllm_qk:
            # The attention forward runs compiled vLLM's q/k norm, RoPE and query scale on the raw projections.
            return hidden_states
        if numerics.qk_rope:
            # The attention forward rounds once after RoPE and the query scale.
            return qk_norm_fp32(hidden_states)
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

        numerics = active_numerics()
        if numerics.vllm_qk:
            if is_thd:
                raise NotImplementedError("vllm_qk numerics support unpacked sequences only")
            query, key = self._vllm_query_key(query, key, rotary_pos_emb)
        elif numerics.qk_rope:
            if is_thd:
                raise NotImplementedError("qk_rope numerics support unpacked sequences only")
            query, key = self._rounded_query_key(query, key, rotary_pos_emb, value.dtype)
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
        if numerics.vllm_xsa:
            if is_thd:
                raise NotImplementedError("vllm_xsa numerics support unpacked sequences only")
            attention = core_attn_out
            xsa = vllm_inductor.xsa_head_gate(attention, value, gate).view_as(attention)
            core_attn_out = vllm_value(
                xsa, lambda: xsa_and_gate_single_rounding(attention, value, gate, self.hidden_size_per_attention_head)
            )
        elif numerics.xsa_gate:
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

    def _rounded_query_key(
        self, query: torch.Tensor, key: torch.Tensor, rotary_pos_emb, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compiled vLLM's rounding points for the fp32-normalized q and k: RoPE with the bf16 table, then the scale.

        k rounds once. q's rotated half is stored once before the scale and rounds again after it; its pass-through
        half rounds once.
        """
        if rotary_pos_emb is not None and not self.skip_rope:
            q_pos_emb, k_pos_emb = rotary_pos_emb if isinstance(rotary_pos_emb, tuple) else (rotary_pos_emb,) * 2
            rotary_dim = q_pos_emb.shape[-1]
            query = rotate_neox_fp32(query, q_pos_emb)
            query = torch.cat((query[..., :rotary_dim].to(dtype).float(), query[..., rotary_dim:]), dim=-1)
            key = rotate_neox_fp32(key, k_pos_emb)
        return (query.float() * self.qk_mult * self.qk_mult_scale).to(dtype), key.to(dtype)

    def _vllm_query_key(
        self, query: torch.Tensor, key: torch.Tensor, rotary_pos_emb
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compiled vLLM's q/k norm, RoPE and query scale on the raw ``[S, B, heads, dim]`` projections.

        The values come from vLLM's Inductor kernels; with gradients enabled they are differentiated as the
        ``qk_rope`` chain, which computes the same values up to rounding.
        """
        if (self.qk_mult, self.qk_mult_scale) != vllm_inductor.QUERY_FACTORS:
            raise NotImplementedError(f"vllm_qk kernels were compiled for query factors {vllm_inductor.QUERY_FACTORS}")
        sequence, batch = query.shape[:2]
        if rotary_pos_emb is not None and not self.skip_rope:
            positions = torch.arange(sequence, device=query.device).repeat_interleave(batch)
            vllm_query, vllm_key = vllm_inductor.query_key_rope(query, key, positions, self.config.rotary_base)
        else:
            vllm_query, vllm_key = vllm_inductor.query_key_full(query, key)
        vllm_query, vllm_key = vllm_query.view_as(query), vllm_key.view_as(key)
        if not torch.is_grad_enabled():
            return vllm_query, vllm_key
        reference_query, reference_key = self._rounded_query_key(
            qk_norm_fp32(query), qk_norm_fp32(key), rotary_pos_emb, query.dtype
        )
        return vllm_value(vllm_query, lambda: reference_query), vllm_value(vllm_key, lambda: reference_key)

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
        numerics = active_numerics()
        if numerics.invariant_router:
            logits = self._invariant_logits(input, numerics)
        elif numerics.vllm_steps:
            # Each sequence's logits from the fp32 GEMM at the row count of the vLLM step that computed it.
            if numerics.router_rows or input.ndim != 3:
                raise NotImplementedError("vllm_steps numerics take the router GEMM's rows from the logged step alone")
            logits = step_rows_linear(input.float(), self.weight.float(), planned_vllm_steps(input.shape[1]))
        elif numerics.router_gemm or numerics.router_rows:
            # Compiled vLLM runs an fp32 GEMM on the stored bf16 input and fp32 weights holding the bf16 values.
            fp32_input, weight = input.float(), self.weight.float()
            if not numerics.router_rows:
                logits = F.linear(fp32_input, weight)
            else:
                # cuBLAS sums the fp32 GEMM in an order that follows its row count; vLLM's full prefill step has
                # VLLM_MAX_BATCHED_TOKENS rows. The value comes from calls of that size, the gradient from the
                # plain GEMM through the exact-zero ``x - x.detach()``.
                with torch.no_grad():
                    logits = fixed_rows_linear(fp32_input, weight, VLLM_MAX_BATCHED_TOKENS)
                if torch.is_grad_enabled():
                    reference = F.linear(fp32_input, weight)
                    logits = logits + (reference - reference.detach())
        else:
            logits = self.gating(input)
        return self.routing(logits, padding_mask)

    def _invariant_logits(self, input: torch.Tensor, numerics) -> torch.Tensor:
        """The router logits a decode-invariant vLLM engine computes: the row-invariant Triton GEMM on the bf16 input
        and the bf16 weight (exact products, fp32 sums), whatever rows share the call; the gradient is the fp32 GEMM's."""
        # Triton is not in the CPU test lane; the kernel module imports it.
        from skyrl_train.models.grug_invariant_kernels import invariant_router_logits

        if numerics.router_rows or numerics.vllm_steps:
            raise NotImplementedError("invariant_router replaces router_rows and vllm_steps' router rows")
        if self.weight.dtype != torch.bfloat16 or input.dtype != torch.bfloat16:
            raise ValueError("invariant_router multiplies the bf16 router input by the bf16 router weight")
        with torch.no_grad():
            value = invariant_router_logits(input.reshape(-1, input.shape[-1]), self.weight)
        value = value.view(*input.shape[:-1], value.shape[-1])
        return vllm_value(value, lambda: F.linear(input.float(), self.weight.float()))


def _install_step_lm_head(model: GPTModel) -> None:
    """Compute the LM head's logits at the row counts of each sequence's logged vLLM step under ``vllm_steps``."""
    output_layer = model.output_layer
    output_forward = output_layer.forward

    def forward(hidden_states, weight=None, runtime_gather_output=None, **kwargs):
        if not active_numerics().vllm_steps:
            return output_forward(hidden_states, weight=weight, runtime_gather_output=runtime_gather_output, **kwargs)
        if kwargs or output_layer.bias is not None or model.config.tensor_model_parallel_size != 1:
            raise NotImplementedError("vllm_steps numerics need the unsharded, bias-free LM head")
        head = output_layer.weight if weight is None else weight
        return step_lm_head_logits(hidden_states, head, planned_vllm_steps(hidden_states.shape[1])), None

    output_layer.forward = forward


def _stage_output_statistic(output: torch.Tensor) -> torch.Tensor | None:
    """The next stage's first input-norm statistic, from the residual sum that formed this stage's last output.

    ``None`` when the numerics take no statistic from the unrounded sum.
    """
    numerics = active_numerics()
    if not (numerics.input_norm_variance or numerics.vllm_norms):
        return None
    parts = take_hand_off(output)
    if not isinstance(parts, ResidualSum):
        raise RuntimeError("a pipeline stage's output carries no residual sum to hand its input-norm statistic on")
    with torch.no_grad():
        return parts.statistic(numerics).reshape(*output.shape[:-1], 1)


class GrugGPTModel(GPTModel):
    """GPTModel with Grug's gated embedding norm on the first pipeline stage.

    Under the norm-input numerics a stage that is not the last appends each token's input-norm statistic (one fp32,
    as two bf16 words) to the hidden states it hands on, and the next stage splits it off for its first layer's input
    norm (``StageStatistic``). Megatron's pipeline exchanges each micro-batch's tensor shape, so the wider tensor
    rides the existing point-to-point transfer in the forward and its gradient in the backward.
    """

    def __init__(self, config: TransformerConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        install_numerics_hooks(self)
        self._received: tuple[torch.Tensor, torch.Tensor] | None = None
        if self.post_process and isinstance(self.decoder.final_layernorm, GrugGatedRMSNorm):
            self.decoder.final_layernorm.role = NormRole.FINAL
            _install_step_lm_head(self)
        if self.pre_process:
            self.embed_norm = GrugGatedRMSNorm(
                config=config, hidden_size=config.hidden_size, eps=config.layernorm_epsilon
            )
            self.embed_norm.role = NormRole.EMBEDDING

    def set_input_tensor(self, input_tensor) -> None:
        tensors = input_tensor if isinstance(input_tensor, list) else [input_tensor]
        if len(tensors) != 1:
            raise ValueError("a Grug pipeline stage receives one tensor")
        received = tensors[0]
        self._received = None
        if received is not None and received.shape[-1] == self.config.hidden_size + STAGE_STATISTIC_COLUMNS:
            self._received = split_stage_statistic(received, self.config.hidden_size)
            received = self._received[0]
        super().set_input_tensor(received)

    def forward(self, *args, **kwargs):
        clear_numerics_handoffs()
        if self._received is not None:
            hidden, statistic = self._received
            self._received = None
            hand_off(hidden, StageStatistic(statistic))
        output = super().forward(*args, **kwargs)
        if self.post_process:
            return output
        statistic = _stage_output_statistic(output)
        return output if statistic is None else append_stage_statistic(output, statistic)

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
