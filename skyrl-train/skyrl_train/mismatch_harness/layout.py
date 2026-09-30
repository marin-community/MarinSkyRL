"""How the trainer's Grug layer changes one sequence's bytes with its micro-batch layout (one H100).

The probe's batch-layout control (the ``repeat`` modes) scores every sequence again in reversed order with twice
the micro-batch size: a sequence then shares its micro-batch with another one, sits at another batch index, and
is right-padded to the longer of the two. This module runs one Snowball decoder layer (real weights) on one
captured sequence inside such layouts and reports, for every region of the layer, the fraction of that
sequence's elements that stay byte-equal to the sequence scored alone:

- ``chained``: the layer runs on its own values, so a region inherits every difference upstream of it;
- ``isolated``: after each region the sequence's rows are overwritten with the lone run's values, so every
  region reads identical inputs and only its own dependence on the layout shows.

The sequence's experts are replayed from the lone run in every layout, as the probe's replay modes do; the
other rows of a micro-batch are rows of the captured sequence drawn at random, so they carry real activations.
Three sweeps widen the layouts beyond the probe's:

- ``gemm``: each projection the trainer (Transformer Engine) or compiled vLLM (``torch.mm``) runs, the RMS norm,
  the q/k norm and the router GEMM, on the same rows placed first or last in calls of 1 to 8,192 rows;
- ``experts``: the routed experts (TE grouped GEMM, or the numerics' expert kernel) on the same token-expert rows
  among other tokens' rows, which changes every expert's row count;
- ``attention``: the core attention on the same sequence at other padded lengths, batch sizes and batch indices.

It also runs the LM head and the trainer's log-probability computation on the same final hidden rows in each
layout. Example::

    python -m skyrl_train.mismatch_harness.layout --capture s3://.../pp-0.pt --capture s3://.../pp-1.pt \\
        --model s3://.../hf-bf16-vllm --layers 0 3 13 25 --input-from 25=13 \\
        --numerics compiled_stack=gated_norm,... --output s3://.../layout/<name>
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import tempfile
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from marinskyrl.resource_locator import join_resource_path
from megatron.core import parallel_state
from megatron.core.transformer.enums import AttnMaskType

from skyrl_train.distributed.megatron.model_utils import from_parallel_logits_to_logprobs
from skyrl_train.io import io
from skyrl_train.mismatch_harness.numerics import compare
from skyrl_train.mismatch_harness.run import (
    export_weights,
    layer_weight_names,
    load_captures,
    parse_input_sources,
    parse_variants,
    stage_config,
)
from skyrl_train.mismatch_harness.trainer_side import (
    build_gated_norm,
    build_layer,
    grug_provider,
    load_hf_weights,
    rotary_embedding,
    single_rank_megatron,
)
from skyrl_train.mismatch_harness.vllm_side import GrugShape, vllm_config_context
from skyrl_train.mismatch_probe.numerics import grug_numerics
from skyrl_train.models.grug_megatron import NormRole, clear_numerics_handoffs
from skyrl_train.models.grug_moe import grug_rms_norm_no_weight, jax_top_k
from skyrl_train.models.grug_rounding import rms_norm_hybrid
from skyrl_train.models.megatron_router_replay import LayerReplayHandle, MegatronRouterReplay, VllmExpertParallel

REFERENCE_LAYOUT = "alone"
# vLLM's expert-parallel size and the home rank the ``ep_sum`` numerics are given for every row.
EP_SIZE = 8
HOME_RANK = 0
SWEEP_REFERENCE_ROWS = 304
SWEEP_ROWS = (1, 2, 3, 8, 16, 31, 64, 127, 128, 255, 300, 304, 512, 600, 913, 1024, 1826, 2048, 4096, 7304, 8192)
EXPERT_SWEEP_FILLER_TOKENS = (0, 1, 7, 64, 300, 1024, 4096)
ATTENTION_SWEEP_LENGTHS = (0, 4, 84, 212, 300, 613, 724, 1748)
ATTENTION_SWEEP_BATCHES = (1, 2, 4, 8)
LOGPROB_CHUNK = 1024
DENSE_PROBE_ROWS = 16


@dataclass(frozen=True)
class Layout:
    """A micro-batch of sequences of the given lengths, the probed one at batch index ``index``."""

    lengths: tuple[int, ...]
    index: int

    @property
    def sequence(self) -> int:
        return max(self.lengths)

    @property
    def batch(self) -> int:
        return len(self.lengths)

    @property
    def length(self) -> int:
        return self.lengths[self.index]

    def token_indices(self, device) -> torch.Tensor:
        """The probed sequence's rows in the router's sequence-major token order (``s * B + b``)."""
        return torch.arange(self.length, device=device) * self.batch + self.index


def probe_layouts(length: int) -> dict[str, Layout]:
    """The lone sequence and the micro-batches the probe's repeat layout (and larger batches) put it in."""
    return {
        REFERENCE_LAYOUT: Layout((length,), 0),
        "second_after_longer": Layout((2 * length, length), 1),
        "first_before_shorter": Layout((length, length // 2), 0),
        "first_before_longer": Layout((length, 3 * length), 0),
        "fourth_of_eight": Layout((144, 913, 400, length, 620, 250, 800, 500), 3),
    }


def filler_rows(source: torch.Tensor, count: int, generator: torch.Generator) -> torch.Tensor:
    """``count`` rows drawn at random from ``source`` rows (real activations for the other sequences)."""
    picks = torch.randint(0, source.shape[0], (count,), generator=generator, device=source.device)
    return source[picks]


def layout_input(ours: torch.Tensor, layout: Layout, generator: torch.Generator) -> torch.Tensor:
    """``[S, B, ...]`` with the probed rows at ``[:L, index]`` and random rows of ``ours`` everywhere else."""
    batch = filler_rows(ours, layout.sequence * layout.batch, generator).view(
        layout.sequence, layout.batch, *ours.shape[1:]
    )
    batch[: layout.length, layout.index] = ours
    return batch


@dataclass
class Tap:
    """Records the probed rows at each region and, given a reference run, overwrites them with its values."""

    layout: Layout
    reference: Mapping[str, torch.Tensor] | None = None
    values: dict[str, torch.Tensor] = field(default_factory=dict)
    selected: torch.Tensor | None = None
    """The probed tokens' experts ``[L, K]`` in ascending id order (the canonical order of expert rows)."""
    expert_rows: torch.Tensor | None = None
    """The permuted rows of the probed ``(token, expert)`` pairs, in ``selected`` order."""

    def _substitute(self, name: str) -> torch.Tensor | None:
        if self.reference is None:
            return None
        return self.reference.get(name)

    def sequence(self, name: str, tensor: torch.Tensor) -> None:
        """``tensor`` is ``[S, B, ...]``."""
        if name in self.values:
            return
        rows = tensor[: self.layout.length, self.layout.index]
        self.values[name] = rows.detach().clone()
        replacement = self._substitute(name)
        if replacement is not None:
            rows.copy_(replacement)

    def tokens(self, name: str, tensor: torch.Tensor) -> None:
        """``tensor`` is ``[S * B, ...]`` in sequence-major order."""
        if name in self.values:
            return
        index = self.layout.token_indices(tensor.device)
        self.values[name] = tensor[index].detach().clone()
        replacement = self._substitute(name)
        if replacement is not None:
            tensor[index] = replacement

    def experts(self, name: str, tensor: torch.Tensor) -> None:
        """``tensor`` is the permuted ``[token-expert rows, ...]`` of the dispatcher."""
        if name in self.values or self.expert_rows is None:
            return
        self.values[name] = tensor[self.expert_rows].detach().clone()
        replacement = self._substitute(name)
        if replacement is not None:
            tensor[self.expert_rows] = replacement

    def set_routing(self, routing_map: torch.Tensor) -> None:
        """Locate the probed tokens' rows in the dispatcher's permuted order (by expert, then by token)."""
        index = self.layout.token_indices(routing_map.device)
        mine = routing_map[index]
        self.selected = mine.nonzero()[:, 1].view(self.layout.length, -1)
        pairs = routing_map.t().nonzero()
        row_of = torch.full(routing_map.shape, -1, dtype=torch.long, device=routing_map.device)
        row_of[pairs[:, 1], pairs[:, 0]] = torch.arange(pairs.shape[0], device=routing_map.device)
        self.expert_rows = row_of[index[:, None], self.selected].flatten()


def _first(value):
    return value[0] if isinstance(value, (tuple, list)) else value


@contextmanager
def tapped(layer: torch.nn.Module, next_norm: torch.nn.Module, tap: Tap) -> Iterator[None]:
    """Hook every region boundary of ``layer`` (and the following norm) into ``tap``."""
    handles = []

    def on_output(module, name, kind=Tap.sequence):
        handles.append(module.register_forward_hook(lambda m, args, out: kind(tap, name, _first(out))))

    def on_input(module, name, kind=Tap.sequence):
        handles.append(module.register_forward_pre_hook(lambda m, args: kind(tap, name, args[0])))

    for prefix, norm in (("attn", layer.input_layernorm), ("mlp", layer.pre_mlp_layernorm), ("next_attn", next_norm)):
        on_input(norm.down_proj, f"{prefix}_rms")
        on_output(norm.down_proj, f"{prefix}_gate_down")
        on_input(norm.up_proj, f"{prefix}_gate_act")
        on_output(norm.up_proj, f"{prefix}_gate_up")
    on_output(layer.input_layernorm, "attention_norm")
    on_output(next_norm, "next_attention_norm")
    attention = layer.self_attention
    on_output(attention.linear_qkv, "qkv")
    on_input(attention.q_layernorm, "q_proj")
    on_output(attention.q_layernorm, "q_norm")
    on_input(attention.k_layernorm, "k_proj")
    on_output(attention.k_layernorm, "k_norm")

    def attention_inputs(module, args):
        for name, tensor in zip(("query", "key", "value"), args[:3], strict=True):
            tap.sequence(name, tensor)

    handles.append(attention.core_attention.register_forward_pre_hook(attention_inputs))
    on_output(attention.core_attention, "core_attention")
    on_output(attention.attn_gate, "attn_gate")
    on_input(attention.linear_proj, "xsa_gate")
    on_output(attention.linear_proj, "o_proj")
    on_input(layer.pre_mlp_layernorm, "residual_after_attention")
    on_output(layer.pre_mlp_layernorm, "mlp_norm")
    moe = layer.mlp
    router, dispatcher = moe.router, moe.token_dispatcher
    routing, combine = router.routing, dispatcher.combine_postprocess

    def tapped_routing(logits, padding_mask=None):
        tap.tokens("router_logits", logits.view(-1, logits.shape[-1]))
        probs, routing_map = routing(logits, padding_mask)
        tap.set_routing(routing_map)
        tap.tokens("router_probs", probs)
        return probs, routing_map

    def tapped_combine(permuted):
        output = combine(permuted)
        tap.sequence("routed", output)
        return output

    router.routing = tapped_routing
    dispatcher.combine_postprocess = tapped_combine
    experts = moe.experts
    on_output(experts.linear_fc1, "experts_fc1", Tap.experts)
    on_input(experts.linear_fc2, "experts_act", Tap.experts)
    on_output(experts.linear_fc2, "experts_fc2", Tap.experts)
    on_output(experts, "expert_slots", Tap.experts)
    shared = moe.shared_experts
    on_output(shared.linear_fc1, "shared_fc1")
    on_input(shared.linear_fc2, "shared_act")
    on_output(shared.linear_fc2, "shared_down")
    on_output(layer, "output")
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
        router.routing, dispatcher.combine_postprocess = routing, combine


def native_experts(layer, hidden: torch.Tensor, rotary_pos_emb, numerics: Mapping[str, bool]) -> torch.Tensor:
    """The experts the router picks natively for every row of ``hidden`` ``[S, B, H]``, in selection order."""
    router = layer.mlp.router
    kept = {}
    routing = router.routing

    def keep(logits, padding_mask=None):
        biased = logits.view(-1, router.config.num_moe_experts).float() + router.expert_bias
        kept["selected"] = jax_top_k(biased, router.topk + 1)[1][:, : router.topk]
        return routing(logits, padding_mask)

    router.routing = keep
    try:
        forward(layer, hidden, rotary_pos_emb, numerics, None)
    finally:
        router.routing = routing
    return kept["selected"]


def forward(layer, hidden, rotary_pos_emb, numerics, replay: tuple[torch.Tensor, torch.Tensor] | None):
    """One no-grad forward of ``layer``; ``replay`` holds ``[S * B, K]`` targets and a ``[S * B]`` mask."""
    router = layer.mlp.router
    controller = MegatronRouterReplay([0], recompute_enabled=False)
    router.router_replay = LayerReplayHandle(controller, 0)
    rows = hidden.shape[0] * hidden.shape[1]
    targets, mask = (
        replay
        if replay is not None
        else (
            torch.zeros(rows, router.topk, dtype=torch.long, device=hidden.device),
            torch.zeros(rows, dtype=torch.bool, device=hidden.device),
        )
    )
    placement = VllmExpertParallel(torch.full((rows,), HOME_RANK, dtype=torch.long, device=hidden.device), EP_SIZE)
    controller.begin_forward({0: targets}, mask, vllm_expert_parallel=placement)
    clear_numerics_handoffs()
    try:
        with torch.no_grad(), grug_numerics(**numerics):
            output, _ = layer(hidden_states=hidden, attention_mask=None, rotary_pos_emb=rotary_pos_emb)
        controller.end_forward()
        return output
    finally:
        router.router_replay = None


def run_in_layout(
    layer,
    next_norm,
    rotary,
    hidden: torch.Tensor,
    layout: Layout,
    routes: torch.Tensor,
    numerics: Mapping[str, bool],
    reference: Mapping[str, torch.Tensor] | None,
) -> dict[str, torch.Tensor]:
    """The probed rows of every region of ``hidden`` (laid out as ``layout``) with the sequence's experts replayed."""
    rows = layout.sequence * layout.batch
    index = layout.token_indices(hidden.device)
    targets = torch.zeros(rows, routes.shape[1], dtype=torch.long, device=hidden.device)
    targets[index] = routes
    mask = torch.zeros(rows, dtype=torch.bool, device=hidden.device)
    mask[index] = True
    tap = Tap(layout, reference)
    with tapped(layer, next_norm, tap):
        output = forward(layer, hidden, rotary(layout.sequence), numerics, (targets, mask))
        with torch.no_grad(), grug_numerics(**numerics):
            next_norm(output)
        clear_numerics_handoffs()
    return {**tap.values, "input": hidden[: layout.length, layout.index].clone()}


def region_stats(values: Mapping[str, torch.Tensor], reference: Mapping[str, torch.Tensor]) -> dict[str, dict]:
    stats = {}
    for name, tensor in values.items():
        if name not in reference or tensor.dtype == torch.bool:
            continue
        other = reference[name]
        if tensor.shape != other.shape:
            stats[name] = {"shape_mismatch": [list(tensor.shape), list(other.shape)]}
            continue
        stats[name] = compare(tensor.reshape(tensor.shape[0], -1), other.reshape(other.shape[0], -1)).to_json()
    return stats


def layer_layouts(
    layer, next_norm, rotary, ours: torch.Tensor, numerics: Mapping[str, bool], generator: torch.Generator
) -> tuple[dict, dict[str, torch.Tensor]]:
    """Chained and isolated region stats of every probe layout against the lone sequence, and the lone run."""
    layouts = probe_layouts(ours.shape[0])
    alone = layouts[REFERENCE_LAYOUT]
    hidden = layout_input(ours, alone, generator)
    routes = native_experts(layer, hidden, rotary(alone.sequence), numerics)
    reference = run_in_layout(layer, next_norm, rotary, hidden, alone, routes, numerics, None)
    repeat = run_in_layout(layer, next_norm, rotary, hidden, alone, routes, numerics, None)
    result = {"alone_repeat": region_stats(repeat, reference)}
    for name, layout in layouts.items():
        if name == REFERENCE_LAYOUT:
            continue
        hidden = layout_input(ours, layout, generator)
        chained = run_in_layout(layer, next_norm, rotary, hidden, layout, routes, numerics, None)
        isolated = run_in_layout(layer, next_norm, rotary, hidden, layout, routes, numerics, reference)
        result[name] = {
            "lengths": list(layout.lengths),
            "index": layout.index,
            "chained": region_stats(chained, reference),
            "isolated": region_stats(isolated, reference),
        }
    return result, reference


def equal_fraction(left: torch.Tensor, right: torch.Tensor) -> float:
    width = {torch.bfloat16: torch.int16, torch.float32: torch.int32}[left.dtype]
    return (left.contiguous().view(width) == right.contiguous().view(width)).float().mean().item()


def row_sweep(fn: Callable[[torch.Tensor], torch.Tensor], rows: torch.Tensor, generator: torch.Generator) -> dict:
    """``fn`` on ``rows`` placed first and last in calls of every ``SWEEP_ROWS`` size, against one call on them.

    ``fn`` maps ``[M, ...]`` to ``[M, ...]``; the other rows of a call are random rows of ``rows``.
    """
    limit = min(SWEEP_REFERENCE_ROWS, rows.shape[0])
    reference = fn(rows[:limit])
    result = {}
    for count in SWEEP_ROWS:
        kept = min(count, limit)
        for offset in sorted({0, count - kept}):
            batch = filler_rows(rows, count, generator)
            batch[offset : offset + kept] = rows[:kept]
            result[f"{count}@{offset}"] = equal_fraction(fn(batch)[offset : offset + kept], reference[:kept])
    return result


def gemm_sweeps(layer, reference: Mapping[str, torch.Tensor], lm_head: torch.Tensor, generator) -> dict:
    """Row-count and row-offset dependence of every projection, norm and router GEMM of the layer."""
    attention, shared, norm = layer.self_attention, layer.mlp.shared_experts, layer.input_layernorm
    heads = attention.num_attention_heads_per_partition
    head_dim = attention.hidden_size_per_attention_head
    attention_in = reference["attention_norm"]
    mlp_in = reference["mlp_norm"]

    def te(module):
        return lambda x: _first(module(x.unsqueeze(1))).squeeze(1)

    def torch_mm(weight):
        return lambda x: torch.mm(x, weight.t())

    qkv = attention.linear_qkv.weight.view(attention.num_query_groups_per_partition, -1, attention_in.shape[-1])
    per_group = heads // attention.num_query_groups_per_partition * head_dim
    weights = {
        "q": qkv[:, :per_group].reshape(-1, qkv.shape[-1]),
        "k": qkv[:, per_group : per_group + head_dim].reshape(-1, qkv.shape[-1]),
        "v": qkv[:, per_group + head_dim :].reshape(-1, qkv.shape[-1]),
        "attn_gate_24": torch.cat((attention.attn_gate.weight, attention.attn_gate.weight.new_zeros(4, qkv.shape[-1]))),
        "o": attention.linear_proj.weight,
        "shared_gate": shared.linear_fc1.weight[: shared.linear_fc1.weight.shape[0] // 2],
        "shared_up": shared.linear_fc1.weight[shared.linear_fc1.weight.shape[0] // 2 :],
        "shared_down": shared.linear_fc2.weight,
        "gated_norm_down": norm.down_proj.weight,
        "gated_norm_up": norm.up_proj.weight,
        "lm_head": lm_head,
    }
    inputs = {
        "q": attention_in,
        "k": attention_in,
        "v": attention_in,
        "attn_gate_24": attention_in,
        "o": reference["xsa_gate"],
        "shared_gate": mlp_in,
        "shared_up": mlp_in,
        "shared_down": reference["shared_act"],
        "gated_norm_down": reference["attn_rms"],
        "gated_norm_up": reference["attn_gate_act"],
        "lm_head": reference["next_attention_norm"],
    }
    router = layer.mlp.router
    sweeps: dict[str, dict] = {}
    with torch.no_grad():
        for name, weight in weights.items():
            sweeps[f"torch.mm {name}"] = row_sweep(torch_mm(weight.contiguous()), inputs[name], generator)
        for name, module, rows in (
            ("qkv", attention.linear_qkv, attention_in),
            ("attn_gate", attention.attn_gate, attention_in),
            ("o_proj", attention.linear_proj, reference["xsa_gate"]),
            ("shared_fc1", shared.linear_fc1, mlp_in),
            ("shared_fc2", shared.linear_fc2, reference["shared_act"]),
            ("rms_norm", norm.norm, reference["input"]),
        ):
            sweeps[f"TE {name}"] = row_sweep(te(module), rows, generator)
        sweeps["router bf16->fp32 (default)"] = row_sweep(
            lambda x: router.gating(x.unsqueeze(1)).squeeze(1), mlp_in, generator
        )
        previous = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            weight = router.weight.float()
            sweeps["router fp32 (router_gemm)"] = row_sweep(lambda x: F.linear(x.float(), weight), mlp_in, generator)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous
        sweeps["q/k norm (default)"] = row_sweep(
            lambda x: grug_rms_norm_no_weight(x.view(x.shape[0], -1, head_dim)).flatten(1),
            reference["q_proj"].flatten(1).contiguous(),
            generator,
        )
        residual = reference["input"]
        weight_norm = norm.norm.weight
        sweeps["rms_norm_hybrid (input_norm_variance)"] = row_sweep(
            lambda x: rms_norm_hybrid(x, x.float(), weight_norm, norm.eps), residual, generator
        )
    return sweeps


def row_classes(fn: Callable[[torch.Tensor], torch.Tensor], rows: torch.Tensor, max_rows: int, generator) -> list:
    """Row counts ``16..max_rows`` grouped by the bytes ``fn`` gives the same 16 leading rows of the call.

    Every call holds the same 16 probed rows first and the same filler rows after them, so only the row count
    changes. Returns runs ``[first_count, last_count, class]``; class 0 is the bytes at 16 rows, and two counts
    share a class exactly when the probed rows' outputs are byte-identical (the GEMM sums them in the same order).
    """
    filler = filler_rows(rows, max_rows, generator)
    filler[:DENSE_PROBE_ROWS] = rows[:DENSE_PROBE_ROWS]
    classes: dict[bytes, int] = {}
    runs: list[list[int]] = []
    for count in range(DENSE_PROBE_ROWS, max_rows + 1):
        output = fn(filler[:count])[:DENSE_PROBE_ROWS].contiguous()
        key = hashlib.sha256(output.view(torch.uint8).cpu().numpy().tobytes()).digest()
        label = classes.setdefault(key, len(classes))
        if runs and runs[-1][2] == label and runs[-1][1] == count - 1:
            runs[-1][1] = count
        else:
            runs.append([count, count, label])
    return runs


def dense_gemm_classes(layer, reference: Mapping[str, torch.Tensor], lm_head: torch.Tensor, max_rows: int, generator):
    """``row_classes`` of every projection the trainer (Transformer Engine) or compiled vLLM (``torch.mm``) runs."""
    attention, shared, router = layer.self_attention, layer.mlp.shared_experts, layer.mlp.router
    attention_in, mlp_in = reference["attention_norm"], reference["mlp_norm"]
    groups = attention.num_query_groups_per_partition
    head_dim = attention.hidden_size_per_attention_head
    query_width = attention.num_attention_heads_per_partition // groups * head_dim
    qkv = attention.linear_qkv.weight.view(groups, query_width + 2 * head_dim, -1)
    gate = attention.attn_gate.weight
    half = shared.linear_fc1.weight.shape[0] // 2

    def te(module):
        return lambda x: _first(module(x.unsqueeze(1))).squeeze(1)

    def mm(weight):
        weight = weight.contiguous()
        return lambda x: torch.mm(x, weight.t())

    functions = {
        "TE attn_gate (N=20)": (te(attention.attn_gate), attention_in),
        "torch.mm attn_gate (N=24 padded)": (
            mm(torch.cat((gate, gate.new_zeros(-gate.shape[0] % 8, gate.shape[1])))),
            attention_in,
        ),
        "TE qkv": (te(attention.linear_qkv), attention_in),
        "torch.mm q": (mm(qkv[:, :query_width].reshape(-1, qkv.shape[-1])), attention_in),
        "torch.mm k": (mm(qkv[:, query_width : query_width + head_dim].reshape(-1, qkv.shape[-1])), attention_in),
        "TE o_proj": (te(attention.linear_proj), reference["xsa_gate"]),
        "torch.mm o": (mm(attention.linear_proj.weight), reference["xsa_gate"]),
        "TE shared_fc1": (te(shared.linear_fc1), mlp_in),
        "torch.mm shared_gate": (mm(shared.linear_fc1.weight[:half]), mlp_in),
        "TE shared_fc2": (te(shared.linear_fc2), reference["shared_act"]),
        "torch.mm shared_down": (mm(shared.linear_fc2.weight), reference["shared_act"]),
        "router bf16->fp32 (default)": (lambda x: router.gating(x.unsqueeze(1)).squeeze(1), mlp_in),
        "torch.mm lm_head": (mm(lm_head), reference["next_attention_norm"]),
    }
    weight = router.weight.float()
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.no_grad():
            result = {name: row_classes(fn, rows, max_rows, generator) for name, (fn, rows) in functions.items()}
            result["router fp32 (router_gemm)"] = row_classes(
                lambda x: F.linear(x.float(), weight), mlp_in, max_rows, generator
            )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
    return result


def expert_sweep(layer, reference: Mapping[str, torch.Tensor], routes: torch.Tensor, numerics, generator) -> dict:
    """The probed tokens' routed-expert rows among other tokens' rows, as the expert row counts change."""
    experts = layer.mlp.experts
    tokens = reference["mlp_norm"]
    probs = reference["router_probs"]
    num_experts = probs.shape[-1]
    top_k = routes.shape[1]

    def run(extra: int) -> torch.Tensor:
        filler = filler_rows(tokens, extra, generator)
        all_tokens = torch.cat((tokens, filler))
        scores = torch.rand(extra, num_experts, generator=generator, device=tokens.device)
        filler_routes = scores.argsort(dim=1)[:, :top_k]
        all_routes = torch.cat((routes, filler_routes))
        routing_map = torch.zeros(all_tokens.shape[0], num_experts, dtype=torch.bool, device=tokens.device)
        routing_map.scatter_(1, all_routes, True)
        all_probs = torch.zeros(all_tokens.shape[0], num_experts, dtype=probs.dtype, device=tokens.device)
        all_probs[: tokens.shape[0]] = probs
        all_probs[tokens.shape[0] :] = torch.where(routing_map[tokens.shape[0] :], 0.5, 0.0).to(probs.dtype)
        pairs = routing_map.t().nonzero()
        permuted = all_tokens[pairs[:, 1]]
        permuted_probs = all_probs[pairs[:, 1], pairs[:, 0]]
        counts = routing_map.sum(0)
        with torch.no_grad(), grug_numerics(**numerics):
            output = _first(experts(permuted, counts.cpu(), permuted_probs))
        mine = pairs[:, 1] < tokens.shape[0]
        order = torch.argsort(pairs[mine, 1] * num_experts + pairs[mine, 0])
        return output[mine][order]

    alone = run(0)
    return {str(extra): equal_fraction(run(extra), alone) for extra in EXPERT_SWEEP_FILLER_TOKENS}


def attention_sweep(layer, reference: Mapping[str, torch.Tensor], numerics, generator) -> dict:
    """The probed sequence's core attention rows at other padded lengths, batch sizes and batch indices."""
    core = layer.self_attention.core_attention
    query, key, value = (reference[name] for name in ("query", "key", "value"))
    length = query.shape[0]

    def run(padding: int, batch: int, index: int) -> torch.Tensor:
        sequence = length + padding
        tensors = []
        for tensor in (query, key, value):
            laid_out = filler_rows(tensor, sequence * batch, generator).view(sequence, batch, *tensor.shape[1:])
            laid_out[:length, index] = tensor
            tensors.append(laid_out)
        with torch.no_grad(), grug_numerics(**numerics):
            output = core(*tensors, None, attn_mask_type=AttnMaskType.causal)
        return output[:length, index]

    alone = run(0, 1, 0)
    result = {}
    for padding in ATTENTION_SWEEP_LENGTHS:
        for batch in ATTENTION_SWEEP_BATCHES:
            for index in sorted({0, batch - 1}):
                result[f"S={length + padding},B={batch},b={index}"] = equal_fraction(run(padding, batch, index), alone)
    return result


def lm_head_layouts(
    final_hidden: torch.Tensor, lm_head: torch.Tensor, token_ids: torch.Tensor, generator: torch.Generator
) -> dict:
    """The LM head (Megatron's ``torch.matmul``) and the trainer's chunked log-softmax in every probe layout."""
    group = parallel_state.get_tensor_model_parallel_group()
    vocab = lm_head.shape[0]

    def run(layout: Layout) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = layout_input(final_hidden, layout, generator)
        logits = torch.matmul(hidden, lm_head.t()).transpose(0, 1).contiguous()
        targets = torch.randint(0, vocab, (layout.batch, layout.sequence), generator=generator, device=hidden.device)
        targets[layout.index, : layout.length] = token_ids[: layout.length]
        logprobs = from_parallel_logits_to_logprobs(
            logits, targets, 0, vocab, group, inference_only=True, cp_group=None, chunk_size=LOGPROB_CHUNK
        )
        return logits[layout.index, : layout.length], logprobs[layout.index, : layout.length - 1]

    with torch.no_grad():
        layouts = probe_layouts(final_hidden.shape[0])
        alone_logits, alone_logprobs = run(layouts[REFERENCE_LAYOUT])
        result = {}
        for name, layout in layouts.items():
            logits, logprobs = run(layout)
            result[name] = {
                "logits": compare(logits, alone_logits).to_json(),
                "logprobs": compare(logprobs.unsqueeze(-1), alone_logprobs.unsqueeze(-1)).to_json(),
                "logprob_max_abs": (logprobs - alone_logprobs).abs().max().item(),
            }
    return result


def render_markdown(results: Mapping) -> str:
    """One table per layer and numerics variant: isolated (and chained) byte-equal fraction per layout."""
    lines = ["# Trainer layout dependence per region", ""]
    for layer, entry in results["layers"].items():
        for variant, by_layout in entry["variants"].items():
            layouts = [name for name in by_layout if name != "alone_repeat"]
            lines += [
                f"## Layer {layer}, `{variant}`",
                "",
                "Byte-equal fraction of the probed sequence against the lone run: isolated / chained.",
                "",
                "| region | " + " | ".join(layouts) + " |",
                "|---|" + "---|" * len(layouts),
            ]
            regions = list(by_layout[layouts[0]]["isolated"]) if layouts else []
            for region in regions:
                cells = []
                for name in layouts:
                    isolated = by_layout[name]["isolated"].get(region, {})
                    chained = by_layout[name]["chained"].get(region, {})
                    cells.append(
                        f"{isolated.get('byte_equal_fraction', float('nan')):.4f} / "
                        f"{chained.get('byte_equal_fraction', float('nan')):.4f}"
                    )
                lines.append(f"| {region} | " + " | ".join(cells) + " |")
            lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--capture", action="append", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--input-from", action="append", default=[], help="LAYER=SOURCE: run LAYER on SOURCE's input")
    parser.add_argument("--numerics", action="append", default=[], help="[label=]comma-separated flags")
    parser.add_argument("--sweep-layers", type=int, nargs="*", default=[0, 3])
    parser.add_argument(
        "--dense-max-rows", type=int, default=0, help="also classify every GEMM row count up to this (0: off)"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    results: dict = {"arguments": vars(args), "gpu": torch.cuda.get_device_name(), "layers": {}, "lm_head": {}}
    with tempfile.TemporaryDirectory() as directory, single_rank_megatron(args.seed), vllm_config_context():
        config_dir = Path(directory)
        shape = GrugShape.from_config(stage_config(args.model, config_dir))
        captured, sequences, attention_mask = load_captures(args.capture)
        token_ids = sequences[0][attention_mask[0].bool()].cuda()
        _, provider = grug_provider(str(config_dir))
        names = {"lm_head.weight", "model.norm.weight"} | {
            f"model.final_gated_norm.{proj}.weight" for proj in ("down_proj", "up_proj")
        }
        for layer_index in args.layers:
            names |= set(layer_weight_names(layer_index, experts=True))
            if layer_index + 1 < shape.layers:
                names |= set(layer_weight_names(layer_index + 1, experts=False))
        weights = export_weights(args.model, config_dir, sorted(names))
        lm_head = weights["lm_head.weight"]
        rotary = rotary_embedding(provider).cuda()
        generator = torch.Generator(device="cuda").manual_seed(args.seed)
        variants = parse_variants(args.numerics)
        sources = parse_input_sources(args.input_from)
        for layer_index in args.layers:
            layer = build_layer(provider, layer_index)
            load_hf_weights(layer, f"decoder.layers.{layer_index}.", weights)
            if layer_index + 1 < shape.layers:
                next_norm = build_gated_norm(provider, NormRole.INPUT)
                load_hf_weights(next_norm, f"decoder.layers.{layer_index + 1}.input_layernorm.", weights)
            else:
                next_norm = build_gated_norm(provider, NormRole.FINAL)
                load_hf_weights(next_norm, "decoder.final_layernorm.", weights)
            ours = captured[sources.get(layer_index, layer_index)]["input"].cuda()[:, 0]
            entry: dict = {"length": ours.shape[0], "variants": {}, "sweeps": {}}
            for label, flags in variants.items():
                by_layout, reference = layer_layouts(layer, next_norm, rotary, ours, flags, generator)
                entry["variants"][label] = by_layout
                print(json.dumps({"layer": layer_index, "variant": label, "done": True}), flush=True)
                if layer_index + 1 == shape.layers:
                    results["lm_head"][label] = lm_head_layouts(
                        reference["next_attention_norm"], lm_head, token_ids, generator
                    )
                if layer_index in args.sweep_layers:
                    routes = reference["router_probs"].gt(0).nonzero()[:, 1].view(ours.shape[0], -1)
                    sweeps = {
                        "experts": expert_sweep(layer, reference, routes, flags, generator),
                        "attention": attention_sweep(layer, reference, flags, generator),
                    }
                    if not flags:
                        sweeps["gemm"] = gemm_sweeps(layer, reference, lm_head, generator)
                        if args.dense_max_rows:
                            sweeps["gemm_classes"] = dense_gemm_classes(
                                layer, reference, lm_head, args.dense_max_rows, generator
                            )
                    entry["sweeps"][label] = sweeps
            results["layers"][str(layer_index)] = entry
            del layer, next_norm
            gc.collect()
            torch.cuda.empty_cache()
    payload = json.dumps(results, indent=1, sort_keys=True, default=str)
    io.write_bytes_atomic(join_resource_path(args.output, "layout.json"), payload.encode())
    markdown = render_markdown(results)
    io.write_bytes_atomic(join_resource_path(args.output, "layout.md"), markdown.encode())
    print(markdown, flush=True)
    print("PASS mismatch harness layout", args.output, flush=True)


if __name__ == "__main__":
    main()
