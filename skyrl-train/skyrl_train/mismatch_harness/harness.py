"""Compare one Grug decoder layer between the Megatron trainer and compiled vLLM, region by region.

Both sides start from the trainer's captured layer input. Every tensor compiled vLLM stores in the
layer (each fused-kernel output, each GEMM output, the attention output, the MoE output) is compared
with the trainer tensor that holds the same value, as byte-equal fraction and maximum ulp over the
valid token rows, in three runs:

- ``isolated``: every vLLM launch reads the trainer's tensors for all of its inputs, so each row
  measures one region's own difference;
- ``chained``: vLLM runs the whole layer from the trainer's layer input on its own values, so the
  first differing row in forward order is the lowest disagreeing region;
- ``floor``: the chained vLLM run repeated with other requests in the batch; the difference between
  the two vLLM runs on the same rows is vLLM's own spread for that region.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
from vllm import _custom_ops as vllm_ops

from skyrl_train.mismatch_harness.expert_parallel import ReduceOrder, reduce_partials
from skyrl_train.mismatch_harness.numerics import RegionStats, compare
from skyrl_train.mismatch_harness.output_code import LaunchKind
from skyrl_train.mismatch_harness.pieces import Piece, PieceKind
from skyrl_train.mismatch_harness.regions import (
    NEXT_NORM_REGIONS,
    POST_ATTENTION_ROLES,
    PRE_ATTENTION_ROLES,
    REGIONS,
)
from skyrl_train.mismatch_harness.replay import CallResult, run_call
from skyrl_train.mismatch_harness.vllm_side import (
    GrugShape,
    Requests,
    VllmMoe,
    fa3_num_splits,
    flash_attention_prefill,
    pad_rows,
)


def hf_parameter(role: str, prev_layer: int | None, next_layer: int | None) -> str:
    """The Hugging Face tensor a piece's parameter role holds."""
    if role.startswith("prev."):
        return f"model.layers.{prev_layer}.{role.removeprefix('prev.')}"
    if role.startswith("next."):
        return f"model.layers.{next_layer}.{role.removeprefix('next.')}"
    if role == "final_norm.weight":
        return "model.norm.weight"
    return f"model.{role}"


@dataclass(frozen=True)
class RowLayout:
    """Valid trainer rows after left-padding removal: sequence ``b`` holds positions ``[0, lengths[b])``."""

    lengths: tuple[int, ...]

    def flatten(self, tensor: torch.Tensor) -> torch.Tensor:
        """``[S, B, ...]`` to request-major ``[tokens, ...]``."""
        return torch.cat([tensor[:length, index] for index, length in enumerate(self.lengths)])

    def flatten_tokens(self, tensor: torch.Tensor, sequence_length: int) -> torch.Tensor:
        """Router tensors ``[S * B, E]`` or ``[S, B, E]`` (sequence-major) to request-major ``[tokens, E]``."""
        return self.flatten(tensor.reshape(sequence_length, len(self.lengths), tensor.shape[-1]))

    @property
    def requests(self) -> Requests:
        return Requests(self.lengths)


@dataclass
class VllmRun:
    """Region tensors from one vLLM run of a layer, request-major."""

    tensors: dict[str, torch.Tensor] = field(default_factory=dict)
    calls: dict[str, CallResult] = field(default_factory=dict)
    moe: VllmMoe | None = None


def trainer_region_tensors(
    tensors: Mapping[str, torch.Tensor],
    layout: RowLayout,
    shape: GrugShape,
    next_norm: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """The trainer's tensors under region names, request-major, in the shapes compiled vLLM stores."""
    sequence_length = tensors["input"].shape[0]
    flat = {
        name: layout.flatten(tensor)
        for name, tensor in tensors.items()
        if tensor.dim() >= 2 and tensor.shape[0] == sequence_length and tensor.shape[1] == len(layout.lengths)
    }
    tokens = layout.requests.tokens
    shared_gate, shared_up = flat["shared_fc1"].chunk(2, dim=-1)
    regions = {
        "input": flat["input"],
        "attn_rms": flat["attn_rms"],
        "attn_gate_down": flat["attn_gate_down"],
        "attn_gate_act": flat["attn_gate_act"],
        "attn_gate_up": flat["attn_gate_up"],
        "attention_norm": flat["attention_norm"],
        "q_proj": flat["q_proj"].reshape(tokens, -1),
        "k_proj": flat["k_proj"].reshape(tokens, -1),
        "v_proj": flat["value"].reshape(tokens, -1),
        "query": flat["query"],
        "key": flat["key"],
        "core_attention": flat["core_attention"].reshape(tokens, shape.heads, shape.head_dim),
        "attn_gate": flat["attn_gate"],
        "xsa_gate": flat["xsa_gate"].reshape(tokens, shape.heads, shape.head_dim),
        "residual_after_attention": flat["residual_after_attention"],
        "mlp_rms": flat["mlp_rms"],
        "mlp_gate_down": flat["mlp_gate_down"],
        "mlp_gate_act": flat["mlp_gate_act"],
        "mlp_gate_up": flat["mlp_gate_up"],
        "mlp_norm": flat["mlp_norm"],
        "router_input": flat["mlp_norm"].float(),
        "router_logits": layout.flatten_tokens(tensors["router_logits"], sequence_length),
        "routed": flat["routed"],
        "shared_gate": shared_gate.contiguous(),
        "shared_up": shared_up.contiguous(),
        "shared_act": flat["shared_act"],
        "shared_down": flat["shared_expert"],
        "output": flat["output"],
    }
    for name, region in NEXT_NORM_REGIONS.items():
        regions[region] = layout.flatten(next_norm[name])
    return regions


def trainer_routing(
    tensors: Mapping[str, torch.Tensor], layout: RowLayout, bias: torch.Tensor, top_k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The trainer's selected experts and combine weights in vLLM's slot order (biased score, descending)."""
    sequence_length = tensors["input"].shape[0]
    routing_map = layout.flatten_tokens(tensors["router_map"], sequence_length)
    probs = layout.flatten_tokens(tensors["router_probs"], sequence_length)
    logits = layout.flatten_tokens(tensors["router_logits"], sequence_length)
    score = torch.where(routing_map, logits + bias, torch.full_like(logits, float("-inf")))
    order = torch.sort(score, dim=-1, descending=True, stable=True).indices[:, :top_k]
    return order.to(torch.int32), probs.gather(1, order)


def trainer_expert_slots(
    tensors: Mapping[str, torch.Tensor], layout: RowLayout, topk_ids: torch.Tensor
) -> torch.Tensor:
    """The trainer's per-slot expert outputs ``[tokens, top_k, hidden]`` from its permuted fc2 output.

    TE's permutation orders rows by expert, then by token in sequence-major order.
    """
    sequence_length = tensors["input"].shape[0]
    routing_map = tensors["router_map"]
    batch = len(layout.lengths)
    token_order = torch.arange(sequence_length * batch, device=routing_map.device).reshape(sequence_length, batch)
    flat_index = layout.flatten(token_order)
    rows = tensors["experts_fc2"]
    permuted = routing_map.t().nonzero()
    row_of = {(int(token), int(expert)): row for row, (expert, token) in enumerate(permuted.tolist())}
    slots = torch.empty(topk_ids.shape[0], topk_ids.shape[1], rows.shape[-1], dtype=rows.dtype, device=rows.device)
    for position, token in enumerate(flat_index.tolist()):
        for slot, expert in enumerate(topk_ids[position].tolist()):
            slots[position, slot] = rows[row_of[(token, expert)]]
    return slots


def emulated_ep_sum(
    slots: torch.Tensor, topk_ids: torch.Tensor, experts: int, ep_size: int, order: ReduceOrder, home_rank: int
) -> torch.Tensor:
    """vLLM's combine applied to given per-slot outputs: per-rank ``moe_sum`` in fp32, then a bf16 reduction."""
    per_rank = experts // ep_size
    partials = []
    for rank in range(ep_size):
        local = ((topk_ids // per_rank) == rank).unsqueeze(-1)
        output = torch.empty(slots.shape[0], slots.shape[2], dtype=slots.dtype, device=slots.device)
        vllm_ops.moe_sum(torch.where(local, slots, torch.zeros_like(slots)).contiguous(), output)
        partials.append(output)
    return reduce_partials(torch.stack(partials), order, home_rank)


class LayerReplay:
    """Compiled vLLM's layer ``L`` from the archived pieces, with the trainer's tensors where asked."""

    def __init__(
        self,
        *,
        pieces: Mapping[tuple[PieceKind, bool], tuple[Piece, dict[str, Any]]],
        shape: GrugShape,
        weights: Mapping[str, torch.Tensor],
        cos_sin: torch.Tensor,
        layer: int,
        router_bias: torch.Tensor,
        ep_size: int,
        home_rank: int,
        order: ReduceOrder,
    ):
        self.pieces = pieces
        self.shape = shape
        self.weights = weights
        self.cos_sin = cos_sin
        self.layer = layer
        self.router_bias = router_bias
        self.ep_size = ep_size
        self.home_rank = home_rank
        self.order = order

    def _piece(self, kind: PieceKind, rope: bool) -> tuple[Piece, dict[str, Any]]:
        key = (kind, rope if kind is not PieceKind.LAST else False)
        if key not in self.pieces:
            raise KeyError(f"no archived output code for a {kind} piece with rope={rope}")
        return self.pieces[key]

    def pre_piece(self) -> tuple[Piece, dict[str, Any]]:
        if self.layer == 0:
            return self._piece(PieceKind.FIRST, True)
        return self._piece(PieceKind.MIDDLE, not self.shape.is_long(self.layer))

    def post_piece(self) -> tuple[Piece, dict[str, Any]]:
        if self.layer == self.shape.layers - 1:
            return self._piece(PieceKind.LAST, False)
        return self._piece(PieceKind.MIDDLE, not self.shape.is_long(self.layer + 1))

    def arguments(
        self, piece: Piece, tokens: int, prev_layer: int | None, next_layer: int | None, activations: Mapping[str, Any]
    ) -> list[Any]:
        by_argument = {argument: role for role, argument in piece.parameters.items()}
        by_activation = {argument: role for role, argument in piece.activations.items()}
        values = []
        for name in piece.code.arguments:
            example = piece.examples[name]
            if name in by_argument:
                role = by_argument[name]
                hf_name = hf_parameter(role, prev_layer, next_layer)
                tensor = self.weights.get(hf_name)
                if tensor is None:
                    tensor = torch.zeros(example.shape, dtype=getattr(torch, example.dtype), device="cuda")
                elif role.endswith("mlp.router.weight"):
                    tensor = tensor.float()
                values.append(_pad_vocabulary(tensor, example.shape).contiguous())
            elif name in by_activation:
                role = by_activation[name]
                if role in activations:
                    values.append(activations[role])
                elif example.shape is None:
                    values.append(None)
                else:
                    values.append(_zeros_like_example(example, tokens))
            elif example.shape is None and example.dtype is None:
                values.append(tokens)
            else:
                raise KeyError(f"argument {name} of the {piece.kind} piece has no role")
        return values

    def run_pre_attention(
        self,
        requests: Requests,
        layer_input: torch.Tensor,
        token_ids: torch.Tensor | None,
        substitutions: Mapping[str, torch.Tensor] | None,
        *,
        embedding_path: bool = False,
    ) -> tuple[CallResult, Piece]:
        """Layer ``L`` up to attention. ``embedding_path`` runs layer 0's piece from token ids unmodified."""
        piece, namespace = self.pre_piece()
        tokens = requests.rows
        activations: dict[str, Any] = {"prev.layer_name": None}
        if token_ids is not None:
            activations["token_ids"] = pad_rows(token_ids.to(torch.int32), tokens)
        if piece.rope:
            activations["next.positions"] = requests.positions()
            activations["next.cos_sin_cache"] = self.cos_sin
        arguments = self.arguments(piece, tokens, None, self.layer, activations)
        before: dict[int, list[tuple[str, torch.Tensor]]] = {}
        residual = piece.values["next.residual"]
        launch = next(launch for launch in piece.code.launches if launch.statement == residual.statement)
        if not embedding_path:
            if piece.kind is PieceKind.FIRST:
                # x * sigmoid(+inf) is x exactly, so layer 0's fused gate-and-norm kernel normalizes the
                # trainer's layer input as its stored residual.
                gated, gate = launch.reads[0], launch.reads[1]
                before[launch.statement] = [
                    (gated, layer_input),
                    (gate, torch.full_like(layer_input, float("inf"))),
                ]
            else:
                # h + (0 + 0) is h exactly in fp32: the fused residual-add-and-norm kernel stores and
                # normalizes the trainer's layer input.
                residual_variable, routed, shared = launch.reads[0], launch.reads[1], launch.reads[2]
                zeros = torch.zeros_like(layer_input)
                before[launch.statement] = [(residual_variable, layer_input), (routed, zeros), (shared, zeros)]
        _add_substitutions(before, piece, PRE_ATTENTION_ROLES, substitutions)
        result = run_call(
            piece.code,
            namespace,
            arguments,
            ops={"vllm.moe_forward.default": _zero_moe},
            before=_apply_before(before),
        )
        return result, piece

    def run_post_attention(
        self,
        requests: Requests,
        attention: torch.Tensor,
        value: torch.Tensor,
        attn_in: torch.Tensor,
        residual: torch.Tensor,
        moe: VllmMoe,
        substitutions: Mapping[str, torch.Tensor] | None,
    ) -> tuple[CallResult, Piece]:
        piece, namespace = self.post_piece()
        tokens = requests.rows
        activations: dict[str, Any] = {
            "prev.attn_out": pad_rows(attention, tokens).contiguous(),
            "prev.value": pad_rows(value, tokens).contiguous(),
            "prev.attn_in": pad_rows(attn_in, tokens).contiguous(),
            "prev.residual": pad_rows(residual, tokens).contiguous(),
            "prev.layer_name": None,
        }
        if piece.rope:
            activations["next.positions"] = requests.positions()
            activations["next.cos_sin_cache"] = self.cos_sin
        next_layer = self.layer + 1 if piece.kind is not PieceKind.LAST else None
        arguments = self.arguments(piece, tokens, self.layer, next_layer, activations)
        before: dict[int, list[tuple[str, torch.Tensor]]] = {}
        _add_substitutions(before, piece, POST_ATTENTION_ROLES, substitutions)
        result = run_call(
            piece.code, namespace, arguments, ops={"vllm.moe_forward.default": moe}, before=_apply_before(before)
        )
        return result, piece

    def attention(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, requests: Requests
    ) -> torch.Tensor:
        window = None if self.shape.is_long(self.layer) else self.shape.sliding_window
        tokens = requests.tokens
        output = flash_attention_prefill(
            query[:tokens].contiguous(),
            key[:tokens].contiguous(),
            value[:tokens].contiguous(),
            requests,
            window=window,
            scale=self.shape.head_dim**-0.5,
            num_splits=fa3_num_splits(tokens),
        )
        return pad_rows(output, requests.rows)

    def moe(self, config_tokens: int | None, routing: tuple[torch.Tensor, torch.Tensor] | None) -> VllmMoe:
        prefix = f"model.layers.{self.layer}.mlp.experts."
        w13 = torch.cat((self.weights[prefix + "gate_proj.weight"], self.weights[prefix + "up_proj.weight"]), dim=1)
        return VllmMoe(
            bias=self.router_bias,
            w13=w13.contiguous(),
            w2=self.weights[prefix + "down_proj.weight"].contiguous(),
            top_k=self.shape.top_k,
            ep_size=self.ep_size,
            home_rank=self.home_rank,
            order=self.order,
            config_tokens=config_tokens,
            routing=routing,
        )


def _pad_vocabulary(tensor: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """vLLM pads the vocabulary of its embedding table to a multiple of 64 rows with zeros."""
    if tuple(tensor.shape) == tuple(shape):
        return tensor
    if tuple(tensor.shape[1:]) != tuple(shape[1:]) or tensor.shape[0] > shape[0]:
        raise ValueError(f"weight of shape {tuple(tensor.shape)} does not fit the compiled {tuple(shape)}")
    padded = torch.zeros(shape, dtype=tensor.dtype, device=tensor.device)
    padded[: tensor.shape[0]] = tensor
    return padded


def _zeros_like_example(example, tokens: int) -> torch.Tensor:
    shape = [tokens if index == 0 else size for index, size in enumerate(example.shape)]
    return torch.zeros(shape, dtype=getattr(torch, example.dtype), device="cuda")


def _zero_moe(hidden_states, *args):
    return torch.zeros_like(hidden_states)


def _add_substitutions(
    before: dict[int, list[tuple[str, torch.Tensor]]],
    piece: Piece,
    roles: Mapping[str, str],
    substitutions: Mapping[str, torch.Tensor] | None,
) -> None:
    """After each launch that stores a value with a trainer counterpart, overwrite it with the trainer's."""
    if substitutions is None:
        return
    for role, ref in piece.values.items():
        region = roles.get(role)
        if region is None or region not in substitutions:
            continue
        before.setdefault(ref.statement + 1, []).append((ref.variable, substitutions[region]))


def _apply_before(before: Mapping[int, list[tuple[str, torch.Tensor]]]) -> Callable[[int, dict[str, Any]], None]:
    def apply(index: int, environment: dict[str, Any]) -> None:
        for variable, tensor in before.get(index, ()):
            target = environment[variable]
            if target.shape[0] > tensor.shape[0]:
                # Buffers hold the step's padded rows; the given tensor fills the real ones.
                target = target[: tensor.shape[0]]
            if target.numel() != tensor.numel() and target.dim() == 2 and tensor.dim() == 2:
                # A GEMM whose weight Inductor padded (attn_gate: 24 rows for 20 heads) stores extra
                # columns; the trainer's tensor fills the leading ones.
                target = target[:, : tensor.shape[1]]
            if target.numel() != tensor.numel():
                raise ValueError(f"cannot write {tuple(tensor.shape)} into {variable} {tuple(target.shape)}")
            target.copy_(tensor.reshape(target.shape).to(target.dtype))

    return apply


def piece_regions(result: CallResult, piece: Piece, roles: Mapping[str, str]) -> dict[str, torch.Tensor]:
    """Stored values of one piece run under region names."""
    stored = {(value.launch.statement, value.variable): value.tensor for value in result.values}
    return {roles[role]: stored[(ref.statement, ref.variable)] for role, ref in piece.values.items() if role in roles}


def attention_inputs(result: CallResult, shape: GrugShape) -> dict[str, torch.Tensor]:
    """The query, key and value a pre-attention piece hands to the attention op.

    ``call()`` returns them with the attention output buffer. The key and value have ``kv_heads``
    heads; the value is the value projection's output. The query and the output buffer have ``heads``
    heads; the query was last written by a fused kernel, the output buffer (reused storage) was not.
    """
    last_writer = {}
    for value in result.values:
        last_writer[value.data_ptr] = value.launch
    outputs = [output for output in result.outputs if isinstance(output, torch.Tensor) and output.dim() == 3]
    selected: dict[str, torch.Tensor] = {}
    if shape.heads == shape.kv_heads:
        raise ValueError("attention inputs are told apart by head count; Grug uses grouped-query attention")
    for output in outputs:
        writer = last_writer.get(output.data_ptr())
        written_by_kernel = writer is not None and writer.kind is LaunchKind.TRITON
        if output.shape[1] == shape.heads:
            role = "query" if written_by_kernel else "output_buffer"
        elif output.shape[1] == shape.kv_heads:
            role = "key" if written_by_kernel else "value"
        else:
            raise ValueError(f"piece output of shape {tuple(output.shape)} is not an attention input")
        if role in selected:
            raise ValueError(f"two attention inputs look like the {role}")
        selected[role] = output
    missing = {"query", "key", "value"} - set(selected)
    if missing:
        raise ValueError(f"cannot identify attention inputs {sorted(missing)} among the piece outputs")
    return selected


def compare_regions(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> dict[str, RegionStats]:
    """Compare every region present on both sides, in forward order."""
    stats: dict[str, RegionStats] = {}
    for region in REGIONS:
        if region in left and region in right:
            a, b = left[region], right[region]
            if a.dtype != b.dtype:
                b = b.to(a.dtype)
            stats[region] = compare(a.reshape(a.shape[0], -1), b.reshape(b.shape[0], -1))
    return stats


def routing_agreement(
    vllm_ids: torch.Tensor, vllm_weights: torch.Tensor, trainer_ids: torch.Tensor, trainer_weights: torch.Tensor
) -> dict[str, float]:
    """Expert-set agreement per token and byte equality of combine weights on tokens with equal sets."""
    same_set = (torch.sort(vllm_ids, dim=-1).values == torch.sort(trainer_ids.to(vllm_ids.dtype), dim=-1).values).all(
        -1
    )
    by_expert_vllm = torch.sort(vllm_ids, dim=-1)
    by_expert_trainer = torch.sort(trainer_ids.to(vllm_ids.dtype), dim=-1)
    weights_vllm = vllm_weights.gather(1, by_expert_vllm.indices)
    weights_trainer = trainer_weights.gather(1, by_expert_trainer.indices)
    rows = same_set.nonzero().flatten()
    weight_stats = compare(weights_vllm[rows].float(), weights_trainer[rows].float()) if rows.numel() else None
    return {
        "expert_set_agreement": same_set.float().mean().item(),
        "weights_byte_equal_fraction": weight_stats.byte_equal_fraction if weight_stats else float("nan"),
        "weights_max_ulp": weight_stats.max_ulp if weight_stats else -1,
    }
