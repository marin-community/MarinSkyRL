"""Views of the live parameters an expert-block sync reads on the trainer and writes on the receiver.

On the trainer, Megatron-Bridge conversion tasks say which HF tensor each
Megatron parameter backs. With Grug's stacked expert layout at TP=1 every expert
matrix is one whole parameter and every dense HF tensor is one or more runs of
one parameter, so a broadcast reads parameter storage directly. On the receiver,
vLLM's fused expert parameters hold one contiguous ``[2I, H]`` or ``[H, I]``
slot per local expert and every dense Grug tensor is replicated, so a
destination is a run of a flat parameter.
"""

from dataclasses import dataclass
import re

import torch

from skyrl_train.weight_sync.expert_block.schedule import (
    BF16,
    FP32,
    WIRE_DTYPE_BYTES,
    DenseSlice,
    ExpertEntry,
    TrainerRank,
)

EXPERT_HF_NAME = re.compile(r"model\.layers\.(\d+)\.mlp\.experts\.(gate|up|down)_proj\.weight")
LAYER_PREFIX = re.compile(r"model\.layers\.(\d+)\.")
ROUTED_EXPERTS = "mlp.experts.routed_experts"
ROUTER_WEIGHT_SUFFIX = ".mlp.router.weight"
EXPERT_MAPPINGS = ("GrugStackedExpertMapping", "GrugStackedGatedExpertMapping")
WIRE_DTYPES = frozenset(WIRE_DTYPE_BYTES)


def dtype_name(dtype: torch.dtype) -> str:
    """``torch.bfloat16`` -> ``"bfloat16"``, the form the wire records and inventories carry."""
    return str(dtype).removeprefix("torch.")


def is_widened_router(hf_name: str, wire_dtype: str, installed_dtype: str) -> bool:
    """The router weight travels as BF16 and is installed into vLLM's FP32 parameter."""
    return hf_name.endswith(ROUTER_WEIGHT_SUFFIX) and wire_dtype == BF16 and installed_dtype == FP32


@dataclass(frozen=True)
class ExpertSlice:
    """One half (``gate``, ``up`` or ``down``) of one expert matrix as a trainer rank holds it."""

    hf_name: str
    expert: int
    part: str
    source_key: str


@dataclass(frozen=True)
class ExpertSource:
    """One local expert matrix and the trainer parameter that is that matrix."""

    entry: ExpertEntry
    source_key: str
    shape: tuple[int, int]


@dataclass(frozen=True)
class LocalSources:
    """What one trainer rank holds: its expert slices, its dense slices and the parameters behind them."""

    experts: list[ExpertSlice]
    dense: list[DenseSlice]
    sources: dict[str, torch.Tensor]


def local_source_slices(tasks, config, *, pp: int) -> LocalSources:
    """Split a rank's conversion tasks into expert slices, dense slices and the parameters behind them."""
    expert, dense, sources = [], [], {}
    for task in tasks:
        # Keep the parameter itself, not a detached view: a later ``param.data``
        # reassignment must show up when the sender re-checks storage before a send.
        source = task.param_weight
        if source is None:
            continue
        key = task.global_param_name
        dtype = dtype_name(source.dtype)
        if key in sources or not source.is_contiguous() or dtype not in WIRE_DTYPES:
            raise ValueError(f"Parameter {key} must be unique, contiguous and BF16 or FP32")
        sources[key] = source
        mapping = task.mapping
        kind = type(mapping).__name__
        # For an expert mapping this is the expert-tensor-parallel size.
        if mapping.tp_size != 1:
            raise ValueError(f"Parameter {key} is tensor-parallel; expert-block sync requires trainer TP=1 and ETP=1")
        if kind in EXPERT_MAPPINGS:
            match = re.search(r"\.weight(\d+)$", key)
            if match is None or source.ndim != 2:
                raise ValueError(f"Expert parameter {key} is not a single per-expert matrix")
            expert_id = int(match.group(1))
            if kind == "GrugStackedExpertMapping":
                expert.append(ExpertSlice(mapping.hf_param, expert_id, "down", key))
            else:
                if source.shape[0] % 2 or set(mapping.hf_param) != {"gate", "up"}:
                    raise ValueError(f"Gated expert parameter {key} is not a complete [gate;up] matrix")
                for part in ("gate", "up"):
                    expert.append(ExpertSlice(mapping.hf_param[part], expert_id, part, key))
            continue

        def add(name, hf_offset, numel, source_offset):
            if source_offset < 0 or numel <= 0 or source_offset + numel > source.numel():
                raise ValueError(f"Slice of {key} exceeds its storage")
            dense.append(DenseSlice(name, hf_offset, numel, dtype, key, source_offset, pp))

        if kind in ("AutoMapping", "ReplicatedMapping"):
            add(mapping.hf_param, 0, source.numel(), 0)
        elif kind == "GatedMLPMapping":
            if source.ndim != 2 or source.shape[0] % 2 or set(mapping.hf_param) != {"gate", "up"}:
                raise ValueError(f"Gated parameter {key} is not a complete [gate;up] matrix")
            half = source.numel() // 2
            for position, part in enumerate(("gate", "up")):
                add(mapping.hf_param[part], 0, half, position * half)
        elif kind == "QKVMapping":
            # Megatron interleaves [q..., k, v] per KV group; HF keeps q, k and v as separate tensors.
            heads, groups, head_dim = config.num_attention_heads, config.num_query_groups, config.kv_channels
            if heads % groups:
                raise ValueError(f"QKV parameter {key}: {heads} heads do not split across {groups} KV groups")
            queries = heads // groups
            expected_shape = (groups * (queries + 2) * head_dim, config.hidden_size)
            if tuple(source.shape) != expected_shape or set(mapping.hf_param) != {"q", "k", "v"}:
                raise ValueError(f"QKV parameter {key} differs from the configured interleaved layout")
            for group in range(groups):
                for part, offset, count in (("q", 0, queries), ("k", queries, 1), ("v", queries + 1, 1)):
                    numel = count * head_dim * config.hidden_size
                    source_offset = (group * (queries + 2) + offset) * head_dim * config.hidden_size
                    add(mapping.hf_param[part], group * numel, numel, source_offset)
        else:
            raise ValueError(f"Unsupported weight mapping for expert-block sync: {kind}")
    return LocalSources(expert, dense, sources)


def local_expert_sources(
    expert_slices: list[ExpertSlice],
    sources: dict[str, torch.Tensor],
    trainer: TrainerRank,
    *,
    num_experts: int,
    expert_parallel_size: int,
    hidden_size: int,
    intermediate_size: int,
) -> list[ExpertSource]:
    """Group a rank's expert slices into whole matrices and check each is one contiguous BF16 parameter."""
    per_block = num_experts // expert_parallel_size
    grouped: dict[tuple[int, int, str], dict[str, ExpertSlice]] = {}
    for item in expert_slices:
        match = EXPERT_HF_NAME.fullmatch(item.hf_name)
        if match is None or match[2] != item.part:
            raise ValueError(f"Expert slice {item.hf_name} is not a Grug expert projection")
        layer = int(match[1])
        if not trainer.ep * per_block <= item.expert < (trainer.ep + 1) * per_block:
            raise ValueError(f"Expert {item.expert} of layer {layer} is not owned by EP rank {trainer.ep}")
        parts = grouped.setdefault((layer, item.expert, "fc2" if item.part == "down" else "fc1"), {})
        if item.part in parts:
            raise ValueError(f"Duplicate expert slice {item.hf_name} for expert {item.expert}")
        parts[item.part] = item
    result = []
    for (layer, expert, projection), parts in sorted(grouped.items()):
        needed = {"down"} if projection == "fc2" else {"gate", "up"}
        if set(parts) != needed or len({item.source_key for item in parts.values()}) != 1:
            raise ValueError(
                f"Expert {expert} of layer {layer} ({projection}) is incomplete or split across parameters"
            )
        first = next(iter(parts.values()))
        source = sources[first.source_key]
        shape = (hidden_size, intermediate_size) if projection == "fc2" else (2 * intermediate_size, hidden_size)
        if tuple(source.shape) != shape or source.dtype != torch.bfloat16:
            raise ValueError(f"Parameter {first.source_key} is not the {shape} BF16 matrix of expert {expert}")
        entry = ExpertEntry(
            f"model.layers.{layer}.mlp.experts.{projection}.expert{expert}",
            layer,
            trainer.pp,
            expert,
            projection,
            source.numel() * WIRE_DTYPE_BYTES[BF16],
        )
        result.append(ExpertSource(entry, first.source_key, shape))
    return result


def expert_source_view(item: ExpertSource, sources: dict[str, torch.Tensor]) -> torch.Tensor:
    source = sources[item.source_key]
    if tuple(source.shape) != item.shape or source.dtype != torch.bfloat16 or not source.is_contiguous():
        raise ValueError(f"Expert parameter {item.source_key} changed shape, dtype or layout")
    return source.detach().view(-1)


def expert_slot_view(entry: ExpertEntry, parameters, expert_maps) -> torch.Tensor:
    """The receiver's whole live slot for a global expert, flat, through the layer's global-to-local expert map."""
    prefix = f"model.layers.{entry.layer}.{ROUTED_EXPERTS}"
    local = expert_maps[prefix][entry.expert]
    if local < 0:
        raise ValueError(f"This receiver does not serve expert {entry.expert} of layer {entry.layer}")
    parameter = parameters[f"{prefix}.w13_weight" if entry.projection == "fc1" else f"{prefix}.w2_weight"]
    slot = parameter.detach()[local]
    if (
        slot.dtype != torch.bfloat16
        or not slot.is_contiguous()
        or slot.numel() * WIRE_DTYPE_BYTES[BF16] != entry.nbytes
    ):
        raise ValueError(f"Receiver slot for {entry.name} differs from the scheduled matrix")
    return slot.view(-1)


def dense_source_view(item: DenseSlice, sources: dict[str, torch.Tensor]) -> torch.Tensor:
    source = sources[item.source_key]
    if dtype_name(source.dtype) != item.wire_dtype or not source.is_contiguous():
        raise ValueError(f"Parameter {item.source_key} changed dtype or layout")
    return source.detach().view(-1).narrow(0, item.source_offset, item.numel)


def dense_installed_view(item: DenseSlice, parameters) -> torch.Tensor:
    """The run of the installed tensor a slice lands in. The router weight is FP32 here, BF16 on the wire."""
    parameter = parameters[item.hf_name]
    if not parameter.is_contiguous():
        raise ValueError(f"Installed parameter {item.hf_name} is not contiguous")
    installed = dtype_name(parameter.dtype)
    if installed != item.wire_dtype and not is_widened_router(item.hf_name, item.wire_dtype, installed):
        raise ValueError(f"Installed dtype of {item.hf_name} differs from the wire dtype {item.wire_dtype}")
    if item.hf_offset + item.numel > parameter.numel():
        raise ValueError(f"Slice of {item.hf_name} exceeds the installed tensor")
    return parameter.detach().view(-1).narrow(0, item.hf_offset, item.numel)
