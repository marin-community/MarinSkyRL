"""Views of live parameters on both ends of an expert-block sync; nothing is copied here.

On the trainer, Megatron-Bridge conversion tasks say which HF tensor each
Megatron parameter backs and, through the mapping's tensor-parallel rank, which
part of it. With Grug's stacked expert layout every expert matrix (or its
expert-tensor-parallel shard) is one whole parameter, and every dense HF tensor
is a few runs of one parameter, so a broadcast can read the parameter storage
directly. On the receiver, vLLM's fused expert parameters expose one contiguous
``[2I, H]`` or ``[H, I]`` slot per local expert, and every dense Grug layer is
replicated, so the destination is a region of a flat parameter.

Trainer tensor parallelism (the section marked below) turns a column-parallel
shard into one run of the HF tensor and a row-parallel shard into a column
block; the schedule carries either as a :class:`Region`.
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
    Region,
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
    """One half (``gate``, ``up`` or ``down``) of one expert matrix shard as a trainer rank holds it."""

    hf_name: str
    expert: int
    part: str
    shard: int
    shards: int
    source_key: str
    source_offset: int
    numel: int


@dataclass(frozen=True)
class ExpertSource:
    """One local expert matrix shard and the trainer parameter that is that shard."""

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
        # --- Trainer tensor parallelism: which part of the HF tensor this rank's shard is ---
        rank, size = mapping.tp_rank, mapping.tp_size
        if kind in EXPERT_MAPPINGS:
            match = re.search(r"\.weight(\d+)$", key)
            if match is None or source.ndim != 2:
                raise ValueError(f"Expert parameter {key} is not a single per-expert matrix")
            expert_id = int(match.group(1))
            if kind == "GrugStackedExpertMapping":
                expert.append(ExpertSlice(mapping.hf_param, expert_id, "down", rank, size, key, 0, source.numel()))
            else:
                if source.shape[0] % 2 or set(mapping.hf_param) != {"gate", "up"}:
                    raise ValueError(f"Gated expert parameter {key} is not a complete [gate;up] matrix")
                half = source.numel() // 2
                for position, part in enumerate(("gate", "up")):
                    expert.append(
                        ExpertSlice(mapping.hf_param[part], expert_id, part, rank, size, key, position * half, half)
                    )
            continue

        def add(name, region: Region, source_offset):
            if source_offset < 0 or region.numel <= 0 or source_offset + region.numel > source.numel():
                raise ValueError(f"Slice of {key} exceeds its storage")
            dense.append(
                DenseSlice(name, region.offset, region.numel, dtype, key, source_offset, pp, region.runs, region.stride)
            )

        if kind == "AutoMapping":
            layout = mapping._detect_parallelism_type(task.megatron_module)
            add(mapping.hf_param, shard_region(layout, tuple(source.shape), rank, size), 0)
        elif kind == "ReplicatedMapping":
            add(mapping.hf_param, Region(0, source.numel()), 0)
        elif kind == "GatedMLPMapping":
            if source.ndim != 2 or source.shape[0] % 2 or set(mapping.hf_param) != {"gate", "up"}:
                raise ValueError(f"Gated parameter {key} is not a complete [gate;up] matrix")
            half = source.numel() // 2
            for position, part in enumerate(("gate", "up")):
                # Each half is a column-parallel shard of its own HF tensor.
                add(mapping.hf_param[part], Region(rank * half, half), position * half)
        elif kind == "QKVMapping":
            heads, groups, head_dim = config.num_attention_heads, config.num_query_groups, config.kv_channels
            if groups % size or heads % groups:
                raise ValueError(f"QKV parameter {key}: {groups} KV groups do not split across TP {size}")
            local_groups = groups // size
            queries = heads // groups
            expected_shape = (local_groups * (queries + 2) * head_dim, config.hidden_size)
            if tuple(source.shape) != expected_shape or set(mapping.hf_param) != {"q", "k", "v"}:
                raise ValueError(f"QKV parameter {key} differs from the configured interleaved layout")
            for local in range(local_groups):
                group = rank * local_groups + local
                for part, offset, count in (("q", 0, queries), ("k", queries, 1), ("v", queries + 1, 1)):
                    numel = count * head_dim * config.hidden_size
                    source_offset = (local * (queries + 2) + offset) * head_dim * config.hidden_size
                    add(mapping.hf_param[part], Region(group * numel, numel), source_offset)
        else:
            raise ValueError(f"Unsupported weight mapping for expert-block sync: {kind}")
    return LocalSources(expert, dense, sources)


def shard_region(layout: str, shape: tuple[int, ...], rank: int, size: int) -> Region:
    """The HF region of a rank's shard: column shards are one run, row shards a column block."""
    numel = 1
    for dimension in shape:
        numel *= dimension
    if layout == "replicated" or size == 1:
        return Region(0, numel)
    if layout == "column":
        return Region(rank * numel, numel)
    if layout == "row":
        if len(shape) != 2:
            raise ValueError(f"Row-parallel shard of shape {shape} is not a matrix")
        rows, columns = shape
        return Region(rank * columns, numel, rows, columns * size)
    raise ValueError(f"Unsupported tensor-parallel layout {layout}")


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
    """Group a rank's expert slices into whole matrices (or shards) and check each is one contiguous BF16 parameter."""
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
        first = parts["down" if projection == "fc2" else "gate"]
        shard, shards = first.shard, first.shards
        if intermediate_size % shards or any((item.shard, item.shards) != (shard, shards) for item in parts.values()):
            raise ValueError(f"Expert {expert} of layer {layer} has inconsistent tensor-parallel shards")
        piece = intermediate_size // shards
        if projection == "fc1" and (first.source_offset != 0 or parts["up"].source_offset != piece * hidden_size):
            raise ValueError(f"Expert {expert} of layer {layer} has reordered or nonadjacent gate/up halves")
        source = sources[first.source_key]
        shape = (hidden_size, piece) if projection == "fc2" else (2 * piece, hidden_size)
        if tuple(source.shape) != shape or source.dtype != torch.bfloat16:
            raise ValueError(f"Parameter {first.source_key} is not the {shape} BF16 matrix of expert {expert}")
        suffix = f".shard{shard}" if shards > 1 else ""
        entry = ExpertEntry(
            f"model.layers.{layer}.mlp.experts.{projection}.expert{expert}{suffix}",
            layer,
            trainer.pp,
            expert,
            projection,
            source.numel() * WIRE_DTYPE_BYTES[BF16],
            shard,
            shards,
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
        or slot.numel() * WIRE_DTYPE_BYTES[BF16] != entry.nbytes * entry.shards
    ):
        raise ValueError(f"Receiver slot for {entry.name} differs from the scheduled matrix")
    return slot.view(-1)


def expert_slot_region(entry: ExpertEntry, slot: torch.Tensor, hidden_size: int) -> Region:
    """Where an expert shard lands in the whole slot: rows of both halves for ``fc1``, a column block for ``fc2``."""
    if entry.shards == 1:
        return Region(0, slot.numel())
    if entry.projection == "fc1":
        intermediate = slot.numel() // (2 * hidden_size)
        piece = intermediate // entry.shards
        return Region(entry.shard * piece * hidden_size, 2 * piece * hidden_size, 2, intermediate * hidden_size)
    intermediate = slot.numel() // hidden_size
    piece = intermediate // entry.shards
    return Region(entry.shard * piece, hidden_size * piece, hidden_size, intermediate)


def dense_source_view(item: DenseSlice, sources: dict[str, torch.Tensor]) -> torch.Tensor:
    source = sources[item.source_key]
    if dtype_name(source.dtype) != item.wire_dtype or not source.is_contiguous():
        raise ValueError(f"Parameter {item.source_key} changed dtype or layout")
    return source.detach().view(-1).narrow(0, item.source_offset, item.numel)


def dense_flat_view(item: DenseSlice, parameters) -> torch.Tensor:
    """The receiver's whole installed tensor, flat; the router weight is FP32 here and BF16 on the wire."""
    parameter = parameters[item.hf_name]
    if not parameter.is_contiguous():
        raise ValueError(f"Installed parameter {item.hf_name} is not contiguous")
    installed = dtype_name(parameter.dtype)
    if installed != item.wire_dtype and not is_widened_router(item.hf_name, item.wire_dtype, installed):
        raise ValueError(f"Installed dtype of {item.hf_name} differs from the wire dtype {item.wire_dtype}")
    return parameter.detach().view(-1)


def region_view(flat: torch.Tensor, region: Region) -> torch.Tensor:
    """The region of a flat tensor: a narrow for one run, a strided ``[runs, run_length]`` view otherwise."""
    if region.offset + (region.runs - 1) * region.stride + region.run_length > flat.numel():
        raise ValueError("Region exceeds the installed tensor")
    if region.direct:
        return flat.narrow(0, region.offset, region.numel)
    return flat.as_strided((region.runs, region.run_length), (region.stride, 1), flat.storage_offset() + region.offset)
