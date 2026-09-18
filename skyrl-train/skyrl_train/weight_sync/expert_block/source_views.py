"""Views of live parameters on both ends of an expert-block sync; nothing is copied here.

On the trainer, Megatron-Bridge conversion tasks say which HF tensor range each
Megatron parameter backs. With TP=1 and Grug's stacked expert layout every expert
matrix is one whole parameter and every dense HF tensor is a few contiguous
runs, so a broadcast can read the parameter storage directly. On the receiver,
vLLM's fused expert parameters expose one contiguous ``[2I, H]`` or ``[H, I]``
slot per local expert, which is the broadcast's destination.
"""

from dataclasses import dataclass
import re

import torch

from skyrl_train.weight_sync.expert_block.schedule import BF16, DenseSlice, ExpertEntry, TrainerRank

EXPERT_HF_NAME = re.compile(r"model\.layers\.(\d+)\.mlp\.experts\.(gate|up|down)_proj\.weight")
ROUTED_EXPERTS = "mlp.experts.routed_experts"
ROUTER_WEIGHT_SUFFIX = ".mlp.router.weight"
EXPERT_MAPPINGS = ("GrugStackedExpertMapping", "GrugStackedGatedExpertMapping")


@dataclass(frozen=True)
class ExpertSource:
    """One local expert matrix and the trainer parameter that is that matrix."""

    entry: ExpertEntry
    source_key: str
    shape: tuple[int, int]


def local_source_slices(
    tasks, config, *, pp: int
) -> tuple[list[DenseSlice], list[DenseSlice], dict[str, torch.Tensor]]:
    """Split a rank's conversion tasks into expert slices, dense slices and the parameters behind them."""
    if config.tensor_model_parallel_size != 1:
        raise ValueError("Expert-block sync requires tensor-parallel size 1 on the trainer")
    expert, dense, sources = [], [], {}
    for task in tasks:
        source = task.param_weight
        if source is None:
            continue
        source = source.detach()
        key = task.global_param_name
        if key in sources or not source.is_contiguous() or str(source.dtype).removeprefix("torch.") not in ("bfloat16", "float32"):
            raise ValueError(f"Parameter {key} must be unique, contiguous and BF16 or FP32")
        sources[key] = source
        mapping = task.mapping
        kind = type(mapping).__name__
        dtype = str(source.dtype).removeprefix("torch.")
        expert_id = 0
        if kind in EXPERT_MAPPINGS:
            match = re.search(r"\.weight(\d+)$", key)
            if match is None or source.ndim != 2:
                raise ValueError(f"Expert parameter {key} is not a single per-expert matrix")
            expert_id = int(match.group(1))
        target = expert if kind in EXPERT_MAPPINGS else dense

        def add(name, hf_offset, source_offset, numel):
            if source_offset < 0 or numel <= 0 or source_offset + numel > source.numel():
                raise ValueError(f"Slice of {key} exceeds its storage")
            target.append(DenseSlice(name, hf_offset, numel, dtype, key, source_offset, pp))

        if kind in ("AutoMapping", "ReplicatedMapping", "GrugStackedExpertMapping"):
            add(mapping.hf_param, expert_id * source.numel(), 0, source.numel())
        elif kind in ("GatedMLPMapping", "GrugStackedGatedExpertMapping"):
            if source.ndim != 2 or source.shape[0] % 2 or set(mapping.hf_param) != {"gate", "up"}:
                raise ValueError(f"Gated parameter {key} is not a complete [gate;up] matrix")
            half = source.numel() // 2
            for position, part in enumerate(("gate", "up")):
                add(mapping.hf_param[part], expert_id * half, position * half, half)
        elif kind == "QKVMapping":
            heads, groups, head_dim = config.num_attention_heads, config.num_query_groups, config.kv_channels
            if source.shape != ((heads + 2 * groups) * head_dim, config.hidden_size) or set(mapping.hf_param) != {"q", "k", "v"}:
                raise ValueError(f"QKV parameter {key} differs from the configured interleaved layout")
            queries = heads // groups
            for group in range(groups):
                for part, offset, count in (("q", 0, queries), ("k", queries, 1), ("v", queries + 1, 1)):
                    numel = count * head_dim * config.hidden_size
                    add(mapping.hf_param[part], group * numel, (group * (queries + 2) + offset) * head_dim * config.hidden_size, numel)
        else:
            raise ValueError(f"Unsupported weight mapping for expert-block sync: {kind}")
    return expert, dense, sources


def local_expert_sources(
    expert_slices: list[DenseSlice],
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
    matrix_size = hidden_size * intermediate_size
    grouped: dict[tuple[int, int, str], dict[str, DenseSlice]] = {}
    for item in expert_slices:
        match = EXPERT_HF_NAME.fullmatch(item.hf_name)
        if match is None or item.wire_dtype != BF16 or item.numel != matrix_size or item.hf_offset % matrix_size:
            raise ValueError(f"Expert slice {item.hf_name}@{item.hf_offset} is not one BF16 expert matrix")
        layer, part = int(match[1]), match[2]
        expert = item.hf_offset // matrix_size
        if not trainer.ep * per_block <= expert < (trainer.ep + 1) * per_block:
            raise ValueError(f"Expert {expert} of layer {layer} is not owned by EP rank {trainer.ep}")
        parts = grouped.setdefault((layer, expert, "fc2" if part == "down" else "fc1"), {})
        if part in parts:
            raise ValueError(f"Duplicate expert slice {item.hf_name}@{item.hf_offset}")
        parts[part] = item
    result = []
    for (layer, expert, projection), parts in sorted(grouped.items()):
        needed = {"down"} if projection == "fc2" else {"gate", "up"}
        if set(parts) != needed or len({item.source_key for item in parts.values()}) != 1:
            raise ValueError(f"Expert {expert} of layer {layer} ({projection}) is incomplete or split across parameters")
        first = parts["down" if projection == "fc2" else "gate"]
        if first.source_offset != 0 or (projection == "fc1" and parts["up"].source_offset != matrix_size):
            raise ValueError(f"Expert {expert} of layer {layer} has reordered or nonadjacent gate/up halves")
        source = sources[first.source_key]
        shape = (hidden_size, intermediate_size) if projection == "fc2" else (2 * intermediate_size, hidden_size)
        if tuple(source.shape) != shape or source.dtype != torch.bfloat16:
            raise ValueError(f"Parameter {first.source_key} is not the {shape} BF16 matrix of expert {expert}")
        entry = ExpertEntry(f"model.layers.{layer}.mlp.experts.{projection}.expert{expert}", layer, trainer.pp, expert, projection, source.numel() * 2)
        result.append(ExpertSource(entry, first.source_key, shape))
    return result


def expert_source_view(item: ExpertSource, sources: dict[str, torch.Tensor]) -> torch.Tensor:
    source = sources[item.source_key]
    if tuple(source.shape) != item.shape or source.dtype != torch.bfloat16 or not source.is_contiguous():
        raise ValueError(f"Expert parameter {item.source_key} changed shape, dtype or layout")
    return source.view(-1)


def expert_destination_view(entry: ExpertEntry, parameters, expert_maps) -> torch.Tensor:
    """The receiver's live slot for a global expert, through the layer's global-to-local expert map."""
    prefix = f"model.layers.{entry.layer}.{ROUTED_EXPERTS}"
    local = expert_maps[prefix][entry.expert]
    if local < 0:
        raise ValueError(f"This receiver does not serve expert {entry.expert} of layer {entry.layer}")
    parameter = parameters[f"{prefix}.w13_weight" if entry.projection == "fc1" else f"{prefix}.w2_weight"]
    view = parameter[local]
    if view.dtype != torch.bfloat16 or not view.is_contiguous() or view.numel() * 2 != entry.nbytes:
        raise ValueError(f"Receiver slot for {entry.name} differs from the scheduled matrix")
    return view.view(-1)


def dense_source_view(item: DenseSlice, sources: dict[str, torch.Tensor]) -> torch.Tensor:
    source = sources[item.source_key]
    if str(source.dtype).removeprefix("torch.") != item.wire_dtype or not source.is_contiguous():
        raise ValueError(f"Parameter {item.source_key} changed dtype or layout")
    return source.view(-1).narrow(0, item.source_offset, item.numel)


def dense_destination_view(item: DenseSlice, parameters) -> torch.Tensor:
    """The installed range for a dense slice; the router weight is FP32 on the receiver and BF16 on the wire."""
    parameter = parameters[item.hf_name]
    if not parameter.is_contiguous():
        raise ValueError(f"Installed parameter {item.hf_name} is not contiguous")
    widened = item.hf_name.endswith(ROUTER_WEIGHT_SUFFIX) and item.wire_dtype == BF16 and parameter.dtype == torch.float32
    if not widened and str(parameter.dtype).removeprefix("torch.") != item.wire_dtype:
        raise ValueError(f"Installed dtype of {item.hf_name} differs from the wire dtype {item.wire_dtype}")
    return parameter.view(-1).narrow(0, item.hf_offset, item.numel)
