"""Allocation-free source slices from resolved TP1 Grug Bridge tasks.

Only views are constructed here. Native gathering must write these slices into
existing transfer buffers; no full HF expert stack or QKV index tensor is made.
"""

from dataclasses import dataclass
import re

import torch


@dataclass(frozen=True)
class FrozenSourceSlice:
    hf_name: str
    hf_offset: int
    numel: int
    wire_dtype: str
    source_key: str
    source_offset: int
    expert: bool


def local_source_slices(tasks, config):
    """Map only the explicit TP1, per-expert Grug layouts to original storage."""
    if config.tensor_model_parallel_size != 1 or getattr(config, "attention_output_gate", False):
        raise ValueError("Frozen source views require TP1 without interleaved attention output gates")
    slices = []
    sources = {}
    for task in tasks:
        if task is None:
            raise ValueError("Frozen source replay requires every resolved conversion task")
        source = task.param_weight
        if source is None:
            continue
        # Reuse storage across optimizer updates without retaining autograd view history.
        source = source.detach()
        key = task.global_param_name
        if key in sources or not source.is_contiguous() or source.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("Frozen source task must have unique contiguous BF16/FP32 storage")
        sources[key] = source
        mapping = task.mapping
        kind = type(mapping).__name__
        expert = kind in ("GrugStackedExpertMapping", "GrugStackedGatedExpertMapping")
        expert_id = 0
        if expert:
            match = re.search(r"\.weight(\d+)$", key)
            if match is None or source.ndim != 2:
                raise ValueError("Frozen expert replay requires a resolved global per-expert matrix")
            expert_id = int(match.group(1))
            if not 0 <= expert_id < config.num_moe_experts:
                raise ValueError("Global expert identity lies outside the configured model")

        def append(name, hf_offset, source_offset, numel):
            if source_offset < 0 or numel <= 0 or source_offset + numel > source.numel():
                raise ValueError("Frozen source slice exceeds original parameter storage")
            slices.append(
                FrozenSourceSlice(
                    name, hf_offset, numel, str(source.dtype).removeprefix("torch."), key, source_offset, expert
                )
            )

        if kind in ("AutoMapping", "ReplicatedMapping", "GrugStackedExpertMapping"):
            if not isinstance(mapping.hf_param, str):
                raise ValueError("Direct frozen mapping requires one HF tensor name")
            append(mapping.hf_param, expert_id * source.numel(), 0, source.numel())
        elif kind in ("GatedMLPMapping", "GrugStackedGatedExpertMapping"):
            if source.ndim != 2 or source.shape[0] % 2 or set(mapping.hf_param) != {"gate", "up"}:
                raise ValueError("Frozen gated mapping requires a complete [gate;up] matrix")
            half = source.numel() // 2
            for position, part in enumerate(("gate", "up")):
                append(mapping.hf_param[part], expert_id * half, position * half, half)
        elif kind == "QKVMapping":
            heads, groups, head_dim = config.num_attention_heads, config.num_query_groups, config.kv_channels
            if (
                source.ndim != 2
                or heads % groups
                or head_dim <= 0
                or source.shape != ((heads + 2 * groups) * head_dim, config.hidden_size)
                or set(mapping.hf_param) != {"q", "k", "v"}
            ):
                raise ValueError("Frozen QKV mapping differs from the configured interleaved matrix")
            queries = heads // groups
            for group in range(groups):
                for part, offset, count in (("q", 0, queries), ("k", queries, 1), ("v", queries + 1, 1)):
                    numel = count * head_dim * config.hidden_size
                    source_offset = (group * (queries + 2) + offset) * head_dim * config.hidden_size
                    append(mapping.hf_param[part], group * numel, source_offset, numel)
        else:
            raise ValueError(f"Unqualified frozen source mapping: {kind}")
    return tuple(slices), sources


def source_view(descriptor: FrozenSourceSlice, sources):
    source = sources[descriptor.source_key]
    if source.dtype != getattr(torch, descriptor.wire_dtype) or not source.is_contiguous():
        raise ValueError("Frozen source storage changed after planning")
    return source.view(-1).narrow(0, descriptor.source_offset, descriptor.numel)
