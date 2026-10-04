"""Describe Bridge conversion tasks for bounded expert-block publication.

This adapter owns Bridge mapping names, fused layouts and HF expert schemas.
It returns live source parameters and normalized publication records at TP=ETP=1;
the schedule and storage-view layers do not interpret Bridge tasks.
"""

import re

from skyrl_train.weight_sync.expert_block.schedule import DenseSlice, WIRE_DTYPE_BYTES
from skyrl_train.weight_sync.expert_block.source_views import ExpertSlice, LocalSources, dtype_name

EXPERT_HF_NAME = re.compile(r"model\.layers\.(\d+)\.mlp\.experts\.(gate|up|down)_proj\.weight")
SPLIT_EXPERT_HF_NAME = re.compile(r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight")
EXPERT_SOURCE_KEY = re.compile(r"decoder\.layers\.(\d+)\.mlp\.experts\.linear_fc[12]\.weight(\d+)")
EXPERT_MAPPINGS = ("GrugStackedExpertMapping", "GrugStackedGatedExpertMapping")
WIRE_DTYPES = frozenset(WIRE_DTYPE_BYTES)


def local_source_slices(tasks, config, *, pp: int) -> LocalSources:
    """Describe supported Bridge layouts as expert identities and dense storage runs.

    Accept whole, gated, QKV and Grug expert mappings at TP=ETP=1. Expert
    schemas are validated and reduced to layer/expert/part identities here.
    Dense runs preserve HF offsets; sources retain the original parameter
    objects so later ``param.data`` reassignment remains visible.
    """
    expert, dense, sources = [], [], {}
    for task in tasks:
        # Keep the parameter, not a detached view, so the sender's storage check sees a
        # reassigned ``param.data``.
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
        # For an expert mapping, tp_size is the expert-tensor-parallel size.
        if mapping.tp_size != 1:
            raise ValueError(f"Parameter {key} is tensor-parallel; expert-block sync requires trainer TP=1 and ETP=1")
        if kind in EXPERT_MAPPINGS:
            match = re.search(r"\.weight(\d+)$", key)
            if match is None or source.ndim != 2:
                raise ValueError(f"Expert parameter {key} is not a single per-expert matrix")
            expert_id = int(match.group(1))
            if kind == "GrugStackedExpertMapping":
                hf = EXPERT_HF_NAME.fullmatch(mapping.hf_param)
                if hf is None or hf[2] != "down":
                    raise ValueError(f"Expert slice {mapping.hf_param} is not a Grug down projection")
                expert.append(ExpertSlice(int(hf[1]), expert_id, "down", key))
            else:
                if source.shape[0] % 2 or set(mapping.hf_param) != {"gate", "up"}:
                    raise ValueError(f"Gated expert parameter {key} is not a complete [gate;up] matrix")
                for part in ("gate", "up"):
                    hf = EXPERT_HF_NAME.fullmatch(mapping.hf_param[part])
                    if hf is None or hf[2] != part:
                        raise ValueError(f"Expert slice {mapping.hf_param[part]} is not a Grug {part} projection")
                    expert.append(ExpertSlice(int(hf[1]), expert_id, part, key))
            continue

        # Split expert artifacts use one HF tensor per expert. Megatron-Bridge
        # maps them with generic GatedMLPMapping/AutoMapping rather than the
        # stacked-expert classes above. At TP=ETP=1, each is a whole matrix.
        if kind in ("GatedMLPMapping", "AutoMapping"):
            names = [mapping.hf_param] if kind == "AutoMapping" else mapping.hf_param.values()
            matches = [SPLIT_EXPERT_HF_NAME.fullmatch(name) for name in names]
            if any(match is not None for match in matches):
                if not all(match is not None for match in matches):
                    raise ValueError(f"Split expert mapping for {key} has mixed HF tensors")
                source_match = EXPERT_SOURCE_KEY.fullmatch(key)
                if source_match is None or source.ndim != 2:
                    raise ValueError(f"Split expert parameter {key} is not a single per-expert matrix")
                expert_id = int(source_match[2])
                layer = int(source_match[1])
                expected_parts = {"down"} if kind == "AutoMapping" else {"gate", "up"}
                if (
                    {match[3] for match in matches} != expected_parts
                    or any(int(match[1]) != layer or int(match[2]) != expert_id for match in matches)
                    or (kind == "AutoMapping" and ".linear_fc2." not in key)
                    or (kind == "GatedMLPMapping" and ".linear_fc1." not in key)
                ):
                    raise ValueError(f"Split expert mapping for {key} disagrees with its HF tensors")
                for part in sorted(expected_parts):
                    expert.append(ExpertSlice(layer, expert_id, part, key))
                continue

        def add(name, hf_offset, numel, source_offset):
            if source_offset < 0 or numel <= 0 or source_offset + numel > source.numel():
                raise ValueError(f"Slice of {key} exceeds its storage")
            dense.append(DenseSlice(name, hf_offset, numel, dtype, key, source_offset, pp))

        if kind in ("AutoMapping", "ReplicatedMapping", "RowParallelMapping"):
            # Hero's sconv_k uses a row-parallel mapping. With required TP=1,
            # its local parameter is the complete HF tensor.
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
