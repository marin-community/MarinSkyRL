"""A tiny Grug model in Megatron's parameter layout, and the HF tensors it must convert to.

Every element is unique to its tensor and position, so a swapped ``[gate;up]`` half, a
transposed matrix, a wrong expert slot or a misplaced QKV run changes the bytes.
:func:`reference_hf` converts the layout with plain tensor operations and shares no code with
the transport's slicing.
"""

import math
import re
from types import SimpleNamespace

import torch

NUM_EXPERTS = 4
HIDDEN = 3
INTERMEDIATE = 2
SHARED_INTERMEDIATE = 2
HEADS, KV_GROUPS, HEAD_DIM = 4, 2, 2
QUERIES_PER_GROUP = HEADS // KV_GROUPS
VOCAB = 5
# The provider fields the transport reads to split the interleaved QKV parameter.
PROVIDER = SimpleNamespace(
    num_attention_heads=HEADS, num_query_groups=KV_GROUPS, kv_channels=HEAD_DIM, hidden_size=HIDDEN
)
# BF16 keeps eight significant bits: 128 + position is exact below this many elements.
MAX_NUMEL = 128


def megatron_shapes(layers, experts, *, last_stage: bool) -> dict[str, tuple[int, ...]]:
    """Megatron parameter name -> shape for a rank holding ``layers`` and the global ``experts``."""
    shapes = {}
    for layer in layers:
        prefix = f"decoder.layers.{layer}"
        shapes[f"{prefix}.self_attention.linear_qkv.weight"] = (
            KV_GROUPS * (QUERIES_PER_GROUP + 2) * HEAD_DIM,
            HIDDEN,
        )
        shapes[f"{prefix}.mlp.router.weight"] = (NUM_EXPERTS, HIDDEN)
        shapes[f"{prefix}.mlp.shared_experts.linear_fc1.weight"] = (2 * SHARED_INTERMEDIATE, HIDDEN)
        for expert in experts:
            shapes[f"{prefix}.mlp.experts.linear_fc1.weight{expert}"] = (2 * INTERMEDIATE, HIDDEN)
            shapes[f"{prefix}.mlp.experts.linear_fc2.weight{expert}"] = (HIDDEN, INTERMEDIATE)
    if last_stage:
        shapes["decoder.final_layernorm.weight"] = (HIDDEN,)
        shapes["output_layer.weight"] = (VOCAB, HIDDEN)
    return shapes


def megatron_parameter(name: str, shape: tuple[int, ...], model_names: list[str]) -> torch.Tensor:
    """BF16-exact values: the position is the mantissa and the tensor's index in the model the exponent."""
    numel = math.prod(shape)
    assert numel <= MAX_NUMEL
    exponent = model_names.index(name) - len(model_names) // 2
    return ((MAX_NUMEL + torch.arange(numel)) * 2.0**exponent).to(torch.bfloat16).reshape(shape)


def megatron_parameters(layers, experts, *, last_stage: bool, model_names: list[str]) -> dict[str, torch.Tensor]:
    shapes = megatron_shapes(layers, experts, last_stage=last_stage)
    return {name: megatron_parameter(name, shape, model_names) for name, shape in shapes.items()}


def mapping(kind: str, hf_param, tp_size: int = 1):
    """A stand-in for a Megatron-Bridge mapping: the transport reads its class name, HF name(s) and TP size."""
    return type(kind, (), {"hf_param": hf_param, "tp_size": tp_size})()


def conversion_tasks(parameters: dict[str, torch.Tensor]) -> list:
    """The conversion tasks the Grug bridge produces for these parameters, one mapping kind per parameter type."""
    tasks = []
    for name, weight in parameters.items():
        layer = re.match(r"decoder\.layers\.(\d+)\.", name)
        hf = f"model.layers.{layer[1]}" if layer else None
        if name.endswith("linear_qkv.weight"):
            kind = mapping("QKVMapping", {part: f"{hf}.self_attn.{part}_proj.weight" for part in "qkv"})
        elif ".shared_experts.linear_fc1" in name:
            kind = mapping(
                "GatedMLPMapping", {part: f"{hf}.shared_expert.{part}_proj.weight" for part in ("gate", "up")}
            )
        elif name.endswith("router.weight"):
            kind = mapping("ReplicatedMapping", f"{hf}.mlp.router.weight")
        elif ".experts.linear_fc1" in name:
            kind = mapping(
                "GrugStackedGatedExpertMapping",
                {part: f"{hf}.mlp.experts.{part}_proj.weight" for part in ("gate", "up")},
            )
        elif ".experts.linear_fc2" in name:
            kind = mapping("GrugStackedExpertMapping", f"{hf}.mlp.experts.down_proj.weight")
        elif name == "decoder.final_layernorm.weight":
            kind = mapping("ReplicatedMapping", "model.norm.weight")
        else:
            kind = mapping("AutoMapping", "lm_head.weight")
        tasks.append(SimpleNamespace(param_weight=weight, global_param_name=name, mapping=kind))
    return tasks


def reference_hf(parameters: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[tuple, torch.Tensor]]:
    """``(dense HF tensors, expert matrices by (projection, layer, expert))`` these Megatron parameters hold."""
    dense, experts = {}, {}
    for name, weight in parameters.items():
        layer = re.match(r"decoder\.layers\.(\d+)\.", name)
        hf = f"model.layers.{layer[1]}" if layer else None
        expert = re.search(r"\.experts\.linear_(fc1|fc2)\.weight(\d+)$", name)
        if expert is not None:
            # vLLM's w13 slot is [gate;up], the trainer's fc1 order; w2 is down.
            experts[expert[1], int(layer[1]), int(expert[2])] = weight
        elif name.endswith("linear_qkv.weight"):
            # Megatron interleaves [q, ..., q, k, v] per KV group; HF stacks all q, all k, all v.
            grouped = weight.view(KV_GROUPS, QUERIES_PER_GROUP + 2, HEAD_DIM, HIDDEN)
            dense[f"{hf}.self_attn.q_proj.weight"] = grouped[:, :QUERIES_PER_GROUP].reshape(-1, HIDDEN)
            dense[f"{hf}.self_attn.k_proj.weight"] = grouped[:, QUERIES_PER_GROUP].reshape(-1, HIDDEN)
            dense[f"{hf}.self_attn.v_proj.weight"] = grouped[:, QUERIES_PER_GROUP + 1].reshape(-1, HIDDEN)
        elif ".shared_experts.linear_fc1" in name:
            gate, up = weight.chunk(2)
            dense[f"{hf}.shared_expert.gate_proj.weight"] = gate
            dense[f"{hf}.shared_expert.up_proj.weight"] = up
        elif name.endswith("router.weight"):
            dense[f"{hf}.mlp.router.weight"] = weight
        elif name == "decoder.final_layernorm.weight":
            dense["model.norm.weight"] = weight
        else:
            dense["lm_head.weight"] = weight
    return dense, experts
