from collections.abc import Iterable
from dataclasses import dataclass
import re
from typing import Protocol

import torch


_FUSED_GATE_UP_SUFFIX = ".experts.gate_up_proj"
_FUSED_DOWN_SUFFIX = ".experts.down_proj"
_GRUG_EXPERT_PARAMETER_MAPPING = (
    ("experts.gate_proj.weight", "experts.routed_experts.w13_weight"),
    ("experts.up_proj.weight", "experts.routed_experts.w13_weight"),
    ("experts.down_proj.weight", "experts.routed_experts.w2_weight"),
)
_PACKED_PARAMETER_MAPPING = (
    ("self_attn.q_proj", "self_attn.qkv_proj"),
    ("self_attn.k_proj", "self_attn.qkv_proj"),
    ("self_attn.v_proj", "self_attn.qkv_proj"),
    ("mlp.gate_proj", "mlp.gate_up_proj"),
    ("mlp.up_proj", "mlp.gate_up_proj"),
)
_EXPERT_SLICE_PATTERN = re.compile(
    r"^(?P<prefix>.+\.experts)\.(?P<expert_id>\d+)\."
    r"(?P<projection>gate_proj|up_proj|down_proj)\.weight$"
)


class VLLMWeightModel(Protocol):
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...


@dataclass(frozen=True)
class VLLMWeightConversion:
    weights: tuple[tuple[str, torch.Tensor], ...]
    expected_parameters: frozenset[str]


def _fused_expert_prefix(name: str, suffix: str) -> str:
    return name[: -len(suffix)]


def expected_vllm_parameter_names(names: Iterable[str]) -> frozenset[str]:
    """Map source checkpoint names to the parameters vLLM reports installed."""
    expected: set[str] = set()
    for name in names:
        if name.endswith(_FUSED_GATE_UP_SUFFIX):
            expected.add(f"{_fused_expert_prefix(name, _FUSED_GATE_UP_SUFFIX)}.experts.w13_weight")
            continue
        if name.endswith(_FUSED_DOWN_SUFFIX):
            expected.add(f"{_fused_expert_prefix(name, _FUSED_DOWN_SUFFIX)}.experts.w2_weight")
            continue
        for source, target in _GRUG_EXPERT_PARAMETER_MAPPING:
            if source in name:
                expected.add(name.replace(source, target))
                break
        else:
            expected.add(name)
    return frozenset(expected)


def convert_transformers_fused_moe_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> VLLMWeightConversion:
    """Convert Transformers fused MoE tensors to vLLM checkpoint names."""
    converted: list[tuple[str, torch.Tensor]] = []
    source_names: list[str] = []

    for name, tensor in weights:
        source_names.append(name)
        if name.endswith(_FUSED_GATE_UP_SUFFIX):
            if tensor.ndim != 3 or tensor.shape[1] % 2:
                raise ValueError(f"Invalid fused MoE weight {name!r} with shape {tuple(tensor.shape)}")
            prefix = _fused_expert_prefix(name, _FUSED_GATE_UP_SUFFIX)
            gate, up = tensor.chunk(2, dim=1)
            for expert_id, (gate_weight, up_weight) in enumerate(zip(gate.unbind(0), up.unbind(0), strict=True)):
                expert_prefix = f"{prefix}.experts.{expert_id}"
                converted.append((f"{expert_prefix}.gate_proj.weight", gate_weight))
                converted.append((f"{expert_prefix}.up_proj.weight", up_weight))
            continue

        if name.endswith(_FUSED_DOWN_SUFFIX):
            if tensor.ndim != 3:
                raise ValueError(f"Invalid fused MoE weight {name!r} with shape {tuple(tensor.shape)}")
            prefix = _fused_expert_prefix(name, _FUSED_DOWN_SUFFIX)
            for expert_id, down_weight in enumerate(tensor.unbind(0)):
                converted.append((f"{prefix}.experts.{expert_id}.down_proj.weight", down_weight))
            continue

        converted.append((name, tensor))

    return VLLMWeightConversion(tuple(converted), expected_vllm_parameter_names(source_names))


def load_weights_into_vllm(
    model: VLLMWeightModel,
    weights: Iterable[tuple[str, torch.Tensor]],
) -> set[str]:
    """Load one weight-sync batch and return its logical parameter receipt.

    vLLM reports the same packed parameter name after loading each q/k/v or
    gate/up source tensor. Load those source tensors separately so one successful
    sibling cannot hide another sibling that the loader silently skipped.
    """
    conversion = convert_transformers_fused_moe_weights(weights)
    ordinary_weights: list[tuple[str, torch.Tensor]] = []
    acknowledged_parameters: set[str] = set()
    missing_parameters: set[str] = set()

    for name, tensor in conversion.weights:
        candidates = _reported_parameter_candidates(name)
        if len(candidates) == 1:
            ordinary_weights.append((name, tensor))
            continue
        loaded_parameters = model.load_weights(iter(((name, tensor),)))
        if candidates.isdisjoint(loaded_parameters):
            missing_parameters.add(name)
        else:
            acknowledged_parameters.add(name)

    if ordinary_weights:
        loaded_parameters = model.load_weights(iter(ordinary_weights))
        missing_parameters.update(
            conversion.expected_parameters.difference(acknowledged_parameters | loaded_parameters)
        )
    if missing_parameters:
        missing = ", ".join(sorted(missing_parameters))
        raise RuntimeError(f"vLLM did not load required parameters: {missing}")
    return set(conversion.expected_parameters)


def _reported_parameter_candidates(expected: str) -> frozenset[str]:
    """Return direct and packed parameter names that vLLM may report."""
    candidates = {expected}
    expert_slice = _EXPERT_SLICE_PATTERN.match(expected)
    if expert_slice is not None:
        projection = expert_slice.group("projection")
        packed = "w13_weight" if projection in {"gate_proj", "up_proj"} else "w2_weight"
        candidates.add(f"{expert_slice.group('prefix')}.{packed}")
    for source, target in _PACKED_PARAMETER_MAPPING:
        if source in expected:
            candidates.add(expected.replace(source, target))
    return frozenset(candidates)
