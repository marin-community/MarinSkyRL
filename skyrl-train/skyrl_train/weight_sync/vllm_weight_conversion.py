from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

import torch


_FUSED_GATE_UP_SUFFIX = ".experts.gate_up_proj"
_FUSED_DOWN_SUFFIX = ".experts.down_proj"


class VLLMWeightModel(Protocol):
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...


@dataclass(frozen=True)
class VLLMWeightConversion:
    weights: tuple[tuple[str, torch.Tensor], ...]
    required_parameters: frozenset[str]


def _fused_expert_prefix(name: str, suffix: str) -> str:
    return name[: -len(suffix)]


def convert_transformers_fused_moe_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> VLLMWeightConversion:
    """Convert Transformers fused MoE tensors to vLLM checkpoint names."""
    converted: list[tuple[str, torch.Tensor]] = []
    required_parameters: set[str] = set()

    for name, tensor in weights:
        if name.endswith(_FUSED_GATE_UP_SUFFIX):
            if tensor.ndim != 3 or tensor.shape[1] % 2:
                raise ValueError(f"Invalid fused MoE weight {name!r} with shape {tuple(tensor.shape)}")
            prefix = _fused_expert_prefix(name, _FUSED_GATE_UP_SUFFIX)
            gate, up = tensor.chunk(2, dim=1)
            for expert_id, (gate_weight, up_weight) in enumerate(zip(gate.unbind(0), up.unbind(0), strict=True)):
                expert_prefix = f"{prefix}.experts.{expert_id}"
                converted.append((f"{expert_prefix}.gate_proj.weight", gate_weight))
                converted.append((f"{expert_prefix}.up_proj.weight", up_weight))
            required_parameters.add(f"{prefix}.experts.w13_weight")
            continue

        if name.endswith(_FUSED_DOWN_SUFFIX):
            if tensor.ndim != 3:
                raise ValueError(f"Invalid fused MoE weight {name!r} with shape {tuple(tensor.shape)}")
            prefix = _fused_expert_prefix(name, _FUSED_DOWN_SUFFIX)
            for expert_id, down_weight in enumerate(tensor.unbind(0)):
                converted.append((f"{prefix}.experts.{expert_id}.down_proj.weight", down_weight))
            required_parameters.add(f"{prefix}.experts.w2_weight")
            continue

        converted.append((name, tensor))

    return VLLMWeightConversion(tuple(converted), frozenset(required_parameters))


def load_weights_into_vllm(
    model: VLLMWeightModel,
    weights: Iterable[tuple[str, torch.Tensor]],
) -> set[str]:
    """Load one weight-sync batch and reject skipped fused MoE parameters."""
    conversion = convert_transformers_fused_moe_weights(weights)
    loaded_parameters = model.load_weights(iter(conversion.weights))
    missing_parameters = conversion.required_parameters.difference(loaded_parameters)
    if missing_parameters:
        missing = ", ".join(sorted(missing_parameters))
        raise RuntimeError(f"vLLM did not load required fused MoE parameters: {missing}")
    return loaded_parameters


def validate_dummy_weight_coverage(
    layers: dict[str, dict], loaded_parameters: set[str], skipped_float_parameters: set[str]
) -> None:
    """Reject incomplete initial dummy weights before vLLM finalizes the reload."""
    satisfied_storage = {
        identity for layer in layers.values() if not layer["can_load"] for identity, _numel in layer["tensors"].values()
    }
    pending = {name: layer for name, layer in layers.items() if layer["can_load"]}
    gaps = []
    while pending:
        gaps = []
        satisfied = []
        for name, layer in pending.items():
            weight_padding = layer["vocab_padding_numel"]
            bias_padding = layer.get("vocab_bias_padding_numel", 0)
            padding = {"weight": weight_padding, "bias": bias_padding} | layer.get("tensor_padding_numel", {})
            required = layer["load_numel_total"] - sum(padding.values()) - layer.get("non_persistent_numel", 0)
            required -= layer.get("generated_numel", 0)
            for tensor_name, (identity, numel) in layer["tensors"].items():
                if identity in satisfied_storage:
                    required -= numel - padding.get(tensor_name, 0)
            if layer["load_numel"] < max(0, required):
                gaps.append(f"{name or '<root>'}: {layer['load_numel']} of {required} elements loaded")
            else:
                satisfied.append(name)
                satisfied_storage.update(identity for identity, _numel in layer["tensors"].values())
        if not satisfied:
            break
        for name in satisfied:
            del pending[name]
    gaps.extend(
        f"{name}: floating skipped tensor not loaded" for name in sorted(skipped_float_parameters - loaded_parameters)
    )
    if gaps:
        raise RuntimeError("Incomplete dummy engine weights: " + "; ".join(gaps))
