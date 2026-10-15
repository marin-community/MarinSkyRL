"""Grug for a decode-invariant vLLM engine: the stock model with its router logits and head gates from the
fixed-tile GEMM, and one Triton configuration for the fused expert kernels at every token count.

``ModelConfig.model_class_overrides`` maps the ``GrugMoeForCausalLM`` architecture to
``DecodeInvariantGrugMoeForCausalLM`` in every process of a decode-invariant engine.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.model_executor.layers.fused_moe import override_config
from vllm.model_executor.layers.linear import ReplicatedLinear, UnquantizedLinearMethod
from vllm.model_executor.models.grugmoe import GrugMoeDecoderLayer, GrugMoeForCausalLM
from vllm.sequence import IntermediateTensors

from skyrl_train.inference_engines.vllm.decode_invariant.gemm import check_bf16_values

ROUTER_WEIGHT_SUFFIX = ".mlp.router.weight"
# The fused expert kernels' tile configuration at every token count, the one vLLM's batch-invariant mode uses; vLLM
# otherwise picks a configuration per token count, with K blocks of 64 or 128 columns.
INVARIANT_MOE_CONFIG = {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8, "SPLIT_K": 1}


class InvariantLinear(ReplicatedLinear):
    """A projection computed by ``skyrl::invariant_gemm`` in the input's dtype; returns the output and, with
    ``skip_bias_add``, the bias, as ``ReplicatedLinear`` does."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.nn.Parameter | None]:
        output = torch.ops.skyrl.invariant_gemm(x, self.weight).to(x.dtype)
        return output, self.bias if self.skip_bias_add else None


def invariant_linear(layer: ReplicatedLinear) -> InvariantLinear:
    """A projection of ``layer``'s shape, dtype and bias that ``skyrl::invariant_gemm`` computes; raises for a
    quantized layer or a bias the layer adds itself, which ``InvariantLinear`` does not compute."""
    if not isinstance(layer.quant_method, UnquantizedLinearMethod):
        raise NotImplementedError(
            f"{layer.prefix} is quantized; a decode-invariant engine runs unquantized projections"
        )
    if layer.has_bias and not layer.skip_bias_add:
        raise NotImplementedError(
            f"{layer.prefix} adds its bias; InvariantLinear returns the bias for the caller to add"
        )
    return InvariantLinear(
        layer.input_size,
        layer.output_size,
        bias=layer.has_bias,
        skip_bias_add=layer.skip_bias_add,
        params_dtype=layer.params_dtype,
        prefix=layer.prefix,
        disable_tp=True,
    )


def check_router_weights(model: torch.nn.Module, loaded: set[str]) -> None:
    """Raise unless every router weight among the ``loaded`` parameter names holds bf16 values."""
    for name, parameter in model.named_parameters():
        if name in loaded and name.endswith(ROUTER_WEIGHT_SUFFIX):
            check_bf16_values(name, parameter.data)


class DecodeInvariantGrugMoeForCausalLM(GrugMoeForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        if not vllm_config.use_v2_model_runner:
            raise ValueError(
                "a decode-invariant engine runs vLLM's model runner V2, whose log-probability kernel the exact "
                "trainer runs; vLLM chose the V1 runner"
            )
        compilation = vllm_config.compilation_config
        if compilation.cudagraph_mode == CUDAGraphMode.NONE or 1 in compilation.cudagraph_capture_sizes:
            # vLLM sets cudagraph_mode NONE for a Grug model with short convolutions.
            raise ValueError(
                "a decode-invariant engine runs every one-token step padded in a CUDA graph; vLLM chose "
                f"cudagraph_mode={compilation.cudagraph_mode.name} and capture sizes {compilation.cudagraph_capture_sizes}"
            )
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        for layer in self.model.layers:
            if isinstance(layer, GrugMoeDecoderLayer):
                layer.mlp.router = invariant_linear(layer.mlp.router)
                # cuBLAS computes the 20-column head-gate GEMM with a token-count-dependent split of its 2,560-wide
                # reduction.
                layer.self_attn.attn_gate = invariant_linear(layer.self_attn.attn_gate)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        with override_config(INVARIANT_MOE_CONFIG):
            return super().forward(input_ids, positions, intermediate_tensors, inputs_embeds)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded = super().load_weights(weights)
        check_router_weights(self, loaded)
        return loaded
