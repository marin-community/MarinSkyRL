"""Public Megatron Bridge seam for the canonical Grug policy model."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import ReplicatedMapping
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module

from skyrl_train.models.grug_moe import (
    GRUG_FLASH_ATTENTION_BACKEND,
    GRUG_MOE_MODEL_TYPE,
    GrugMoeConfig,
    GrugMoeForCausalLM,
    GrugMoeRouter,
    validate_grug_megatron_policy,
)


GRUG_CAUSAL_LM_SPEC = ModuleSpec(module=GrugMoeForCausalLM)


class GrugMoeMegatronModel(MegatronModule):
    """MCore lifecycle adapter around the single canonical Grug implementation."""

    def __init__(self, config: GrugMoeModelProvider, grug_config: GrugMoeConfig) -> None:
        super().__init__(config)
        self.model = build_module(GRUG_CAUSAL_LM_SPEC, config=grug_config)

    def _restore_router_bias_dtype(self) -> None:
        for module in self.model.modules():
            if isinstance(module, GrugMoeRouter):
                module.bias.data = module.bias.data.float()

    def bfloat16(self) -> GrugMoeMegatronModel:
        super().bfloat16()
        self._restore_router_bias_dtype()
        return self

    def half(self) -> GrugMoeMegatronModel:
        super().half()
        self._restore_router_bias_dtype()
        return self

    def set_input_tensor(self, input_tensor: Any) -> None:
        tensors = input_tensor if isinstance(input_tensor, list) else [input_tensor]
        if any(tensor is not None for tensor in tensors):
            raise ValueError("Grug Megatron does not admit pipeline input tensors")

    def begin_query_bias_capture(self, candidate_count: int, token_mask: torch.Tensor) -> None:
        self.model.begin_query_bias_capture(candidate_count, token_mask)

    def take_query_bias_observation(self, *, candidate_count: int):
        return self.model.take_query_bias_observation(candidate_count=candidate_count)

    def set_query_bias(self, bias: torch.Tensor) -> None:
        self.model.set_query_bias(bias)

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor | None,
        attention_mask: torch.Tensor | None,
        decoder_input: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        packed_seq_params: Any = None,
        fp32_output: bool = False,
        return_dict: bool = False,
        **kwargs: Any,
    ) -> Any:
        if decoder_input is not None:
            raise ValueError("Grug Megatron does not admit pipeline decoder inputs")
        if packed_seq_params is not None:
            raise ValueError("Grug Megatron does not admit packed sequences")
        output = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
            **kwargs,
        )
        if return_dict:
            return output
        return output.logits.float() if fp32_output else output.logits


@dataclass
class GrugMoeModelProvider(GPTModelProvider):
    """Construct the guarded world-size-one Grug MCore model."""

    grug_hf_config: dict[str, Any] = field(default_factory=dict)

    def _validate_topology(self) -> None:
        validate_grug_megatron_policy(
            GRUG_MOE_MODEL_TYPE,
            is_policy_worker=True,
            world_size=dist.get_world_size() if dist.is_initialized() else 1,
            tensor_model_parallel_size=self.tensor_model_parallel_size,
            pipeline_model_parallel_size=self.pipeline_model_parallel_size,
            context_parallel_size=self.context_parallel_size,
            expert_model_parallel_size=self.expert_model_parallel_size,
            expert_tensor_parallel_size=self.expert_tensor_parallel_size,
            virtual_pipeline_model_parallel_size=self.virtual_pipeline_model_parallel_size,
            use_sample_packing=False,
        )

    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> GrugMoeMegatronModel:
        self._validate_topology()
        if pre_process is False or post_process is False or vp_stage not in (None, 0):
            raise ValueError("Grug Megatron requires one unsplit pipeline stage")
        grug_config = GrugMoeConfig(**deepcopy(self.grug_hf_config))
        attention_backend = getattr(self.attention_backend, "value", self.attention_backend)
        grug_config._attn_implementation = GRUG_FLASH_ATTENTION_BACKEND if attention_backend == "flash" else "eager"
        return GrugMoeMegatronModel(self, grug_config)


_GRUG_REPLICATED_STATE_NAMES = (
    "model.embed_tokens.weight",
    "model.embed_norm.weight",
    "model.embed_gated_norm.down_proj.weight",
    "model.embed_gated_norm.up_proj.weight",
    "model.layers.*.self_attn.q_proj.weight",
    "model.layers.*.self_attn.k_proj.weight",
    "model.layers.*.self_attn.v_proj.weight",
    "model.layers.*.self_attn.o_proj.weight",
    "model.layers.*.self_attn.attn_gate.weight",
    "model.layers.*.input_layernorm.weight",
    "model.layers.*.attn_gated_norm.down_proj.weight",
    "model.layers.*.attn_gated_norm.up_proj.weight",
    "model.layers.*.post_attention_layernorm.weight",
    "model.layers.*.mlp_gated_norm.down_proj.weight",
    "model.layers.*.mlp_gated_norm.up_proj.weight",
    "model.layers.*.mlp.router.weight",
    "model.layers.*.mlp.router.bias",
    "model.layers.*.mlp.experts.gate_proj.weight",
    "model.layers.*.mlp.experts.up_proj.weight",
    "model.layers.*.mlp.experts.down_proj.weight",
    "model.layers.*.shared_expert.gate_proj.weight",
    "model.layers.*.shared_expert.up_proj.weight",
    "model.layers.*.shared_expert.down_proj.weight",
    "model.norm.weight",
    "model.final_gated_norm.down_proj.weight",
    "model.final_gated_norm.up_proj.weight",
    "lm_head.weight",
)


@MegatronModelBridge.register_bridge(
    source=GrugMoeForCausalLM.__name__,
    target=GrugMoeMegatronModel,
    provider=GrugMoeModelProvider,
    model_type=GRUG_MOE_MODEL_TYPE,
)
class GrugMoeBridge(MegatronModelBridge):
    """Translate exact Grug HF names without changing tensor layout."""

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> GrugMoeModelProvider:
        hf_config = hf_pretrained.config
        dtype = self.dtype_from_hf(hf_config, default=torch.float32)
        return GrugMoeModelProvider(
            num_layers=hf_config.num_hidden_layers,
            hidden_size=hf_config.hidden_size,
            num_attention_heads=hf_config.num_attention_heads,
            num_query_groups=hf_config.num_key_value_heads,
            kv_channels=hf_config.head_dim,
            ffn_hidden_size=hf_config.intermediate_size,
            seq_length=hf_config.max_position_embeddings,
            vocab_size=hf_config.vocab_size,
            make_vocab_size_divisible_by=self.make_vocab_size_divisible_by(hf_config.vocab_size),
            share_embeddings_and_output_weights=False,
            params_dtype=dtype,
            fp16=dtype == torch.float16,
            bf16=dtype == torch.bfloat16,
            grug_hf_config=hf_config.to_dict(),
        )

    @classmethod
    def megatron_to_hf_config(cls, provider: GrugMoeModelProvider) -> dict[str, Any]:
        hf_config = deepcopy(provider.grug_hf_config)
        hf_config["architectures"] = [GrugMoeForCausalLM.__name__]
        hf_config["model_type"] = GRUG_MOE_MODEL_TYPE
        hf_config["dtype"] = str(provider.params_dtype).removeprefix("torch.")
        return hf_config

    def mapping_registry(self) -> MegatronMappingRegistry:
        return MegatronMappingRegistry(
            *(ReplicatedMapping(f"model.{name}", name) for name in _GRUG_REPLICATED_STATE_NAMES)
        )
