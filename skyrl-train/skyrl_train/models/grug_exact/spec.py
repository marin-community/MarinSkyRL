"""The Megatron layer spec of the exact Grug trainer, which ``GrugModelProvider`` selects under exact numerics."""

from functools import partial

from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec_for_backend
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayerSubmodules
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add

from skyrl_train.models.grug_exact.layers import (
    ExactGatedNorm,
    ExactGroupedMLP,
    ExactInputGatedNorm,
    ExactMoELayer,
    ExactSelfAttention,
    ExactSharedExpertMLP,
    ExactTopKRouter,
    ExactTransformerLayer,
)
from skyrl_train.models.grug_megatron import grug_self_attention_spec

# The parallelism the exact modules run on: TP 1, CP 1 and ETP 1, as the decode-invariant engines' kernels take one
# rank's whole rows.
EXACT_PARALLELISM = {"tensor_model_parallel_size": 1, "context_parallel_size": 1, "expert_tensor_parallel_size": 1}


def exact_layer_spec(config: TransformerConfig) -> ModuleSpec:
    """The Transformer Engine layer spec for one exact Grug decoder layer."""
    if config.grug_hero or config.grug_sconv_sites or config.moe_latent_size:
        raise ValueError("exact numerics cover Grug without Hero layers, ShortConv or latent experts")
    found = {name: getattr(config, name) for name in EXACT_PARALLELISM}
    if found != EXACT_PARALLELISM:
        raise ValueError(f"exact numerics run on {EXACT_PARALLELISM}, got {found}")
    backend = TESpecProvider()
    moe = get_moe_module_spec_for_backend(backend, num_experts=config.num_moe_experts, moe_grouped_gemm=True)
    submodules = moe.keywords["submodules"]
    submodules.router = ExactTopKRouter
    submodules.experts = partial(ExactGroupedMLP, submodules=submodules.experts.keywords["submodules"])
    submodules.shared_experts = partial(
        ExactSharedExpertMLP, submodules=submodules.shared_experts.keywords["submodules"]
    )
    return ModuleSpec(
        module=ExactTransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=ExactInputGatedNorm,
            self_attention=grug_self_attention_spec(backend, ExactSelfAttention),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=ExactGatedNorm,
            mlp=partial(ExactMoELayer, submodules=submodules),
            mlp_bda=get_bias_dropout_add,
        ),
    )
