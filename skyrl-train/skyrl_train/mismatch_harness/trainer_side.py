"""Run one Megatron Grug decoder layer on one GPU and record the tensor at every region boundary.

The layer is built from the same provider settings as ``MegatronWorker.init_configs`` for a probe
forward (``trainer.flash_attn: False``, alltoall dispatcher), at TP=PP=CP=EP=1, with the layer number
of the production layer so its attention type, window and RoPE follow the model's layer pattern. The
harness checks that it reproduces the probe's captured regions byte for byte before it compares any
region with vLLM.
"""

from __future__ import annotations

from types import SimpleNamespace
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from megatron.bridge import AutoBridge
from megatron.core import parallel_state
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.spec_utils import build_module
from omegaconf import OmegaConf

import skyrl_train.models.grug_megatron_bridge  # noqa: F401  (registers the Grug bridge)
from skyrl_train.mismatch_probe.capture import capture_layer_regions
from skyrl_train.mismatch_probe.numerics import grug_numerics
from skyrl_train.models.grug_megatron import (
    GrugGatedRMSNorm,
    NormRole,
    clear_numerics_handoffs,
    grug_layer_spec,
    install_numerics_hooks,
)
from skyrl_train.models.grug_megatron_bridge import GrugMoeBridge
from skyrl_train.models.grug_vllm_kernels import fa3_attention_plan
from skyrl_train.models.megatron_router_replay import LayerReplayHandle, MegatronRouterReplay, VllmExpertParallel

POLICY_CONFIG = Path(__file__).resolve().parents[1] / "config" / "megatron_config" / "policy.yaml"
ATTENTION_BACKEND = "fused"
DISPATCHER = "alltoall"


@contextmanager
def single_rank_megatron(seed: int) -> Iterator[None]:
    """Initialize torch.distributed and Megatron model parallelism for one process on one GPU."""
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", store=dist.HashStore(), rank=0, world_size=1)
    parallel_state.initialize_model_parallel(1, 1, expert_model_parallel_size=1)
    model_parallel_cuda_manual_seed(seed)
    try:
        yield
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


def grug_provider(hf_model_dir: str):
    """The Grug Megatron provider with the probe worker's forward settings at TP=PP=CP=EP=1."""
    bridge = AutoBridge.from_hf_pretrained(hf_model_dir, trust_remote_code=True)
    provider = bridge.to_megatron_provider(load_weights=False)
    provider.tensor_model_parallel_size = 1
    provider.pipeline_model_parallel_size = 1
    provider.context_parallel_size = 1
    provider.expert_model_parallel_size = 1
    provider.expert_tensor_parallel_size = 1
    provider.sequence_parallel = False
    provider.pipeline_dtype = torch.bfloat16
    provider.attention_backend = ATTENTION_BACKEND
    provider.variable_seq_lengths = True
    provider.masked_softmax_fusion = True
    provider.moe_token_dispatcher_type = DISPATCHER
    provider.gradient_accumulation_fusion = False
    provider.perform_initialization = False
    policy = OmegaConf.to_container(OmegaConf.load(POLICY_CONFIG), resolve=True)
    for key, value in policy["transformer_config_kwargs"].items():
        setattr(provider, key, value)
    provider.finalize()
    return bridge, provider


def build_layer(provider, layer: int) -> torch.nn.Module:
    """One decoder layer with its production (1-based) layer number, in bf16 as Float16Module keeps it."""
    module = build_module(grug_layer_spec(provider), config=provider, layer_number=layer + 1)
    install_numerics_hooks(module)
    return module.cuda().bfloat16().eval()


def build_gated_norm(provider, role: NormRole) -> GrugGatedRMSNorm:
    """A standalone Grug gated norm in the given place of the model (the numerics read its role)."""
    norm = GrugGatedRMSNorm(provider, provider.hidden_size, provider.layernorm_epsilon).cuda().bfloat16().eval()
    norm.role = role
    return norm


def load_hf_weights(module: torch.nn.Module, prefix: str, tensors: Mapping[str, torch.Tensor]) -> None:
    """Copy Hugging Face tensors into ``module`` through the Grug bridge's parameter mappings.

    ``prefix`` is the module's Megatron name in the full model (``decoder.layers.13.``). Values are
    copied into the existing parameters and persistent buffers, as Megatron-Bridge does after the
    model is converted to bf16, so a buffer keeps the dtype the production module gives it.
    """
    registry = GrugMoeBridge().mapping_registry()
    state = module.state_dict(keep_vars=True)
    for local_name, target in state.items():
        if "_extra_state" in local_name:
            continue
        mapping = registry.megatron_to_hf_lookup(prefix + local_name)
        if mapping is None:
            raise KeyError(f"no Grug bridge mapping for {prefix + local_name}")
        owner = module.get_submodule(local_name.rsplit(".", 1)[0]) if "." in local_name else module
        if not hasattr(owner, "config"):
            owner.config = getattr(module, "config", None)
        if isinstance(mapping.hf_param, str):
            hf_weights = tensors[mapping.hf_param]
        else:
            hf_weights = {role: tensors[name] for role, name in mapping.hf_param.items()}
        converted = mapping.hf_to_megatron(_to_device(hf_weights), owner)
        if converted.shape != target.shape:
            raise ValueError(f"{prefix + local_name}: converted {tuple(converted.shape)} != {tuple(target.shape)}")
        with torch.no_grad():
            target.copy_(converted)


def _to_device(weights):
    if isinstance(weights, dict):
        return {key: value.cuda() for key, value in weights.items()}
    return weights.cuda()


@dataclass(frozen=True)
class VllmStep:
    """What the trainer's vLLM-kernel numerics need about vLLM's step: FA3 split counts and EP placement."""

    fa3_splits: Sequence[int] | None
    """Each sequence's FA3 split count; ``None`` runs the trainer's FA3 call unsplit."""
    home_rank: int
    ep_size: int


@dataclass(frozen=True)
class TrainerRegions:
    """Tensors the trainer layer produced, sequence-major ``[S, B, ...]`` as Megatron holds them."""

    tensors: dict[str, torch.Tensor]
    next_norm: dict[str, torch.Tensor]
    """The next layer's input norm (or the final norm) on this layer's output, under the same numerics."""


def _first(value):
    return value[0] if isinstance(value, (tuple, list)) else value


def run_layer(
    layer: torch.nn.Module,
    hidden_states: torch.Tensor,
    rotary: RotaryEmbedding,
    *,
    next_norm: GrugGatedRMSNorm,
    numerics: Mapping[str, bool],
    lengths: Sequence[int] | None = None,
    vllm_step: VllmStep | None = None,
) -> TrainerRegions:
    """Run the layer on ``hidden_states`` ``[S, B, H]`` and record every region boundary.

    The capture module's own hooks record the probe regions (so their names and hook points match the
    archived capture); the extra hooks here record the tensors inside each region that compiled vLLM
    also stores. ``next_norm`` runs on the layer's output inside the same numerics, so a residual the
    numerics keep in fp32 reaches it as it would reach the next layer. ``vllm_step`` supplies what the
    ``fa3_attention`` and ``ep_sum`` numerics take from vLLM's step: with ``ep_sum`` a router replay
    controller (armed with no captured routes, so routing stays native) serves every row the home rank.
    """
    records: dict[str, torch.Tensor] = {}

    def keep(name: str, tensor: torch.Tensor) -> None:
        if name not in records:
            records[name] = tensor.detach().clone()

    handles = []

    def hook_output(module: torch.nn.Module, name: str) -> None:
        handles.append(module.register_forward_hook(lambda m, a, out: keep(name, _first(out))))

    def hook_input(module: torch.nn.Module, name: str) -> None:
        handles.append(module.register_forward_pre_hook(lambda m, args: keep(name, args[0])))

    for prefix, norm in (("attn", layer.input_layernorm), ("mlp", layer.pre_mlp_layernorm)):
        hook_input(norm.down_proj, f"{prefix}_rms")
        hook_output(norm.down_proj, f"{prefix}_gate_down")
        hook_input(norm.up_proj, f"{prefix}_gate_act")
        hook_output(norm.up_proj, f"{prefix}_gate_up")
    attention = layer.self_attention
    hook_output(attention.linear_qkv, "qkv")
    hook_input(attention.q_layernorm, "q_proj")
    hook_output(attention.q_layernorm, "q_norm")
    hook_input(attention.k_layernorm, "k_proj")
    hook_output(attention.k_layernorm, "k_norm")

    def keep_attention_inputs(module, args) -> None:
        for name, tensor in zip(("query", "key", "value"), args[:3], strict=True):
            keep(name, tensor)

    handles.append(attention.core_attention.register_forward_pre_hook(keep_attention_inputs))
    hook_output(attention.core_attention, "core_attention")
    hook_output(attention.attn_gate, "attn_gate")
    hook_input(attention.linear_proj, "xsa_gate")
    hook_output(attention.linear_proj, "o_proj")
    moe = layer.mlp
    router = moe.router
    routing = router.routing

    def recorded_routing(logits: torch.Tensor, padding_mask=None):
        keep("router_logits", logits)
        return routing(logits, padding_mask)

    router.routing = recorded_routing
    dispatcher = moe.token_dispatcher
    combine_postprocess = dispatcher.combine_postprocess

    def recorded_combine(output: torch.Tensor) -> torch.Tensor:
        routed = combine_postprocess(output)
        keep("routed", routed)
        return routed

    dispatcher.combine_postprocess = recorded_combine
    # The experts' output rows (one per token-expert slot, route weight applied), whichever kernel computed them.
    hook_output(moe.experts, "expert_outputs")
    shared = moe.shared_experts
    hook_output(shared.linear_fc1, "shared_fc1")
    hook_input(shared.linear_fc2, "shared_act")
    hook_output(shared.linear_fc2, "shared_down")

    rotary_pos_emb = rotary(hidden_states.shape[0])
    controller = None
    if numerics.get("ep_sum"):
        if vllm_step is None:
            raise ValueError("ep_sum numerics need vLLM's home rank and EP size")
        rows = hidden_states.shape[0] * hidden_states.shape[1]
        controller = MegatronRouterReplay([0], recompute_enabled=False)
        router.router_replay = LayerReplayHandle(controller, 0)
        controller.begin_forward(
            {0: torch.zeros(rows, router.topk, dtype=torch.long, device=hidden_states.device)},
            torch.zeros(rows, dtype=torch.bool, device=hidden_states.device),
            vllm_expert_parallel=VllmExpertParallel(
                torch.full((rows,), vllm_step.home_rank, dtype=torch.long, device=hidden_states.device),
                vllm_step.ep_size,
            ),
        )
    plan = nullcontext()
    if numerics.get("fa3_attention") and vllm_step is not None and vllm_step.fa3_splits is not None:
        plan = fa3_attention_plan(vllm_step.fa3_splits, lengths)
    clear_numerics_handoffs()
    try:
        with (
            torch.no_grad(),
            grug_numerics(**numerics),
            plan,
            capture_layer_regions(
                [SimpleNamespace(decoder=SimpleNamespace(layers=[layer]))], [layer.layer_number - 1], "", enabled=True
            ) as captured,
        ):
            output, _ = layer(hidden_states=hidden_states, attention_mask=None, rotary_pos_emb=rotary_pos_emb)
            if controller is not None:
                controller.end_forward()
            next_regions = gated_norm_regions(next_norm, output)
    finally:
        for handle in handles:
            handle.remove()
        # The numerics hooks may have wrapped these per instance; put back what was there.
        router.routing = routing
        router.router_replay = None
        dispatcher.combine_postprocess = combine_postprocess
        clear_numerics_handoffs()
    regions = {name: tensor.cuda() for name, tensor in next(iter(captured.values())).items()}
    keep("layer_output", output)
    return TrainerRegions(tensors={**records, **regions}, next_norm=next_regions)


def rotary_embedding(provider) -> RotaryEmbedding:
    """The rotary embedding GPTModel builds for Grug (half-RoPE)."""
    return RotaryEmbedding(
        kv_channels=provider.kv_channels,
        rotary_percent=provider.rotary_percent,
        rotary_interleaved=provider.rotary_interleaved,
        seq_len_interpolation_factor=provider.seq_len_interpolation_factor,
        rotary_base=provider.rotary_base,
        rope_scaling=provider.rope_scaling,
        rope_scaling_factor=provider.rope_scaling_factor,
        use_cpu_initialization=provider.use_cpu_initialization,
    )


def gated_norm_regions(norm: GrugGatedRMSNorm, hidden_states: torch.Tensor) -> dict[str, torch.Tensor]:
    """A Grug gated norm's intermediate tensors (norm, down, SiLU, up, gated output) under the active numerics."""
    kept: dict[str, torch.Tensor] = {}

    def keep(name: str, tensor: torch.Tensor) -> None:
        kept[name] = tensor.detach().clone()

    handles = [
        norm.down_proj.register_forward_pre_hook(lambda module, args: keep("rms", args[0])),
        norm.down_proj.register_forward_hook(lambda module, args, output: keep("gate_down", output)),
        norm.up_proj.register_forward_pre_hook(lambda module, args: keep("gate_act", args[0])),
        norm.up_proj.register_forward_hook(lambda module, args, output: keep("gate_up", output)),
    ]
    try:
        with torch.no_grad():
            out = norm(hidden_states)
    finally:
        for handle in handles:
            handle.remove()
    return {**kept, "out": out}
