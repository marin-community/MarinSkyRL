"""1-GPU install test for Megatron MoE router replay (R3), Stage 2.

Builds a tiny random-init Grug MoE Megatron model at TP1/PP1/EP1 through the
same bridge/provider path the worker uses and drives the replay controller
directly: capture-layer mapping from real ``TopKRouter.layer_number`` values,
replayed indices controlling ``routing_map`` on masked rows only, gradients
flowing through the gate, bit-identity of the substitution math when no row is
replayed, and the fail-fast install asserts. Worker/micro-batch plumbing is
Stage 3; this file must not depend on it.

Requires 1 GPU; run on an otherwise idle node (not part of the CPU PR gate).
"""

from __future__ import annotations

import functools
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

from skyrl_train.distributed.megatron.megatron_utils import get_model_config
from skyrl_train.models.grug_moe import GrugMoeConfig, GrugMoeForCausalLM
from skyrl_train.models.megatron_router_replay import (
    capture_layer_indices,
    expand_moe_layer_freq,
)
from skyrl_train.workers.megatron.router_replay_install import install_megatron_router_replay

TOKENIZER = "Qwen/Qwen2.5-0.5B-Instruct"
NUM_LAYERS = 8
NUM_EXPERTS = 8
TOPK = 4
SEQ_LEN = 64
HIDDEN = 64


def _write_tiny_checkpoint(path: Path) -> None:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    config = GrugMoeConfig(
        vocab_size=len(tokenizer),
        num_experts_per_tok=TOPK,
        max_position_embeddings=256,
        initializer_range=0.02,
        hidden_size=HIDDEN,
        intermediate_size=64,
        shared_expert_intermediate_size=32,
        num_local_experts=NUM_EXPERTS,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        sliding_window=64,
    )
    torch.manual_seed(17)
    GrugMoeForCausalLM(config).save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)


@functools.lru_cache(maxsize=1)
def _init_parallel_state() -> None:
    import torch.distributed

    torch.distributed.init_process_group(backend="nccl", world_size=1, rank=0, init_method="tcp://127.0.0.1:29531")
    import megatron.core.parallel_state as mpu

    mpu.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)


def _build_megatron_model(model_path: str) -> list[torch.nn.Module]:
    """Mirror MegatronWorker.init_configs + make_megatron_module at TP1/PP1/EP1."""
    import skyrl_train.models.grug_megatron_bridge  # noqa: F401  # registers the Grug bridge
    from megatron.bridge import AutoBridge

    _init_parallel_state()
    bridge = AutoBridge.from_hf_pretrained(model_path, trust_remote_code=True)
    provider = bridge.to_megatron_provider()
    provider.tensor_model_parallel_size = 1
    provider.pipeline_model_parallel_size = 1
    provider.pipeline_dtype = torch.bfloat16
    provider.context_parallel_size = 1
    provider.expert_model_parallel_size = 1
    provider.expert_tensor_parallel_size = 1
    provider.sequence_parallel = False
    provider.attention_backend = "fused"
    provider.variable_seq_lengths = True
    provider.masked_softmax_fusion = True
    provider.moe_token_dispatcher_type = "alltoall"
    # No activation recompute: exactly one router entry per forward, so the
    # recompute FIFO stays out of the picture (Stage 3 covers recompute).
    provider.recompute_granularity = None
    provider.recompute_method = None
    provider.recompute_num_layers = None
    provider.finalize()
    model = provider.provide_distributed_model(ddp_config=None, wrap_with_ddp=False, bf16=True)
    return model if isinstance(model, list) else [model]


def _routers(model: list[torch.nn.Module]) -> list:
    from megatron.core.transformer.moe.router import TopKRouter

    found = [m for chunk in model for m in chunk.modules() if isinstance(m, TopKRouter)]
    return sorted(found, key=lambda r: r.layer_number)


def _clear_handles(model: list[torch.nn.Module]) -> None:
    """Return every router to its flag-off state (router_replay is None)."""
    for router in _routers(model):
        router.router_replay = None


def _model_input(device: str) -> torch.Tensor:
    generator = torch.Generator().manual_seed(11)
    return torch.randint(10, 500, (1, SEQ_LEN), dtype=torch.long, device=device, generator=generator)


def _forward(model: list[torch.nn.Module], input_ids: torch.Tensor) -> torch.Tensor:
    logits = model[0](input_ids, None, None, fp32_output=False)[0]
    return logits


def _captured_routing_maps(model: list[torch.nn.Module], input_ids: torch.Tensor) -> dict[int, torch.Tensor]:
    """Run a forward and return layer_number -> routing_map ([tokens, experts] bool)."""
    captured: dict[int, torch.Tensor] = {}

    def make_hook(router):
        def hook(_module, _inputs, output):
            captured[router.layer_number] = output[1].detach().clone()

        return hook

    handles = [router.register_forward_hook(make_hook(router)) for router in _routers(model)]
    try:
        with torch.no_grad():
            _forward(model, input_ids)
    finally:
        for handle in handles:
            handle.remove()
    return captured


def _replay_targets(n: int, num_experts: int, topk: int, n_masked: int, device: str):
    generator = torch.Generator().manual_seed(23)
    targets = torch.randint(0, num_experts, (n, topk), generator=generator).to(device)
    mask = torch.zeros(n, dtype=torch.bool, device=device)
    mask[:n_masked] = True
    return targets, mask


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory) -> list[torch.nn.Module]:
    model_path = tmp_path_factory.mktemp("model")
    _write_tiny_checkpoint(model_path)
    return _build_megatron_model(str(model_path))


def test_installed_capture_indices_match_the_layer_pattern(tiny_model):
    config = get_model_config(tiny_model[0])
    pattern = expand_moe_layer_freq(config.moe_layer_freq, config.num_layers)
    expected = capture_layer_indices(pattern)

    controller = install_megatron_router_replay(tiny_model, recompute_enabled=True)

    installed = {router.layer_number: router.router_replay.layer_idx for router in _routers(tiny_model)}
    assert installed == expected
    assert controller.local_layer_indices == tuple(sorted(expected.values()))
    assert controller.num_moe_layers_total == len(expected)
    assert controller.topk == config.moe_router_topk


def test_replayed_indices_control_routing_on_masked_rows_only(tiny_model):
    device = torch.cuda.current_device()
    input_ids = _model_input(device)
    config = get_model_config(tiny_model[0])
    _clear_handles(tiny_model)

    native = _captured_routing_maps(tiny_model, input_ids)
    assert all(router.router_replay is None for router in _routers(tiny_model)), "flag-off router_replay must be None"

    controller = install_megatron_router_replay(tiny_model, recompute_enabled=False)
    pattern = expand_moe_layer_freq(config.moe_layer_freq, config.num_layers)
    mapping = capture_layer_indices(pattern)
    targets, mask = _replay_targets(SEQ_LEN, config.num_moe_experts, config.moe_router_topk, SEQ_LEN // 2, device)

    per_layer = {mapping[layer_number]: targets for layer_number in mapping}
    controller.begin_forward(per_layer, mask)
    replayed = _captured_routing_maps(tiny_model, input_ids)
    controller.end_forward()
    controller.assert_drained()

    for layer_number, routing_map in replayed.items():
        native_map = native[layer_number]
        for row in range(SEQ_LEN):
            replayed_experts = set(routing_map[row].nonzero().flatten().tolist())
            if mask[row]:
                assert replayed_experts == set(targets[row].tolist()), (layer_number, row)
            else:
                assert replayed_experts == set(native_map[row].nonzero().flatten().tolist()), (layer_number, row)


def test_all_false_mask_replay_is_bit_identical_to_native(tiny_model):
    device = torch.cuda.current_device()
    input_ids = _model_input(device)
    _clear_handles(tiny_model)
    with torch.no_grad():
        native_logits = _forward(tiny_model, input_ids).clone()

    controller = install_megatron_router_replay(tiny_model, recompute_enabled=False)
    config = get_model_config(tiny_model[0])
    pattern = expand_moe_layer_freq(config.moe_layer_freq, config.num_layers)
    mapping = capture_layer_indices(pattern)
    targets = torch.zeros(SEQ_LEN, config.moe_router_topk, dtype=torch.long, device=device)
    mask = torch.zeros(SEQ_LEN, dtype=torch.bool, device=device)

    controller.begin_forward({idx: targets for idx in mapping.values()}, mask)
    with torch.no_grad():
        replay_logits = _forward(tiny_model, input_ids)
    controller.end_forward()
    controller.assert_drained()
    assert torch.equal(replay_logits, native_logits)


def test_replay_keeps_gradients_flowing_through_the_gate(tiny_model):
    device = torch.cuda.current_device()
    input_ids = _model_input(device)
    controller = install_megatron_router_replay(tiny_model, recompute_enabled=False)
    config = get_model_config(tiny_model[0])
    pattern = expand_moe_layer_freq(config.moe_layer_freq, config.num_layers)
    mapping = capture_layer_indices(pattern)
    targets, mask = _replay_targets(SEQ_LEN, config.num_moe_experts, config.moe_router_topk, SEQ_LEN, device)

    controller.begin_forward({idx: targets for idx in mapping.values()}, mask)
    logits = _forward(tiny_model, input_ids)
    controller.end_forward()
    logits.float().sum().backward()
    controller.assert_drained()

    for router in _routers(tiny_model):
        grad = router.weight.main_grad if router.weight.main_grad is not None else router.weight.grad
        assert grad is not None and torch.isfinite(grad).all(), router.layer_number


def test_install_rejects_incompatible_router_configs(tiny_model):
    config = get_model_config(tiny_model[0])
    overrides = [
        ("moe_router_fusion", True, "moe_router_fusion"),
        ("moe_router_load_balancing_type", "sinkhorn", "sinkhorn"),
        ("moe_expert_capacity_factor", 1.5, "capacity"),
        ("moe_router_topk", 1, "moe_router_topk"),
        ("moe_enable_routing_replay", True, "moe_enable_routing_replay"),
    ]
    original = {name: getattr(config, name) for name, _, _ in overrides}
    try:
        for name, value, message in overrides:
            setattr(config, name, value)
            with pytest.raises(ValueError, match=message):
                install_megatron_router_replay(tiny_model, recompute_enabled=True)
    finally:
        for name, value in original.items():
            setattr(config, name, value)
