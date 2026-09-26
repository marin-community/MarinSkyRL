"""CPU oracle for the pinned Marin Grug MuonH production recipe."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from skyrl_train.distributed.megatron.grug_muonh import MegatronGrugMuonH, megatron_grug_route


FIXTURE = Path(__file__).with_name("fixtures") / "grug_muonh_jax_golden.npz"


class _Weight(nn.Module):
    def __init__(self, value: torch.Tensor) -> None:
        super().__init__()
        self.weight = nn.Parameter(value.clone())


class _Router(_Weight):
    def __init__(self, value: torch.Tensor) -> None:
        super().__init__(value)
        self.register_buffer("bias", torch.zeros(value.shape[0], dtype=torch.float32), persistent=True)


class _TinyGrug(nn.Module):
    """Small module whose parameter names cover every production route."""

    def __init__(self, fixture) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = _Weight(_tensor(fixture["initial__embed"]))
        self.model.layers = nn.ModuleList([nn.Module()])
        layer = self.model.layers[0]
        layer.self_attn = nn.Module()
        layer.self_attn.q_proj = _Weight(_tensor(fixture["initial__q_proj"]))
        layer.self_attn.attn_gate = _Weight(_tensor(fixture["initial__attn_gate"]))
        layer.mlp = nn.Module()
        layer.mlp.router = _Router(_tensor(fixture["initial__router"]))
        layer.mlp.experts = nn.Module()
        layer.mlp.experts.gate_proj = _Weight(_tensor(fixture["initial__expert"]))
        layer.mlp.shared_expert = nn.Module()
        layer.mlp.shared_expert.up_proj = _Weight(_tensor(fixture["initial__shared"]))
        self.model.embed_gated_norm = nn.Module()
        self.model.embed_gated_norm.down_proj = _Weight(_tensor(fixture["initial__gated_norm"]))
        layer.input_layernorm = _Weight(_tensor(fixture["initial__norm"]))
        layer.bias = nn.Parameter(_tensor(fixture["initial__bias"]))
        self.lm_head = _Weight(_tensor(fixture["initial__output"]))


PARAMETER_NAMES = {
    "embed": "model.embed_tokens.weight",
    "q_proj": "model.layers.0.self_attn.q_proj.weight",
    "attn_gate": "model.layers.0.self_attn.attn_gate.weight",
    "router": "model.layers.0.mlp.router.weight",
    "expert": "model.layers.0.mlp.experts.gate_proj.weight",
    "shared": "model.layers.0.mlp.shared_expert.up_proj.weight",
    "gated_norm": "model.embed_gated_norm.down_proj.weight",
    "norm": "model.layers.0.input_layernorm.weight",
    "bias": "model.layers.0.bias",
    "output": "lm_head.weight",
}


def _tensor(array) -> torch.Tensor:
    return torch.from_numpy(np.asarray(array).copy())


def _assert_close(actual: torch.Tensor, expected, *, muon_bf16: bool = False) -> None:
    # The pinned transform executes Newton--Schulz in BF16. XLA and oneDNN use
    # different reduction orders, while all FP32 state math agrees more tightly.
    if muon_bf16:
        torch.testing.assert_close(actual, _tensor(expected), rtol=3e-3, atol=1.5e-3)
    else:
        torch.testing.assert_close(actual, _tensor(expected), rtol=2e-6, atol=5e-7)


def test_megatron_muonh_matches_independent_jax_steps_after_own_state_resume():
    with np.load(FIXTURE, allow_pickle=False) as fixture:
        model = _TinyGrug(fixture)
        parameters = dict(model.named_parameters())
        routes = dict(zip(fixture["metadata_names"].tolist(), fixture["metadata_routes"].tolist()))
        route_names = {"muonh": "grug_muonh", "adamh": "grug_adamh"}

        def optimizers(current_model):
            current = dict(current_model.named_parameters())
            groups = [
                {"params": [current[PARAMETER_NAMES[key]]], "optimizer": route_names[routes[key]]}
                for key in PARAMETER_NAMES
                if routes[key] != "adam"
            ]
            adam_parameters = [current[PARAMETER_NAMES[key]] for key in PARAMETER_NAMES if routes[key] == "adam"]
            hero_optimizer = MegatronGrugMuonH(
                groups,
                lr=float(fixture["metadata_shared_lr"]),
                betas=(0.9, 0.95),
                momentum=0.95,
                nesterov=True,
                ns_steps=5,
                eps=1e-8,
                muon_eps=1e-8,
                qkv_split_shapes=(4, 2, 2),
            )
            adam_optimizer = torch.optim.Adam(
                adam_parameters, lr=float(fixture["metadata_adam_lr"]), betas=(0.9, 0.95), eps=1e-8
            )
            return hero_optimizer, adam_optimizer

        hero_optimizer, adam_optimizer = optimizers(model)
        assert not hero_optimizer.state
        for step in range(1, int(fixture["metadata_steps"]) + 1):
            for key, name in PARAMETER_NAMES.items():
                parameters[name].grad = _tensor(fixture[f"gradient_{step}__{key}"])
            hero_optimizer.step()
            adam_optimizer.step()

            for key, name in PARAMETER_NAMES.items():
                _assert_close(parameters[name], fixture[f"parameter_{step}__{key}"], muon_bf16=routes[key] == "muonh")

            if step == 1:
                saved_model = copy.deepcopy(model.state_dict())
                saved_hero = copy.deepcopy(hero_optimizer.state_dict())
                saved_adam = copy.deepcopy(adam_optimizer.state_dict())
                model = _TinyGrug(fixture)
                model.load_state_dict(saved_model)
                parameters = dict(model.named_parameters())
                hero_optimizer, adam_optimizer = optimizers(model)
                hero_optimizer.initialize_state()
                hero_optimizer.load_state_dict(saved_hero)
                adam_optimizer.load_state_dict(saved_adam)


@pytest.mark.parametrize("groups", [1, 2])
def test_megatron_muonh_splits_fused_qkv_and_gate_up_before_hyperball_update(groups):
    torch.manual_seed(31)
    q, k, v = (torch.randn(rows * groups, 5) for rows in (4, 2, 2))
    q_grad, k_grad, v_grad = (torch.randn_like(weight) for weight in (q, k, v))
    fused = torch.cat((q.view(groups, 4, 5), k.view(groups, 2, 5), v.view(groups, 2, 5)), dim=1).reshape(8 * groups, 5)
    fused_grad = torch.cat(
        (q_grad.view(groups, 4, 5), k_grad.view(groups, 2, 5), v_grad.view(groups, 2, 5)), dim=1
    ).reshape(8 * groups, 5)
    gate, up = torch.randn(6, 5), torch.randn(6, 5)
    gate_grad, up_grad = torch.randn_like(gate), torch.randn_like(up)
    fused_gate_up = torch.cat((gate, up), dim=0)

    reference = [torch.nn.Parameter(value.clone()) for value in (q, k, v, gate, up)]
    for parameter, gradient in zip(reference, (q_grad, k_grad, v_grad, gate_grad, up_grad)):
        parameter.grad = gradient.clone()
    reference_optimizer = MegatronGrugMuonH(
        [{"params": reference, "optimizer": "grug_muonh"}],
        lr=0.03,
        betas=(0.9, 0.95),
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        eps=1e-8,
        muon_eps=1e-8,
        qkv_split_shapes=(4, 2, 2),
    )

    actual_qkv = torch.nn.Parameter(fused.clone())
    actual_gate_up = torch.nn.Parameter(fused_gate_up.clone())
    actual_qkv.grad = fused_grad.clone()
    actual_gate_up.grad = torch.cat((gate_grad, up_grad), dim=0)
    optimizer = MegatronGrugMuonH(
        [
            {"params": [actual_qkv], "optimizer": "grug_muonh_qkv"},
            {"params": [actual_gate_up], "optimizer": "grug_muonh_gate_up"},
        ],
        lr=0.03,
        betas=(0.9, 0.95),
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        eps=1e-8,
        muon_eps=1e-8,
        qkv_split_shapes=(4, 2, 2),
    )
    reference_optimizer.step()
    optimizer.step()

    expected_qkv = torch.cat(
        (reference[0].view(groups, 4, 5), reference[1].view(groups, 2, 5), reference[2].view(groups, 2, 5)), dim=1
    ).reshape(8 * groups, 5)
    torch.testing.assert_close(actual_qkv, expected_qkv, rtol=0, atol=0)
    torch.testing.assert_close(actual_gate_up, torch.cat((reference[3], reference[4]), dim=0), rtol=0, atol=0)


def test_megatron_muonh_routes_hero_parameter_families():
    matrix = torch.empty(8, 8)
    vector = torch.empty(8)
    assert megatron_grug_route("decoder.layers.0.self_attention.linear_qkv.weight", matrix) == "grug_muonh_qkv"
    assert megatron_grug_route("decoder.layers.0.mlp.experts.linear_fc1.weight3", matrix) == "grug_muonh_gate_up"
    assert (
        megatron_grug_route("decoder.layers.0.mlp.shared_experts.experts.1.linear_fc1.weight", matrix)
        == "grug_muonh_gate_up"
    )
    assert megatron_grug_route("decoder.layers.0.attn_gated_norm.down_proj.weight", matrix) == "grug_muonh"
    assert megatron_grug_route("embed_norm.down_proj.weight", matrix) == "grug_muonh"
    assert megatron_grug_route("decoder.layers.0.input_layernorm.down_proj.weight", matrix) == "grug_muonh"
    assert megatron_grug_route("decoder.layers.0.pre_mlp_layernorm.up_proj.weight", matrix) == "grug_muonh"
    assert megatron_grug_route("decoder.final_layernorm.up_proj.weight", matrix) == "grug_muonh"
    assert megatron_grug_route("output_layer.weight", matrix) == "grug_adamh"
    assert megatron_grug_route("decoder.layers.0.self_attention.sconv_k.weight", matrix) == "adam"
    assert megatron_grug_route("decoder.layers.0.mlp.router.weight", matrix) == "adam"
    assert megatron_grug_route("decoder.layers.0.input_layernorm.weight", vector) == "adam"


def test_megatron_muonh_rejects_gradient_clipping_in_mcore_config():
    try:
        from skyrl_train.distributed.megatron.optimizer import init_megatron_optim_config
    except ImportError:
        pytest.skip("Megatron Core optimizer is not in the CPU test profile")

    recipe = {"optimizer": "MuonH", "lr": 0.03, "weight_decay": 0.0, "max_grad_norm": 0.0}
    config = init_megatron_optim_config(recipe, {})
    assert config.clip_grad == 0.0
    with pytest.raises(ValueError, match="max_grad_norm=0.0"):
        init_megatron_optim_config({**recipe, "max_grad_norm": 1.0}, {})
    with pytest.raises(ValueError, match="max_grad_norm=0.0"):
        init_megatron_optim_config(recipe, {"clip_grad": 1.0})
