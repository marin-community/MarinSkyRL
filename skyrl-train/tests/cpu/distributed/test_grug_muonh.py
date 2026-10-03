"""CPU oracle for the pinned Marin Grug MuonH production recipe."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from skyrl_train.distributed.megatron.grug_muonh import (
    MegatronGrugMuonH,
    _adamh_direction_in_grad_,
    megatron_grug_route,
)


FIXTURE = Path(__file__).with_name("fixtures") / "grug_muonh_jax_golden.npz"


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


@pytest.mark.parametrize("offload_momentum", [False, True])
def test_megatron_muonh_matches_independent_jax_steps_after_own_state_resume(tmp_path, offload_momentum):
    with np.load(FIXTURE, allow_pickle=False) as fixture:
        parameters = {key: torch.nn.Parameter(_tensor(fixture[f"initial__{key}"])) for key in PARAMETER_NAMES}
        routes = dict(zip(fixture["metadata_names"].tolist(), fixture["metadata_routes"].tolist()))

        def optimizers(current):
            groups = [
                {"params": [current[key]], "optimizer": megatron_grug_route(PARAMETER_NAMES[key], current[key])}
                for key in PARAMETER_NAMES
                if routes[key] != "adam"
            ]
            adam_parameters = [current[key] for key in PARAMETER_NAMES if routes[key] == "adam"]
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
                offload_momentum=offload_momentum,
            )
            adam_optimizer = torch.optim.Adam(
                adam_parameters, lr=float(fixture["metadata_adam_lr"]), betas=(0.9, 0.95), eps=1e-8
            )
            return hero_optimizer, adam_optimizer

        hero_optimizer, adam_optimizer = optimizers(parameters)
        for step in range(1, int(fixture["metadata_steps"]) + 1):
            for key, parameter in parameters.items():
                parameter.grad = _tensor(fixture[f"gradient_{step}__{key}"])
            hero_optimizer.step()
            adam_optimizer.step()

            for key, parameter in parameters.items():
                _assert_close(parameter, fixture[f"parameter_{step}__{key}"], muon_bf16=routes[key] == "muonh")
                if routes[key] == "muonh":
                    _assert_close(
                        hero_optimizer.state[parameter]["momentum_buffer"], fixture[f"momentum_{step}__{key}"]
                    )
                else:
                    state = (adam_optimizer if routes[key] == "adam" else hero_optimizer).state[parameter]
                    _assert_close(state["exp_avg"], fixture[f"{routes[key]}_mu_{step}__{key}"])
                    _assert_close(state["exp_avg_sq"], fixture[f"{routes[key]}_nu_{step}__{key}"])
                    assert state["step"].item() == step

            if step == 1:
                checkpoint = tmp_path / "optimizer.pt"
                torch.save(
                    {
                        "parameters": parameters,
                        "hero": hero_optimizer.state_dict(),
                        "adam": adam_optimizer.state_dict(),
                    },
                    checkpoint,
                )
                saved = torch.load(checkpoint, weights_only=True)
                parameters = {key: torch.nn.Parameter(value) for key, value in saved["parameters"].items()}
                hero_optimizer, adam_optimizer = optimizers(parameters)
                hero_optimizer.initialize_state()
                hero_optimizer.load_state_dict(saved["hero"])
                adam_optimizer.load_state_dict(saved["adam"])


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


def test_megatron_muonh_rejects_expert_tensor_shards():
    try:
        from megatron.core.optimizer.emerging_optimizers import _EMERGING_OPTIMIZERS

        from skyrl_train.distributed.megatron.optimizer import _register_grug_muonh
    except ImportError:
        pytest.skip("Megatron Core optimizer is not in the CPU test profile")

    original = dict(_EMERGING_OPTIMIZERS)
    try:
        _register_grug_muonh({"optimizer": "MuonH", "lr": 0.03, "weight_decay": 0.0})
        model = SimpleNamespace(config=SimpleNamespace(tensor_model_parallel_size=1, expert_tensor_parallel_size=2))
        with pytest.raises(ValueError, match="expert tensor parallel size 1"):
            _EMERGING_OPTIMIZERS["grug_muonh"].config_to_kwargs(None, [model], None)
    finally:
        _EMERGING_OPTIMIZERS.clear()
        _EMERGING_OPTIMIZERS.update(original)


def test_megatron_adamh_reuses_gradient_across_scratch_chunks_without_changing_direction():
    torch.manual_seed(11)
    shape = (1025, 4096)  # Just over the 16 MiB scratch chunk boundary.
    exp_avg = torch.randn(shape)
    exp_avg_sq = torch.rand(shape).add_(0.01)
    gradient = torch.empty_like(exp_avg)
    expected = (exp_avg / (1 - 0.9**3)) / ((exp_avg_sq / (1 - 0.95**3)).sqrt() + 1e-8)

    _adamh_direction_in_grad_(gradient, exp_avg, exp_avg_sq, step=3, betas=(0.9, 0.95), eps=1e-8)

    torch.testing.assert_close(gradient, expected, rtol=0, atol=0)
