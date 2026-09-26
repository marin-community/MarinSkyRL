from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from skyrl_train.distributed.megatron.grug_muonh import GrugMegatronMuonH, grug_muonh_route

FIXTURE = Path(__file__).with_name("fixtures") / "grug_muonh_jax_golden.npz"
NAMES = {
    "embed": "embedding.word_embeddings.weight",
    "q_proj": "decoder.layers.0.self_attention.linear_qkv.weight",
    "attn_gate": "decoder.layers.0.self_attention.attn_gate.weight",
    "router": "decoder.layers.0.mlp.router.weight",
    "expert": "decoder.layers.0.mlp.experts.linear_fc1.weight",
    "shared": "decoder.layers.0.mlp.shared_experts.linear_fc1.weight",
    "gated_norm": "decoder.layers.0.input_gated_norm.down_proj.weight",
    "norm": "decoder.layers.0.input_layernorm.weight",
    "bias": "decoder.layers.0.bias",
    "output": "output_layer.weight",
}


def _tensor(value) -> torch.Tensor:
    return torch.from_numpy(np.asarray(value).copy())


def _optimizer(parameters: dict[str, torch.nn.Parameter], *, lr: float, adam_lr: float) -> GrugMegatronMuonH:
    groups = []
    for identifier, parameter in parameters.items():
        groups.append({"params": [parameter], "grug_route": grug_muonh_route(NAMES[identifier], parameter)})
    return GrugMegatronMuonH(groups, lr=lr, adam_lr=adam_lr)


def test_three_step_jax_oracle_and_checkpoint_resume() -> None:
    with np.load(FIXTURE, allow_pickle=False) as fixture:
        parameters = {
            identifier: torch.nn.Parameter(_tensor(fixture[f"initial__{identifier}"])) for identifier in NAMES
        }
        optimizer = _optimizer(
            parameters,
            lr=float(fixture["metadata_shared_lr"]),
            adam_lr=float(fixture["metadata_adam_lr"]),
        )
        expected_routes = dict(zip(fixture["metadata_names"].tolist(), fixture["metadata_routes"].tolist()))
        assert {
            identifier: grug_muonh_route(NAMES[identifier], parameter) for identifier, parameter in parameters.items()
        } == expected_routes
        assert {group["grug_route"]: group["lr"] for group in optimizer.param_groups} == {
            "muonh": 0.03,
            "adamh": 0.03,
            "adam": 0.004,
        }
        for step in range(1, 4):
            for identifier, parameter in parameters.items():
                parameter.grad = _tensor(fixture[f"gradient_{step}__{identifier}"])
            optimizer.step()
            for identifier, parameter in parameters.items():
                expected = _tensor(fixture[f"parameter_{step}__{identifier}"])
                route = expected_routes[identifier]
                torch.testing.assert_close(
                    parameter,
                    expected,
                    rtol=3e-3 if route == "muonh" else 2e-6,
                    atol=1.5e-3 if route == "muonh" else 5e-7,
                    msg=f"{identifier} step {step}",
                )
            if step == 2:
                resumed_parameters = {
                    identifier: torch.nn.Parameter(parameter.detach().clone())
                    for identifier, parameter in parameters.items()
                }
                resumed = _optimizer(
                    resumed_parameters,
                    lr=float(fixture["metadata_shared_lr"]),
                    adam_lr=float(fixture["metadata_adam_lr"]),
                )
                resumed.initialize_state()
                resumed.load_state_dict(copy.deepcopy(optimizer.state_dict()))
                for identifier, parameter in resumed_parameters.items():
                    parameter.grad = _tensor(fixture[f"gradient_3__{identifier}"])
                resumed.step()
        for identifier, parameter in parameters.items():
            torch.testing.assert_close(resumed_parameters[identifier], parameter, rtol=0, atol=0)


def test_muonh_rejects_nonzero_weight_decay() -> None:
    parameter = torch.nn.Parameter(torch.ones(2, 2))
    with pytest.raises(ValueError, match="weight_decay=0"):
        GrugMegatronMuonH([{"params": [parameter], "weight_decay": 0.01}], lr=0.03)


def test_embedding_gate_route_matches_hf_gated_norm_route() -> None:
    matrix = torch.nn.Parameter(torch.ones(4, 4))
    norm = torch.nn.Parameter(torch.ones(4))
    for projection in ("down_proj", "up_proj"):
        assert grug_muonh_route(f"embed_norm.{projection}.weight", matrix) == "muonh"
        assert grug_muonh_route(f"model.embed_gated_norm.{projection}.weight", matrix) == "muonh"
    assert grug_muonh_route("embed_norm.norm.weight", norm) == "adam"


def test_fused_gate_up_matches_independent_projection_updates() -> None:
    gate = torch.nn.Parameter(torch.linspace(-0.6, 0.8, 12).reshape(3, 4))
    up = torch.nn.Parameter(torch.linspace(0.3, -0.7, 12).reshape(3, 4))
    fused = torch.nn.Parameter(torch.cat((gate.detach(), up.detach())))
    separate_optimizer = GrugMegatronMuonH(
        [{"params": [gate], "grug_route": "muonh"}, {"params": [up], "grug_route": "muonh"}], lr=0.03
    )
    fused_optimizer = GrugMegatronMuonH([{"params": [fused], "grug_route": "muonh", "grug_layout": "gate_up"}], lr=0.03)

    for step in range(2):
        gate.grad = torch.linspace(-0.4, 0.5, 12).reshape(3, 4) + step * 0.1
        up.grad = torch.linspace(0.6, -0.3, 12).reshape(3, 4) - step * 0.1
        fused.grad = torch.cat((gate.grad, up.grad))
        separate_optimizer.step()
        fused_optimizer.step()

    torch.testing.assert_close(fused, torch.cat((gate, up)), rtol=0, atol=0)


def test_interleaved_qkv_matches_independent_projection_updates() -> None:
    num_groups, heads_per_group, head_dim, hidden = 2, 2, 2, 4
    q = torch.nn.Parameter(torch.linspace(-0.8, 0.7, 32).reshape(8, hidden))
    k = torch.nn.Parameter(torch.linspace(0.4, -0.6, 16).reshape(4, hidden))
    v = torch.nn.Parameter(torch.linspace(-0.3, 0.9, 16).reshape(4, hidden))

    def interleave(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (
                query.reshape(num_groups, heads_per_group * head_dim, hidden),
                key.reshape(num_groups, head_dim, hidden),
                value.reshape(num_groups, head_dim, hidden),
            ),
            dim=1,
        ).reshape(-1, hidden)

    fused = torch.nn.Parameter(interleave(q.detach(), k.detach(), v.detach()))
    separate_optimizer = GrugMegatronMuonH(
        [{"params": [parameter], "grug_route": "muonh"} for parameter in (q, k, v)], lr=0.03
    )
    fused_optimizer = GrugMegatronMuonH(
        [{"params": [fused], "grug_route": "muonh", "grug_layout": "qkv"}],
        lr=0.03,
        qkv_num_query_groups=num_groups,
        qkv_heads_per_group=heads_per_group,
        qkv_head_dim=head_dim,
    )

    for step in range(2):
        q.grad = torch.linspace(-0.6, 0.5, q.numel()).reshape_as(q) + step * 0.1
        k.grad = torch.linspace(0.5, -0.4, k.numel()).reshape_as(k) - step * 0.1
        v.grad = torch.linspace(-0.2, 0.7, v.numel()).reshape_as(v) + step * 0.05
        fused.grad = interleave(q.grad, k.grad, v.grad)
        separate_optimizer.step()
        fused_optimizer.step()

    torch.testing.assert_close(fused, interleave(q, k, v), rtol=0, atol=0)


def test_muonh_requires_unsharded_tensor_parallelism() -> None:
    parameter = torch.nn.Parameter(torch.ones(2, 2))
    with pytest.raises(ValueError, match="tensor_model_parallel_size=1"):
        GrugMegatronMuonH([{"params": [parameter]}], lr=0.03, tensor_model_parallel_size=2)
