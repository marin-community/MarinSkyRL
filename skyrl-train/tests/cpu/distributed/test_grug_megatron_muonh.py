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
