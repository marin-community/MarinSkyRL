from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
from megatron.core import parallel_state
from omegaconf import OmegaConf
from skyrl_train.distributed.megatron.grug_muonh import GrugMegatronMuonH
from skyrl_train.distributed.megatron.optimizer import (
    get_megatron_optimizer,
    get_megatron_optimizer_param_scheduler,
    init_megatron_optim_config,
)
from torch import nn


@pytest.fixture
def distributed_parallel_state():
    assert not torch.distributed.is_initialized()
    torch.distributed.init_process_group("nccl", store=torch.distributed.HashStore(), rank=0, world_size=1)
    try:
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=1,
        )
        yield
    finally:
        try:
            parallel_state.destroy_model_parallel()
        finally:
            torch.distributed.destroy_process_group()


class _TinyGrug(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            num_attention_heads=2,
            num_query_groups=1,
            kv_channels=2,
            tensor_model_parallel_size=1,
        )
        self.embedding = nn.Embedding(8, 4, device="cuda", dtype=torch.bfloat16)
        self.hidden = nn.Linear(4, 4, bias=False, device="cuda", dtype=torch.bfloat16)
        self.output_layer = nn.Linear(4, 8, bias=False, device="cuda", dtype=torch.bfloat16)
        self.decoder = nn.Module()
        self.decoder.layers = nn.ModuleList([nn.Module()])
        layer = self.decoder.layers[0]
        layer.self_attention = nn.Module()
        layer.self_attention.linear_qkv = nn.Linear(4, 8, bias=False, device="cuda", dtype=torch.bfloat16)
        layer.mlp = nn.Module()
        layer.mlp.shared_experts = nn.Module()
        layer.mlp.shared_experts.linear_fc1 = nn.Linear(4, 6, bias=False, device="cuda", dtype=torch.bfloat16)


def test_megatron_wrapper_routes_updates_and_restores_muonh_state(distributed_parallel_state) -> None:
    torch.manual_seed(17)
    model = _TinyGrug()
    config = init_megatron_optim_config(
        {
            "optimizer": "MuonH",
            "lr": 0.03,
            "weight_decay": 0.0,
            "adam_betas": [0.9, 0.95],
            "optimizer_kwargs": {"adam_lr": 0.004},
            "max_grad_norm": 0.0,
        },
        {"adam_beta1": 0.73, "adam_beta2": 0.82, "adam_eps": 0.005},
    )
    optimizer = get_megatron_optimizer([model], config)
    base = optimizer.optimizer
    assert isinstance(base, GrugMegatronMuonH)
    assert {group["grug_route"] for group in base.param_groups} == {"muonh", "adamh", "adam"}
    assert {group.get("grug_layout") for group in base.param_groups} == {None, "qkv", "gate_up"}
    adam_parameter = next(
        parameter for group in base.param_groups if group["grug_route"] == "adam" for parameter in group["params"]
    )
    reference_parameter = nn.Parameter(adam_parameter.detach().clone())
    reference_optimizer = torch.optim.AdamW(
        [reference_parameter], lr=0.004, betas=(0.73, 0.82), eps=0.005, weight_decay=0.0
    )
    initial = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 0.125)
    reference_parameter.grad = torch.full_like(reference_parameter, 0.125)
    reference_optimizer.step()
    success, _, _ = optimizer.step()
    assert success
    assert all(not torch.equal(parameter, initial[name]) for name, parameter in model.named_parameters())
    first_step_weights = copy.deepcopy(model.state_dict())
    base.initialize_state()
    state = copy.deepcopy(optimizer.state_dict())
    assert {"momentum_buffer", "exp_avg", "exp_avg_sq", "step"}.issubset(
        {key for parameter_state in base.state.values() for key in parameter_state}
    )
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, -0.125)
    reference_parameter.grad = torch.full_like(reference_parameter, -0.125)
    reference_optimizer.step()
    success, _, _ = optimizer.step()
    assert success
    torch.testing.assert_close(adam_parameter, reference_parameter, rtol=1e-6, atol=1e-6)
    expected = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    restored_model = _TinyGrug()
    restored_model.load_state_dict(first_step_weights)
    restored_optimizer = get_megatron_optimizer([restored_model], config)
    restored_optimizer.load_state_dict(state)
    loaded = restored_optimizer.state_dict()
    for saved_group, loaded_group in zip(state["fp32_from_fp16_params"], loaded["fp32_from_fp16_params"]):
        for saved_parameter, loaded_parameter in zip(saved_group, loaded_group):
            torch.testing.assert_close(saved_parameter, loaded_parameter, rtol=0, atol=0)
    saved_states = state["optimizer"]["state"]
    loaded_states = loaded["optimizer"]["state"]
    assert saved_states.keys() == loaded_states.keys()
    for parameter_key in sorted(saved_states):
        saved_parameter_state = saved_states[parameter_key]
        loaded_parameter_state = loaded_states[parameter_key]
        assert saved_parameter_state.keys() == loaded_parameter_state.keys()
        for key, saved_value in saved_parameter_state.items():
            torch.testing.assert_close(saved_value, loaded_parameter_state[key], rtol=0, atol=0)
    assert state["optimizer"]["param_groups"] == loaded["optimizer"]["param_groups"]
    for parameter in restored_model.parameters():
        parameter.grad = torch.full_like(parameter, -0.125)
    restored_optimizer.step()
    for name, parameter in restored_model.named_parameters():
        torch.testing.assert_close(parameter, expected[name], rtol=0, atol=0, msg=lambda error: f"{name}: {error}")


def test_adam_route_keeps_its_rate_when_megatron_scheduler_steps() -> None:
    muon = torch.nn.Parameter(torch.ones(4, 4, device="cuda"))
    adam = torch.nn.Parameter(torch.ones(4, device="cuda"))
    optimizer = GrugMegatronMuonH(
        [{"params": [muon], "grug_route": "muonh"}, {"params": [adam], "grug_route": "adam"}],
        lr=0.03,
        adam_lr=0.004,
    )
    config = OmegaConf.create({"lr": 0.03, "num_warmup_steps": 0, "weight_decay": 0.0})
    scheduler = get_megatron_optimizer_param_scheduler(optimizer, config, num_training_steps=3)
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx([0.03, 0.004])
    scheduler.step(1)
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx([0.03, 0.004])


def test_muonh_checks_effective_megatron_weight_decay() -> None:
    policy = {"optimizer": "MuonH", "lr": 0.03, "weight_decay": 0.0}
    with pytest.raises(ValueError, match="weight_decay=0"):
        init_megatron_optim_config(policy, {"weight_decay": 0.01})
    config = init_megatron_optim_config({**policy, "weight_decay": 0.01}, {"weight_decay": 0.0})
    assert config.weight_decay == 0.0
