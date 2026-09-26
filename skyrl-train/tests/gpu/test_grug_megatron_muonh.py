from __future__ import annotations

import copy
import pytest
import torch
from megatron.core import parallel_state
from skyrl_train.distributed.megatron.grug_muonh import GrugMegatronMuonH
from skyrl_train.distributed.megatron.optimizer import get_megatron_optimizer, init_megatron_optim_config
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
        self.embedding = nn.Embedding(8, 4, device="cuda", dtype=torch.bfloat16)
        self.hidden = nn.Linear(4, 4, bias=False, device="cuda", dtype=torch.bfloat16)
        self.output_layer = nn.Linear(4, 8, bias=False, device="cuda", dtype=torch.bfloat16)


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
        {},
    )
    optimizer = get_megatron_optimizer([model], config)
    base = optimizer.optimizer
    assert isinstance(base, GrugMegatronMuonH)
    assert {group["grug_route"] for group in base.param_groups} == {"muonh", "adamh", "adam"}
    initial = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 0.125)
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
    success, _, _ = optimizer.step()
    assert success
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
