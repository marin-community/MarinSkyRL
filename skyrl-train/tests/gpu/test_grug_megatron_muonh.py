from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from megatron.core import parallel_state
from omegaconf import OmegaConf
from skyrl_train.distributed.megatron.grug_muonh import GrugMegatronMuonH
from skyrl_train.distributed.megatron.megatron_utils import offload_megatron_optimizer, load_megatron_optimizer
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
            num_query_groups=2,
            kv_channels=3,
            tensor_model_parallel_size=1,
            expert_tensor_parallel_size=1,
        )
        self.embedding = nn.Embedding(8, 4, device="cuda", dtype=torch.bfloat16)
        self.hidden = nn.Linear(4, 4, bias=False, device="cuda", dtype=torch.bfloat16)
        self.output_layer = nn.Linear(4, 8, bias=False, device="cuda", dtype=torch.bfloat16)
        self.decoder = nn.Module()
        self.decoder.layers = nn.ModuleList([nn.Module()])
        layer = self.decoder.layers[0]
        layer.self_attention = nn.Module()
        layer.self_attention.linear_qkv = nn.Linear(4, 18, bias=False, device="cuda", dtype=torch.bfloat16)
        layer.mlp = nn.Module()
        layer.mlp.shared_experts = nn.Module()
        layer.mlp.shared_experts.linear_fc1 = nn.Linear(4, 12, bias=False, device="cuda", dtype=torch.bfloat16)


def test_megatron_wrapper_routes_updates_and_restores_muonh_state(distributed_parallel_state) -> None:
    torch.manual_seed(17)
    model = _TinyGrug()
    path = Path(__file__).parents[1] / "cpu/distributed/fixtures/grug_muonh_jax_golden.npz"
    with np.load(path, allow_pickle=False) as archive:
        golden = {name: torch.from_numpy(value.copy()).cuda() for name, value in archive.items() if "__" in name}

    def fused_values(prefix: str) -> tuple[torch.Tensor, torch.Tensor]:
        query = golden[f"{prefix}__q_proj"]
        key = golden[f"{prefix}__shared"]
        value = golden[f"{prefix}__expert"][0]
        qkv = torch.cat([part.reshape(2, 3, 4) for part in (query, key, value)], dim=1).reshape(18, 4)
        return qkv, torch.cat((query, key))

    fused_parameters = (
        model.decoder.layers[0].self_attention.linear_qkv.weight,
        model.decoder.layers[0].mlp.shared_experts.linear_fc1.weight,
    )
    with torch.no_grad():
        for parameter, value in zip(fused_parameters, fused_values("initial")):
            parameter.copy_(value)
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
    for parameter, gradient in zip(fused_parameters, fused_values("gradient_1")):
        parameter.grad = gradient.to(parameter.dtype)
    reference_parameter.grad = torch.full_like(reference_parameter, 0.125)
    reference_optimizer.step()
    success, _, _ = optimizer.step()
    assert success
    for parameter, expected_value in zip(fused_parameters, fused_values("parameter_1")):
        torch.testing.assert_close(parameter.float(), expected_value, rtol=5e-3, atol=3e-3)
    assert all(not torch.equal(parameter, initial[name]) for name, parameter in model.named_parameters())
    first_step_weights = copy.deepcopy(model.state_dict())
    base.initialize_state()
    state = copy.deepcopy(optimizer.state_dict())
    assert {"momentum_buffer", "exp_avg", "exp_avg_sq", "step"}.issubset(
        {key for parameter_state in base.state.values() for key in parameter_state}
    )
    wrapped_optimizer = optimizer.chained_optimizers[0]
    offload_megatron_optimizer(optimizer)
    assert all(
        parameter.device.type == "cpu" for group in wrapped_optimizer.fp32_from_float16_groups for parameter in group
    )
    assert all(
        value.device.type == "cpu"
        for parameter_state in base.state.values()
        for name, value in parameter_state.items()
        if name in {"momentum_buffer", "exp_avg", "exp_avg_sq"}
    )
    load_megatron_optimizer(optimizer)
    assert all(
        parameter.device.type == "cuda" for group in wrapped_optimizer.fp32_from_float16_groups for parameter in group
    )
    assert all(
        value.device.type == "cuda"
        for parameter_state in base.state.values()
        for name, value in parameter_state.items()
        if name in {"momentum_buffer", "exp_avg", "exp_avg_sq"}
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
    checkpoint_state = copy.deepcopy(state)
    checkpoint_state["optimizer"]["state"]["common_step"] = torch.tensor(1)
    restored_optimizer.load_state_dict(checkpoint_state)
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
    config = OmegaConf.create({"lr": 0.03, "num_warmup_steps": 2, "lr_warmup_init": 0.006, "weight_decay": 0.0})
    scheduler = get_megatron_optimizer_param_scheduler(optimizer, config, num_training_steps=3)
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx([0.006, 0.0008])
    scheduler.step(1)
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx([0.018, 0.0024])
    scheduler.step(1)
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx([0.03, 0.004])


def test_muonh_checks_effective_megatron_weight_decay() -> None:
    policy = {"optimizer": "MuonH", "lr": 0.03, "weight_decay": 0.0}
    with pytest.raises(ValueError, match="weight_decay=0"):
        init_megatron_optim_config(policy, {"weight_decay": 0.01})
    init_megatron_optim_config({**policy, "weight_decay": 0.01}, {"weight_decay": 0.0})
