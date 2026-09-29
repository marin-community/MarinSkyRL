from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from megatron.core import parallel_state
from omegaconf import OmegaConf
from skyrl_train.distributed.megatron.grug_muonh import MegatronGrugMuonH
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
            num_attention_heads=4,
            num_query_groups=2,
            kv_channels=3,
            tensor_model_parallel_size=1,
            expert_tensor_parallel_size=1,
        )
        self.embedding = nn.Embedding(8, 4, device="cuda")
        self.hidden = nn.Linear(4, 4, bias=False, device="cuda")
        self.output_layer = nn.Linear(4, 8, bias=False, device="cuda")
        self.decoder = nn.Module()
        self.decoder.layers = nn.ModuleList([nn.Module()])
        layer = self.decoder.layers[0]
        layer.self_attention = nn.Module()
        layer.self_attention.linear_qkv = nn.Linear(4, 24, bias=False, device="cuda")
        layer.mlp = nn.Module()
        layer.mlp.shared_experts = nn.Module()
        layer.mlp.shared_experts.linear_fc1 = nn.Linear(4, 12, bias=False, device="cuda")


def _recipe(**kwargs):
    return {
        "optimizer": "MuonH",
        "lr": 0.03,
        "weight_decay": 0.0,
        "adam_betas": [0.9, 0.95],
        "optimizer_kwargs": {"adam_lr": 0.004},
        "max_grad_norm": 0.0,
        "num_warmup_steps": 0,
        **kwargs,
    }


def _optimizer(model, recipe):
    optimizer = get_megatron_optimizer([model], init_megatron_optim_config(recipe, {}), grug_optimizer_config=recipe)
    scheduler = get_megatron_optimizer_param_scheduler(optimizer, OmegaConf.create(recipe), num_training_steps=3)
    return optimizer, scheduler


def _reference_optimizers(model):
    layer = model.decoder.layers[0]
    muon = MegatronGrugMuonH(
        [
            {"params": [model.hidden.weight], "optimizer": "grug_muonh"},
            {"params": [model.output_layer.weight], "optimizer": "grug_adamh"},
            {"params": [layer.self_attention.linear_qkv.weight], "optimizer": "grug_muonh_qkv"},
            {"params": [layer.mlp.shared_experts.linear_fc1.weight], "optimizer": "grug_muonh_gate_up"},
        ],
        lr=0.03,
        betas=(0.9, 0.95),
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        eps=1e-8,
        muon_eps=1e-8,
        qkv_split_shapes=(6, 3, 3),
    )
    adam = torch.optim.Adam([model.embedding.weight], lr=0.004, betas=(0.9, 0.95), eps=1e-8)
    return muon, adam


def _gradients(model, amplitude):
    for parameter in model.parameters():
        parameter.grad = torch.linspace(-1, 1, parameter.numel(), device="cuda").reshape_as(parameter) * amplitude


def _step(optimizer, model):
    for parameter in model.parameters():
        parameter.main_grad = parameter.grad
    return optimizer.step()


def _assert_weights(actual, expected):
    for name, parameter in actual.named_parameters():
        reference = expected.get_parameter(name)
        if name == "embedding.weight":
            torch.testing.assert_close(parameter, reference, rtol=1e-6, atol=1e-6)
        else:
            torch.testing.assert_close(parameter, reference, rtol=5e-3, atol=3e-3)


@pytest.mark.parametrize("offload_momentum", [False, True])
def test_native_factory_matches_three_jax_steps_and_adam(distributed_parallel_state, offload_momentum):
    torch.manual_seed(17)
    model = _TinyGrug()
    path = Path(__file__).parents[1] / "cpu/distributed/fixtures/grug_muonh_jax_golden.npz"
    with np.load(path, allow_pickle=False) as archive:
        golden = {name: torch.from_numpy(value.copy()).cuda() for name, value in archive.items() if "__" in name}

    def fused_values(prefix):
        query = golden[f"{prefix}__q_proj"]
        key = golden[f"{prefix}__shared"]
        value = golden[f"{prefix}__expert"][0]
        gqa_query = golden[f"{prefix}__gqa_q_proj"]
        qkv = torch.cat((gqa_query.reshape(2, 6, 4), key.reshape(2, 3, 4), value.reshape(2, 3, 4)), dim=1)
        return qkv.reshape(24, 4), torch.cat((query, key))

    layer = model.decoder.layers[0]
    fused_parameters = (layer.self_attention.linear_qkv.weight, layer.mlp.shared_experts.linear_fc1.weight)
    with torch.no_grad():
        for parameter, value in zip(fused_parameters, fused_values("initial")):
            parameter.copy_(value)
    recipe = _recipe(optimizer_kwargs={"adam_lr": 0.004, "offload_momentum": offload_momentum})
    optimizer, _ = _optimizer(model, recipe)
    reference = nn.Parameter(model.embedding.weight.detach().clone())
    adam = torch.optim.Adam([reference], lr=0.004, betas=(0.9, 0.95), eps=1e-8)
    for step in range(1, 4):
        for parameter in model.parameters():
            parameter.grad = torch.full_like(parameter, 0.125 * step)
        for parameter, gradient in zip(fused_parameters, fused_values(f"gradient_{step}")):
            parameter.grad = gradient.clone()
        reference.grad = torch.full_like(reference, 0.125 * step)
        adam.step()
        success, _, _ = _step(optimizer, model)
        assert success
        for parameter, expected in zip(fused_parameters, fused_values(f"parameter_{step}")):
            torch.testing.assert_close(parameter, expected, rtol=5e-3, atol=3e-3)
        torch.testing.assert_close(model.embedding.weight, reference, rtol=1e-6, atol=1e-6)


def test_native_factory_clips_all_routes_by_the_global_norm(distributed_parallel_state):
    torch.manual_seed(17)
    model = _TinyGrug()
    reference = copy.deepcopy(model)
    optimizer, _ = _optimizer(model, _recipe(max_grad_norm=1.0))
    muon, adam = _reference_optimizers(reference)
    for amplitude in (8.0, -0.5, 4.0):
        _gradients(model, amplitude)
        _gradients(reference, amplitude)
        expected_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
        muon.step()
        adam.step()
        success, norm, _ = _step(optimizer, model)
        assert success
        assert norm > 1.0
        assert norm == pytest.approx(expected_norm.item(), rel=1e-6)
        _assert_weights(model, reference)


def test_route_warmup_and_scheduler_resume_match_weight_updates(distributed_parallel_state):
    torch.manual_seed(17)
    model = _TinyGrug()
    reference = copy.deepcopy(model)
    recipe = _recipe(num_warmup_steps=2, lr_warmup_init=0.006)
    optimizer, scheduler = _optimizer(model, recipe)
    muon, adam = _reference_optimizers(reference)
    for step, (matrix_lr, adam_lr) in enumerate(((0.006, 0.0008), (0.018, 0.0024), (0.03, 0.004))):
        if step == 1:
            state = scheduler.state_dict()
            scheduler.step(1)
            scheduler.load_state_dict(state)
        _gradients(model, step + 1)
        _gradients(reference, step + 1)
        for group in muon.param_groups:
            group["lr"] = matrix_lr
        adam.param_groups[0]["lr"] = adam_lr
        muon.step()
        adam.step()
        success, _, _ = _step(optimizer, model)
        assert success
        _assert_weights(model, reference)
        scheduler.step(1)
