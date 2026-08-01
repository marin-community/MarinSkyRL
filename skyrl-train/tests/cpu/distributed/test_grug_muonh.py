"""CPU oracle for the pinned Marin Grug MuonH production recipe."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from skyrl_train.distributed.fsdp_strategy import FSDPStrategy
from skyrl_train.distributed.grug_muonh import GrugMuonH
from skyrl_train.distributed.grug_muonh import _matrix_delta, build_grug_muonh, grug_muonh_route


FIXTURE = Path(__file__).with_name("fixtures") / "grug_muonh_jax_golden.npz"


class _Config(dict):
    __getattr__ = dict.__getitem__


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


def _optimizer(model: nn.Module, fixture):
    return build_grug_muonh(
        model.named_parameters(),
        _Config(
            lr=float(fixture["metadata_shared_lr"]),
            weight_decay=123.0,
            adam_betas=(0.1, 0.2),
            optimizer_kwargs={
                "adam_lr": float(fixture["metadata_adam_lr"]),
                "weight_decay": 456.0,
                "adam_weight_decay": 789.0,
                "muon_weight_decay": 321.0,
            },
        ),
    )


def _assert_close(actual: torch.Tensor, expected, *, muon_bf16: bool = False) -> None:
    # The pinned transform executes Newton--Schulz in BF16. XLA and oneDNN use
    # different reduction orders, while all FP32 state math agrees more tightly.
    if muon_bf16:
        torch.testing.assert_close(actual, _tensor(expected), rtol=3e-3, atol=1.5e-3)
    else:
        torch.testing.assert_close(actual, _tensor(expected), rtol=2e-6, atol=5e-7)


def test_routes():
    with np.load(FIXTURE, allow_pickle=False) as fixture:
        model = _TinyGrug(fixture)
        expected_routes = dict(zip(fixture["metadata_names"].tolist(), fixture["metadata_routes"].tolist()))
        parameters = dict(model.named_parameters())
        actual_routes = {
            identifier: grug_muonh_route(PARAMETER_NAMES[identifier], parameters[PARAMETER_NAMES[identifier]])
            for identifier in PARAMETER_NAMES
        }

    assert actual_routes == expected_routes
    matrix = parameters[PARAMETER_NAMES["q_proj"]]
    assert grug_muonh_route("model.layers.0.attn_gated_norm.down_proj.weight", matrix) == "muonh"
    assert grug_muonh_route("model.layers.0.mlp_gated_norm.up_proj.weight", matrix) == "muonh"
    assert grug_muonh_route("model.final_gated_norm.down_proj.weight", matrix) == "muonh"


def test_matches_pinned_jax_for_three_steps_and_all_states():
    with np.load(FIXTURE, allow_pickle=False) as fixture:
        model = _TinyGrug(fixture)
        optimizer = _optimizer(model, fixture)
        parameters = dict(model.named_parameters())
        routes = dict(zip(fixture["metadata_names"].tolist(), fixture["metadata_routes"].tolist()))

        assert type(optimizer.adam) is torch.optim.Adam
        assert all(group.get("weight_decay", 0.0) == 0.0 for group in optimizer.param_groups)
        assert optimizer.muonh.param_groups[0]["lr"] == pytest.approx(float(fixture["metadata_shared_lr"]))
        assert optimizer.adamh.param_groups[0]["lr"] == pytest.approx(float(fixture["metadata_shared_lr"]))
        assert optimizer.adam.param_groups[0]["lr"] == pytest.approx(float(fixture["metadata_adam_lr"]))

        initial_expert_norm = torch.linalg.vector_norm(parameters[PARAMETER_NAMES["expert"]], dim=(-2, -1))
        for step in range(1, int(fixture["metadata_steps"]) + 1):
            for identifier, parameter_name in PARAMETER_NAMES.items():
                parameters[parameter_name].grad = _tensor(fixture[f"gradient_{step}__{identifier}"])
            optimizer.step()

            for identifier, parameter_name in PARAMETER_NAMES.items():
                parameter = parameters[parameter_name]
                route = routes[identifier]
                _assert_close(
                    parameter,
                    fixture[f"parameter_{step}__{identifier}"],
                    muon_bf16=route == "muonh",
                )
                if route == "muonh":
                    _assert_close(
                        optimizer.muonh.state[parameter]["momentum_buffer"],
                        fixture[f"momentum_{step}__{identifier}"],
                    )
                elif route == "adamh":
                    _assert_close(
                        optimizer.adamh.state[parameter]["exp_avg"], fixture[f"adamh_mu_{step}__{identifier}"]
                    )
                    _assert_close(
                        optimizer.adamh.state[parameter]["exp_avg_sq"],
                        fixture[f"adamh_nu_{step}__{identifier}"],
                    )
                else:
                    _assert_close(optimizer.adam.state[parameter]["exp_avg"], fixture[f"adam_mu_{step}__{identifier}"])
                    _assert_close(
                        optimizer.adam.state[parameter]["exp_avg_sq"],
                        fixture[f"adam_nu_{step}__{identifier}"],
                    )

            expert_norm = torch.linalg.vector_norm(parameters[PARAMETER_NAMES["expert"]], dim=(-2, -1))
            _assert_close(expert_norm, fixture[f"expert_norm_{step}"], muon_bf16=True)
            torch.testing.assert_close(expert_norm, initial_expert_norm, rtol=2e-5, atol=2e-6)


def test_scheduler_preserves_two_learning_rate_tracks():
    with np.load(FIXTURE, allow_pickle=False) as fixture:
        optimizer = _optimizer(_TinyGrug(fixture), fixture)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 0.5**step)
        optimizer.step()
        scheduler.step()

    assert optimizer.muonh.param_groups[0]["lr"] == pytest.approx(0.015)
    assert optimizer.adamh.param_groups[0]["lr"] == pytest.approx(0.015)
    assert optimizer.adam.param_groups[0]["lr"] == pytest.approx(0.002)


def test_fp32_parameter_gradient_update_and_state_contract():
    with np.load(FIXTURE, allow_pickle=False) as fixture:
        model = _TinyGrug(fixture)
        optimizer = _optimizer(model, fixture)
        parameters = dict(model.named_parameters())

        for step in (1, 2):
            before = {name: parameter.detach().clone() for name, parameter in parameters.items()}
            for identifier, parameter_name in PARAMETER_NAMES.items():
                parameter = parameters[parameter_name]
                parameter.grad = _tensor(fixture[f"gradient_{step}__{identifier}"])
                assert parameter.dtype == torch.float32
                assert parameter.grad.dtype == torch.float32
            optimizer.step()

            for name, parameter in parameters.items():
                update = parameter.detach() - before[name]
                assert update.dtype == torch.float32
                assert parameter.dtype == torch.float32
                for value in optimizer.state[parameter].values():
                    if isinstance(value, torch.Tensor) and value.is_floating_point():
                        assert value.dtype == torch.float32

        matrix = parameters[PARAMETER_NAMES["q_proj"]]
        delta = _matrix_delta(
            matrix,
            torch.ones_like(matrix),
            lr=0.03,
            muon_steps=5,
            clamp_final_norm=True,
        )
        assert delta.dtype == torch.float32


def test_fsdp_strategy_selects_only_explicit_muonh(monkeypatch):
    with np.load(FIXTURE, allow_pickle=False) as fixture:
        model = _TinyGrug(fixture)
        config = _Config(
            optimizer="MuonH",
            lr=0.03,
            weight_decay=123.0,
            adam_betas=(0.1, 0.2),
            max_grad_norm=0.0,
            offload_after_step=False,
            num_warmup_steps=0,
            scheduler="constant",
            optimizer_kwargs={"adam_lr": 0.004},
        )
        strategy = FSDPStrategy(
            fsdp_config=_Config(cpu_offload=False),
            optimizer_config=config,
            model_config=None,
            fsdp_strategy="fsdp2",
            num_training_steps=3,
        )
        monkeypatch.setattr(strategy, "_fsdp_init_model", lambda selected_model, **_kwargs: selected_model)
        _, optimizer, _ = strategy._fsdp_init_train_model(model, optimizer=None, scheduler=None)

    assert isinstance(optimizer, GrugMuonH)
    assert strategy.is_muonh_optimizer
    assert strategy.collect_optimizer_metrics


def test_optimizer_measurement_is_off_for_default_adamw(monkeypatch):
    config = _Config(
        optimizer="AdamW",
        lr=0.03,
        weight_decay=0.0,
        adam_betas=(0.9, 0.95),
        max_grad_norm=0.0,
        offload_after_step=False,
    )
    strategy = FSDPStrategy(
        fsdp_config=_Config(cpu_offload=False),
        optimizer_config=config,
        model_config=None,
        fsdp_strategy="fsdp2",
    )
    model = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)

    synchronize_calls = []
    optimizer_step_calls = []
    monkeypatch.setattr(optimizer, "step", lambda: optimizer_step_calls.append(True))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: synchronize_calls.append(True))
    strategy.optimizer_step(optimizer, model, scheduler=None)

    assert not strategy.is_muonh_optimizer
    assert not strategy.collect_optimizer_metrics
    assert optimizer_step_calls == [True]
    assert strategy.last_optimizer_step_seconds is None
    assert synchronize_calls == []


def test_optimizer_measurement_can_be_enabled_for_control(monkeypatch):
    config = _Config(
        optimizer="AdamW",
        lr=0.03,
        weight_decay=0.0,
        adam_betas=(0.9, 0.95),
        max_grad_norm=0.0,
        offload_after_step=False,
        collect_optimizer_metrics=True,
    )
    strategy = FSDPStrategy(
        fsdp_config=_Config(cpu_offload=False),
        optimizer_config=config,
        model_config=None,
        fsdp_strategy="fsdp2",
    )
    model = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)

    synchronize_calls = []
    optimizer_step_calls = []
    times = iter((10.0, 10.5))
    monkeypatch.setattr(optimizer, "step", lambda: optimizer_step_calls.append(True))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: synchronize_calls.append(True))
    monkeypatch.setattr("skyrl_train.distributed.fsdp_strategy.time.perf_counter", lambda: next(times))
    strategy.optimizer_step(optimizer, model, scheduler=None)

    assert not strategy.is_muonh_optimizer
    assert strategy.collect_optimizer_metrics
    assert optimizer_step_calls == [True]
    assert strategy.last_optimizer_step_seconds == 0.5
    assert synchronize_calls == [True, True]


def test_fsdp_strategy_rejects_muonh_with_expert_parallelism(monkeypatch):
    with np.load(FIXTURE, allow_pickle=False) as fixture:
        model = _TinyGrug(fixture)
        config = _Config(
            optimizer="MuonH",
            lr=0.03,
            weight_decay=0.0,
            max_grad_norm=0.0,
            offload_after_step=False,
            optimizer_kwargs={},
        )
        strategy = FSDPStrategy(
            fsdp_config=_Config(cpu_offload=False, expert_model_parallel_size=2),
            optimizer_config=config,
            model_config=None,
            fsdp_strategy="fsdp2",
            num_training_steps=3,
        )
        monkeypatch.setattr(strategy, "_fsdp_init_model", lambda selected_model, **_kwargs: selected_model)
        with pytest.raises(ValueError, match="expert_model_parallel_size=1"):
            strategy._fsdp_init_train_model(model, optimizer=None, scheduler=None)


def test_checkpoint_resume_next_step_matches_uninterrupted():
    with np.load(FIXTURE, allow_pickle=False) as fixture:
        uninterrupted_model = _TinyGrug(fixture)
        uninterrupted_optimizer = _optimizer(uninterrupted_model, fixture)
        uninterrupted_scheduler = torch.optim.lr_scheduler.LambdaLR(
            uninterrupted_optimizer, lr_lambda=lambda step: 0.8**step
        )
        uninterrupted_parameters = dict(uninterrupted_model.named_parameters())

        for identifier, parameter_name in PARAMETER_NAMES.items():
            uninterrupted_parameters[parameter_name].grad = _tensor(fixture[f"gradient_1__{identifier}"])
        uninterrupted_optimizer.step()
        uninterrupted_scheduler.step()

        model_state = {name: value.detach().clone() for name, value in uninterrupted_model.state_dict().items()}
        optimizer_state = copy.deepcopy(uninterrupted_optimizer.state_dict())
        scheduler_state = copy.deepcopy(uninterrupted_scheduler.state_dict())

        resumed_model = _TinyGrug(fixture)
        resumed_model.load_state_dict(model_state)
        resumed_optimizer = _optimizer(resumed_model, fixture)
        resumed_scheduler = torch.optim.lr_scheduler.LambdaLR(resumed_optimizer, lr_lambda=lambda step: 0.8**step)
        resumed_optimizer.load_state_dict(optimizer_state)
        resumed_scheduler.load_state_dict(scheduler_state)
        resumed_parameters = dict(resumed_model.named_parameters())

        # Two post-resume steps make the first scheduler update observable to
        # the child optimizers and pin the composite param-group refresh.
        for step in (2, 3):
            for identifier, parameter_name in PARAMETER_NAMES.items():
                gradient = _tensor(fixture[f"gradient_{step}__{identifier}"])
                uninterrupted_parameters[parameter_name].grad = gradient.clone()
                resumed_parameters[parameter_name].grad = gradient.clone()
            uninterrupted_optimizer.step()
            resumed_optimizer.step()
            uninterrupted_scheduler.step()
            resumed_scheduler.step()

    for parameter_name in uninterrupted_parameters:
        torch.testing.assert_close(
            resumed_parameters[parameter_name],
            uninterrupted_parameters[parameter_name],
            rtol=0,
            atol=0,
        )
    assert resumed_scheduler.state_dict() == uninterrupted_scheduler.state_dict()
