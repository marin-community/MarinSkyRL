"""CPU oracle for the pinned Marin Grug MuonH production recipe."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from skyrl_train.config.utils import get_default_config
from skyrl_train.distributed.fsdp_strategy import FSDPStrategy, resolve_fsdp_parameter_storage_dtype
from skyrl_train.distributed.grug_muonh import GrugMuonH
from skyrl_train.distributed.grug_muonh import build_grug_muonh, grug_muonh_route
from skyrl_train.distributed.megatron.grug_muonh import (
    MegatronGrugMuonH,
    _adamh_direction_in_grad_,
    megatron_grug_route,
)


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
            weight_decay=0.0,
            adam_betas=(0.9, 0.95),
            optimizer_kwargs={
                "adam_lr": float(fixture["metadata_adam_lr"]),
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


def test_routes_and_three_step_jax_oracle():
    with np.load(FIXTURE, allow_pickle=False) as fixture:
        model = _TinyGrug(fixture)
        optimizer = _optimizer(model, fixture)
        parameters = dict(model.named_parameters())
        routes = dict(zip(fixture["metadata_names"].tolist(), fixture["metadata_routes"].tolist()))

        actual_routes = {
            identifier: grug_muonh_route(PARAMETER_NAMES[identifier], parameters[PARAMETER_NAMES[identifier]])
            for identifier in PARAMETER_NAMES
        }
        assert actual_routes == routes
        matrix = parameters[PARAMETER_NAMES["q_proj"]]
        assert grug_muonh_route("model.layers.0.attn_gated_norm.down_proj.weight", matrix) == "muonh"
        assert grug_muonh_route("model.layers.0.mlp_gated_norm.up_proj.weight", matrix) == "muonh"
        assert grug_muonh_route("model.final_gated_norm.down_proj.weight", matrix) == "muonh"

        assert type(optimizer.adam) is torch.optim.Adam
        assert all(group.get("weight_decay", 0.0) == 0.0 for group in optimizer.param_groups)
        assert optimizer.muonh.param_groups[0]["lr"] == pytest.approx(float(fixture["metadata_shared_lr"]))
        assert optimizer.adamh.param_groups[0]["lr"] == pytest.approx(float(fixture["metadata_shared_lr"]))
        assert optimizer.adam.param_groups[0]["lr"] == pytest.approx(float(fixture["metadata_adam_lr"]))

        initial_expert_norm = torch.linalg.vector_norm(parameters[PARAMETER_NAMES["expert"]], dim=(-2, -1))
        for step in range(1, int(fixture["metadata_steps"]) + 1):
            for identifier, parameter_name in PARAMETER_NAMES.items():
                parameter = parameters[parameter_name]
                parameter.grad = _tensor(fixture[f"gradient_{step}__{identifier}"])
                assert parameter.dtype == torch.float32
                assert parameter.grad.dtype == torch.float32
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
                    assert optimizer.muonh.state[parameter]["momentum_buffer"].dtype == torch.float32
                    _assert_close(
                        optimizer.muonh.state[parameter]["momentum_buffer"],
                        fixture[f"momentum_{step}__{identifier}"],
                    )
                elif route == "adamh":
                    assert optimizer.adamh.state[parameter]["exp_avg"].dtype == torch.float32
                    assert optimizer.adamh.state[parameter]["exp_avg_sq"].dtype == torch.float32
                    _assert_close(
                        optimizer.adamh.state[parameter]["exp_avg"], fixture[f"adamh_mu_{step}__{identifier}"]
                    )
                    _assert_close(
                        optimizer.adamh.state[parameter]["exp_avg_sq"],
                        fixture[f"adamh_nu_{step}__{identifier}"],
                    )
                else:
                    assert optimizer.adam.state[parameter]["exp_avg"].dtype == torch.float32
                    assert optimizer.adam.state[parameter]["exp_avg_sq"].dtype == torch.float32
                    _assert_close(optimizer.adam.state[parameter]["exp_avg"], fixture[f"adam_mu_{step}__{identifier}"])
                    _assert_close(
                        optimizer.adam.state[parameter]["exp_avg_sq"],
                        fixture[f"adam_nu_{step}__{identifier}"],
                    )

            expert_norm = torch.linalg.vector_norm(parameters[PARAMETER_NAMES["expert"]], dim=(-2, -1))
            _assert_close(expert_norm, fixture[f"expert_norm_{step}"], muon_bf16=True)
            torch.testing.assert_close(expert_norm, initial_expert_norm, rtol=2e-5, atol=2e-6)


def test_muonh_routes_default_to_master_learning_rate():
    with np.load(FIXTURE, allow_pickle=False) as fixture:
        model = _TinyGrug(fixture)
        master_learning_rate = 1e-5
        optimizer = build_grug_muonh(
            model.named_parameters(),
            _Config(
                lr=master_learning_rate,
                weight_decay=0.0,
                adam_betas=(0.9, 0.95),
                optimizer_kwargs={},
            ),
        )

    assert {group["lr"] for group in optimizer.param_groups} == {master_learning_rate}


def test_megatron_adamh_reuses_gradient_across_scratch_chunks_without_changing_direction():
    torch.manual_seed(11)
    shape = (1025, 4096)  # Just over the 16 MiB scratch chunk boundary.
    exp_avg = torch.randn(shape)
    exp_avg_sq = torch.rand(shape).add_(0.01)
    gradient = torch.empty_like(exp_avg)
    expected = (exp_avg / (1 - 0.9**3)) / ((exp_avg_sq / (1 - 0.95**3)).sqrt() + 1e-8)

    _adamh_direction_in_grad_(gradient, exp_avg, exp_avg_sq, step=3, betas=(0.9, 0.95), eps=1e-8)

    torch.testing.assert_close(gradient, expected, rtol=0, atol=0)


@pytest.mark.parametrize("offload_momentum", [False, True])
def test_megatron_muonh_matches_independent_jax_steps_after_own_state_resume(offload_momentum):
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
                offload_momentum=offload_momentum,
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
    reference_optimizer = GrugMuonH(reference, [torch.nn.Parameter(torch.ones(2, 2))], [], lr=0.03)

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


def test_fsdp_parameter_storage_dtype_defaults_and_overrides_optimizer():
    cfg = get_default_config()
    policy = cfg.trainer.policy.optimizer_config
    critic = cfg.trainer.critic.optimizer_config

    assert resolve_fsdp_parameter_storage_dtype(policy.optimizer, policy.fsdp_parameter_storage_dtype) == torch.bfloat16
    assert resolve_fsdp_parameter_storage_dtype(critic.optimizer, critic.fsdp_parameter_storage_dtype) == torch.bfloat16

    policy.optimizer = "MuonH"
    assert resolve_fsdp_parameter_storage_dtype(policy.optimizer, policy.fsdp_parameter_storage_dtype) == torch.float32
    policy.fsdp_parameter_storage_dtype = "bfloat16"
    assert resolve_fsdp_parameter_storage_dtype(policy.optimizer, policy.fsdp_parameter_storage_dtype) == torch.bfloat16

    critic.fsdp_parameter_storage_dtype = "float32"
    assert resolve_fsdp_parameter_storage_dtype(critic.optimizer, critic.fsdp_parameter_storage_dtype) == torch.float32

    with pytest.raises(ValueError, match="must be float32 or bfloat16"):
        resolve_fsdp_parameter_storage_dtype("AdamW", "float16")


def test_fsdp_strategy_selects_only_explicit_muonh(monkeypatch):
    with np.load(FIXTURE, allow_pickle=False) as fixture:
        model = _TinyGrug(fixture)
        config = _Config(
            optimizer="MuonH",
            lr=0.03,
            weight_decay=0.0,
            adam_betas=(0.9, 0.95),
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


def test_muonh_rejects_nonzero_weight_decay():
    with np.load(FIXTURE, allow_pickle=False) as fixture:
        model = _TinyGrug(fixture)
        config = _Config(
            lr=0.03,
            weight_decay=0.01,
            adam_betas=(0.9, 0.95),
            optimizer_kwargs={},
        )

        with pytest.raises(ValueError, match="requires weight_decay=0"):
            build_grug_muonh(model.named_parameters(), config)


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

        malformed_optimizer = _optimizer(_TinyGrug(fixture), fixture)
        malformed_state = copy.deepcopy(optimizer_state)
        del malformed_state["adam"]
        with pytest.raises(ValueError, match="checkpoint Adam state"):
            malformed_optimizer.load_state_dict(malformed_state)

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

        for optimizer in (uninterrupted_optimizer, resumed_optimizer):
            assert optimizer.muonh.param_groups[0]["lr"] == pytest.approx(float(fixture["metadata_shared_lr"]) * 0.8**3)
            assert optimizer.adamh.param_groups[0]["lr"] == pytest.approx(float(fixture["metadata_shared_lr"]) * 0.8**3)
            assert optimizer.adam.param_groups[0]["lr"] == pytest.approx(float(fixture["metadata_adam_lr"]) * 0.8**3)

    for parameter_name in uninterrupted_parameters:
        torch.testing.assert_close(
            resumed_parameters[parameter_name],
            uninterrupted_parameters[parameter_name],
            rtol=0,
            atol=0,
        )
    assert resumed_scheduler.state_dict() == uninterrupted_scheduler.state_dict()
