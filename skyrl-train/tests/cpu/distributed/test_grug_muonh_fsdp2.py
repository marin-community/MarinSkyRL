"""Two-rank CPU/Gloo FSDP2 value and checkpoint test for Grug MuonH."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor, distribute_tensor
from torch.distributed.tensor.placement_types import Shard
from transformers import PretrainedConfig

from skyrl_train.distributed.fsdp_strategy import FSDPStrategy
from skyrl_train.distributed.grug_muonh import build_grug_muonh, grug_muonh_route
from skyrl_train.distributed.utils import get_free_port


class _Config(dict):
    __getattr__ = dict.__getitem__


class _Weight(nn.Module):
    def __init__(self, value: torch.Tensor) -> None:
        super().__init__()
        self.weight = nn.Parameter(value.clone())


class _FSDPTinyGrug(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dense = _Weight(torch.linspace(-0.7, 0.9, 24).reshape(6, 4))
        self.experts = nn.Parameter(torch.linspace(-0.8, 0.6, 48).reshape(2, 6, 4))
        self.lm_head = _Weight(torch.linspace(-0.65, 0.85, 20).reshape(5, 4))
        self.bias = nn.Parameter(torch.linspace(-0.2, 0.3, 6))
        self.config = PretrainedConfig()


def _optimizer_config() -> _Config:
    return _Config(
        optimizer="MuonH",
        lr=0.03,
        weight_decay=999.0,
        adam_betas=(0.1, 0.2),
        max_grad_norm=0.0,
        offload_after_step=False,
        num_warmup_steps=0,
        scheduler="constant",
        optimizer_kwargs={"adam_lr": 0.004},
    )


def _full(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor


def _set_gradients(model: nn.Module, gradients: dict[str, torch.Tensor]) -> None:
    for name, parameter in model.named_parameters():
        gradient = gradients[name]
        if isinstance(parameter, DTensor):
            parameter.grad = distribute_tensor(
                gradient,
                parameter.device_mesh,
                parameter.placements,
                src_data_rank=None,
            )
        else:
            parameter.grad = gradient.clone()
        assert parameter.dtype == torch.float32
        assert parameter.grad.dtype == torch.float32


def _state_tensors(optimizer, name: str, parameter: torch.Tensor) -> dict[str, torch.Tensor]:
    route = grug_muonh_route(name, parameter)
    if route == "muonh":
        return {"momentum_buffer": optimizer.muonh.state[parameter]["momentum_buffer"]}
    if route == "adamh":
        state = optimizer.adamh.state[parameter]
    else:
        state = optimizer.adam.state[parameter]
    return {"exp_avg": state["exp_avg"], "exp_avg_sq": state["exp_avg_sq"]}


def _assert_matches_unsharded(sharded_model, sharded_optimizer, reference_model, reference_optimizer) -> None:
    sharded_parameters = dict(sharded_model.named_parameters())
    reference_parameters = dict(reference_model.named_parameters())
    assert sharded_parameters.keys() == reference_parameters.keys()
    for name in sharded_parameters:
        sharded_parameter = sharded_parameters[name]
        reference_parameter = reference_parameters[name]
        assert sharded_parameter.dtype == torch.float32
        assert reference_parameter.dtype == torch.float32
        torch.testing.assert_close(_full(sharded_parameter), reference_parameter, rtol=2e-6, atol=5e-7)
        sharded_state = _state_tensors(sharded_optimizer, name, sharded_parameter)
        reference_state = _state_tensors(reference_optimizer, name, reference_parameter)
        assert sharded_state.keys() == reference_state.keys()
        for key in sharded_state:
            if sharded_state[key].is_floating_point():
                assert sharded_state[key].dtype == torch.float32
                assert reference_state[key].dtype == torch.float32
            torch.testing.assert_close(_full(sharded_state[key]), reference_state[key], rtol=2e-6, atol=5e-7)

    expert_norm = torch.linalg.vector_norm(_full(sharded_parameters["experts"]), dim=(-2, -1))
    reference_expert_norm = torch.linalg.vector_norm(reference_parameters["experts"], dim=(-2, -1))
    torch.testing.assert_close(expert_norm, reference_expert_norm, rtol=2e-6, atol=5e-7)


def _snapshot(model, optimizer) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, torch.Tensor]]]:
    parameters = {}
    states = {}
    for name, parameter in model.named_parameters():
        parameters[name] = _full(parameter).detach().clone()
        states[name] = {
            key: _full(value).detach().clone() for key, value in _state_tensors(optimizer, name, parameter).items()
        }
    return parameters, states


def _assert_snapshot(model, optimizer, expected) -> None:
    expected_parameters, expected_states = expected
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(_full(parameter), expected_parameters[name], rtol=0, atol=0)
        actual_state = _state_tensors(optimizer, name, parameter)
        for key, expected_value in expected_states[name].items():
            torch.testing.assert_close(_full(actual_state[key]), expected_value, rtol=0, atol=0)


def _worker(rank: int, world_size: int, port: int, checkpoint_dir: str) -> None:
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
        init_method=f"tcp://127.0.0.1:{port}",
    )
    try:
        torch.manual_seed(20260730)
        reference_model = _FSDPTinyGrug()
        sharded_model = _FSDPTinyGrug()
        mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("fsdp",))
        fully_shard(sharded_model, mesh=mesh, reshard_after_forward=False)

        sharded_parameters = dict(sharded_model.named_parameters())
        assert isinstance(sharded_parameters["dense.weight"], DTensor)
        assert isinstance(sharded_parameters["experts"], DTensor)
        assert sharded_parameters["dense.weight"].placements == (Shard(0),)
        assert sharded_parameters["experts"].placements == (Shard(0),)

        config = _optimizer_config()
        sharded_optimizer = build_grug_muonh(sharded_model.named_parameters(), config)
        reference_optimizer = build_grug_muonh(reference_model.named_parameters(), config)
        sharded_scheduler = torch.optim.lr_scheduler.LambdaLR(sharded_optimizer, lr_lambda=lambda step: 0.8**step)
        reference_scheduler = torch.optim.lr_scheduler.LambdaLR(reference_optimizer, lr_lambda=lambda step: 0.8**step)

        generator = torch.Generator().manual_seed(20260730)
        gradients = [
            {
                name: torch.randn(parameter.shape, generator=generator) + 0.05 * (step + 1)
                for name, parameter in reference_model.named_parameters()
            }
            for step in range(3)
        ]

        _set_gradients(sharded_model, gradients[0])
        _set_gradients(reference_model, gradients[0])
        sharded_optimizer.step()
        reference_optimizer.step()
        sharded_scheduler.step()
        reference_scheduler.step()
        _assert_matches_unsharded(sharded_model, sharded_optimizer, reference_model, reference_optimizer)

        strategy = FSDPStrategy(
            fsdp_config=_Config(cpu_offload=False),
            optimizer_config=config,
            model_config=SimpleNamespace(lora=SimpleNamespace(rank=0)),
            fsdp_strategy="fsdp2",
        )
        strategy.world_size = world_size
        strategy.save_checkpoint(
            sharded_model,
            checkpoint_dir,
            node_local_rank=rank,
            optimizer=sharded_optimizer,
            scheduler=sharded_scheduler,
            client_state={"completed_steps": 1},
        )

        _set_gradients(sharded_model, gradients[1])
        _set_gradients(reference_model, gradients[1])
        sharded_optimizer.step()
        reference_optimizer.step()
        sharded_scheduler.step()
        reference_scheduler.step()
        _assert_matches_unsharded(sharded_model, sharded_optimizer, reference_model, reference_optimizer)
        expected_next_step = _snapshot(sharded_model, sharded_optimizer)
        expected_scheduler = sharded_scheduler.state_dict()

        _set_gradients(sharded_model, gradients[2])
        sharded_optimizer.step()
        sharded_scheduler.step()

        _, client_state = strategy.load_checkpoint(
            sharded_model,
            checkpoint_dir,
            optimizer=sharded_optimizer,
            scheduler=sharded_scheduler,
        )
        assert client_state["client_state"] == {"completed_steps": 1}
        _set_gradients(sharded_model, gradients[1])
        sharded_optimizer.step()
        sharded_scheduler.step()
        _assert_snapshot(sharded_model, sharded_optimizer, expected_next_step)
        assert sharded_scheduler.state_dict() == expected_scheduler
    finally:
        dist.destroy_process_group()


def test_two_rank_fsdp2_values_and_checkpoint(tmp_path: Path):
    mp.spawn(
        _worker,
        args=(2, get_free_port(), str(tmp_path / "checkpoint")),
        nprocs=2,
        join=True,
    )
