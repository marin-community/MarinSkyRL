from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import Tensor, nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor, distribute_tensor
from torch.distributed.tensor.placement_types import Replicate
from transformers import PretrainedConfig

from skyrl_train.distributed.bf16_adamw import BFloat16AdamW, BFloat16UpdateMode, build_adamw
from skyrl_train.distributed.fsdp_strategy import FSDPStrategy, resolve_fsdp_parameter_storage_dtype
from skyrl_train.distributed.utils import get_free_port


class _Config(dict):
    __getattr__ = dict.__getitem__


def _step(optimizer: torch.optim.Optimizer, parameter: torch.nn.Parameter) -> None:
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _optimizer(parameter: torch.nn.Parameter, mode: BFloat16UpdateMode, *, seed: int = 17) -> BFloat16AdamW:
    optimizer = build_adamw(
        [parameter],
        update_mode=mode,
        seed=seed,
        lr=1e-5,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    assert isinstance(optimizer, BFloat16AdamW)
    return optimizer


def test_stochastic_one_step_moves_the_expected_fraction_by_one_ulp() -> None:
    initial = torch.tensor(0.01, dtype=torch.bfloat16)
    parameter = torch.nn.Parameter(initial.expand(200_000).clone())
    optimizer = _optimizer(parameter, BFloat16UpdateMode.STOCHASTIC)

    reference = torch.nn.Parameter(initial.float().clone())
    reference_optimizer = torch.optim.AdamW([reference], lr=1e-5, weight_decay=0.0)
    _step(reference_optimizer, reference)
    lower = torch.nextafter(initial, torch.tensor(float("-inf"), dtype=torch.bfloat16))
    expected_changed = float((initial.float() - reference.detach()) / (initial.float() - lower.float()))

    _step(optimizer, parameter)

    changed = parameter.detach() != initial
    assert changed.float().mean().item() == pytest.approx(expected_changed, abs=0.01)
    assert torch.all(parameter.detach()[changed] == lower)


@pytest.mark.parametrize("mode", [BFloat16UpdateMode.STOCHASTIC, BFloat16UpdateMode.KAHAN])
def test_low_precision_updates_track_fp32_adamw_over_many_steps(mode: BFloat16UpdateMode) -> None:
    parameter = torch.nn.Parameter(torch.full((4_096,), 0.1, dtype=torch.bfloat16))
    optimizer = _optimizer(parameter, mode)
    reference = torch.nn.Parameter(parameter.detach().float().mean().reshape(()))
    reference_optimizer = torch.optim.AdamW([reference], lr=1e-5, weight_decay=0.0)

    for _ in range(1_000):
        _step(optimizer, parameter)
        _step(reference_optimizer, reference)

    torch.testing.assert_close(parameter.detach().float().mean(), reference.detach(), rtol=0.02, atol=2e-4)


def test_nearest_rounding_preserves_the_current_cancelled_update() -> None:
    parameter = torch.nn.Parameter(torch.full((128,), 0.1, dtype=torch.bfloat16))
    before = parameter.detach().clone()
    optimizer = build_adamw(
        [parameter],
        update_mode=BFloat16UpdateMode.NEAREST,
        seed=17,
        lr=1e-5,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )

    _step(optimizer, parameter)

    assert type(optimizer) is torch.optim.AdamW
    torch.testing.assert_close(parameter, before, rtol=0, atol=0)


@pytest.mark.parametrize("mode", [BFloat16UpdateMode.STOCHASTIC, BFloat16UpdateMode.KAHAN])
def test_checkpoint_resume_matches_uninterrupted_updates(mode: BFloat16UpdateMode) -> None:
    parameter = torch.nn.Parameter(torch.full((1_024,), 0.1, dtype=torch.bfloat16))
    optimizer = _optimizer(parameter, mode, seed=20260823)
    for _ in range(13):
        _step(optimizer, parameter)

    saved_parameter = parameter.detach().clone()
    saved_optimizer = copy.deepcopy(optimizer.state_dict())
    for _ in range(17):
        _step(optimizer, parameter)
    expected = parameter.detach().clone()

    resumed_parameter = torch.nn.Parameter(saved_parameter)
    resumed_optimizer = _optimizer(resumed_parameter, mode, seed=20260823)
    resumed_optimizer.load_state_dict(saved_optimizer)
    for _ in range(17):
        _step(resumed_optimizer, resumed_parameter)

    torch.testing.assert_close(resumed_parameter, expected, rtol=0, atol=0)
    if mode is BFloat16UpdateMode.KAHAN:
        assert resumed_optimizer.state[resumed_parameter]["rounding_residual"].dtype == torch.bfloat16


def test_stochastic_stream_is_seeded_without_mutating_global_rng() -> None:
    global_state = torch.random.get_rng_state()
    first = torch.nn.Parameter(torch.full((4_096,), 0.1, dtype=torch.bfloat16))
    second = torch.nn.Parameter(first.detach().clone())
    other = torch.nn.Parameter(first.detach().clone())

    _step(_optimizer(first, BFloat16UpdateMode.STOCHASTIC, seed=11), first)
    _step(_optimizer(second, BFloat16UpdateMode.STOCHASTIC, seed=11), second)
    _step(_optimizer(other, BFloat16UpdateMode.STOCHASTIC, seed=12), other)

    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert not torch.equal(first, other)
    torch.testing.assert_close(torch.random.get_rng_state(), global_state, rtol=0, atol=0)


def test_stochastic_mode_resumes_an_older_torch_adamw_checkpoint() -> None:
    old_parameter = torch.nn.Parameter(torch.full((128,), 0.1, dtype=torch.bfloat16))
    old_optimizer = torch.optim.AdamW([old_parameter], lr=1e-5, weight_decay=0.0)
    _step(old_optimizer, old_parameter)

    parameter = torch.nn.Parameter(old_parameter.detach().clone())
    optimizer = _optimizer(parameter, BFloat16UpdateMode.STOCHASTIC)
    optimizer.load_state_dict(copy.deepcopy(old_optimizer.state_dict()))
    _step(optimizer, parameter)

    assert optimizer.param_groups[0]["bf16_update_mode"] == "stochastic"
    assert optimizer.param_groups[0]["bf16_update_step"] == 2


def test_checkpoint_rejects_an_explicit_update_mode_change() -> None:
    parameter = torch.nn.Parameter(torch.full((128,), 0.1, dtype=torch.bfloat16))
    stochastic = _optimizer(parameter, BFloat16UpdateMode.STOCHASTIC)
    _step(stochastic, parameter)
    kahan = _optimizer(torch.nn.Parameter(parameter.detach().clone()), BFloat16UpdateMode.KAHAN)

    with pytest.raises(ValueError, match="does not match configured mode"):
        kahan.load_state_dict(stochastic.state_dict())


def test_fp32_parameters_use_torch_adamw_for_every_low_precision_mode() -> None:
    for mode in (BFloat16UpdateMode.STOCHASTIC, BFloat16UpdateMode.KAHAN, BFloat16UpdateMode.FP32_MASTER):
        parameter = torch.nn.Parameter(torch.ones(2, dtype=torch.float32))
        optimizer = build_adamw([parameter], update_mode=mode, seed=17, lr=1e-5)
        assert type(optimizer) is torch.optim.AdamW


def test_fp32_master_mode_controls_fsdp_parameter_storage() -> None:
    mode = BFloat16UpdateMode.FP32_MASTER
    assert resolve_fsdp_parameter_storage_dtype("AdamW", None, mode) is torch.float32
    assert resolve_fsdp_parameter_storage_dtype("AdamW", "float32", mode) is torch.float32
    with pytest.raises(ValueError, match="conflicts with non-FP32"):
        resolve_fsdp_parameter_storage_dtype("AdamW", "bfloat16", mode)


def _fsdp_optimizer_config(**overrides) -> _Config:
    values = {
        "optimizer": "AdamW",
        "fsdp_parameter_storage_dtype": "bfloat16",
        "lr": 1e-5,
        "weight_decay": 0.0,
        "adam_betas": (0.9, 0.999),
        "max_grad_norm": 0.0,
        "offload_after_step": False,
        "num_warmup_steps": 0,
        "scheduler": "constant",
        "optimizer_kwargs": {},
    }
    values.update(overrides)
    return _Config(values)


def test_fsdp_strategy_rejects_an_explicit_mode_for_an_unsupported_optimizer() -> None:
    with pytest.raises(ValueError, match="applies only to AdamW"):
        FSDPStrategy(
            fsdp_config=_Config(cpu_offload=False),
            optimizer_config=_fsdp_optimizer_config(optimizer="SGD", bf16_update_mode="stochastic"),
            fsdp_strategy="fsdp2",
        )


def test_fsdp_strategy_builds_stochastic_adamw_for_bf16_parameters(monkeypatch: pytest.MonkeyPatch) -> None:
    model = torch.nn.Linear(2, 2).to(torch.bfloat16)
    optimizer_config = _fsdp_optimizer_config()
    strategy = FSDPStrategy(
        fsdp_config=_Config(cpu_offload=False),
        optimizer_config=optimizer_config,
        fsdp_strategy="fsdp2",
        seed=29,
        num_training_steps=1,
    )
    monkeypatch.setattr(strategy, "_fsdp_init_model", lambda selected_model, **_kwargs: selected_model)

    _, optimizer, _ = strategy._fsdp_init_train_model(model, optimizer=None, scheduler=None)

    assert isinstance(optimizer, BFloat16AdamW)
    assert optimizer.param_groups[0]["rounding_seed"] == 29


class _TinyBFloat16Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.full((4_096,), 0.1, dtype=torch.bfloat16))
        self.config = PretrainedConfig()


def _set_fsdp_gradient(parameter: Tensor) -> None:
    full_gradient = torch.ones(parameter.shape, dtype=torch.bfloat16)
    if isinstance(parameter, DTensor):
        parameter.grad = distribute_tensor(
            full_gradient,
            parameter.device_mesh,
            parameter.placements,
            src_data_rank=None,
        )
    else:
        parameter.grad = full_gradient


def _run_fsdp_steps(optimizer: torch.optim.Optimizer, parameter: Tensor, count: int) -> None:
    for _ in range(count):
        _set_fsdp_gradient(parameter)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)


def _fsdp_checkpoint_worker(rank: int, world_size: int, port: int, checkpoint_root: str) -> None:
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
        init_method=f"tcp://127.0.0.1:{port}",
    )
    try:
        mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("fsdp",))
        strategy = FSDPStrategy(
            fsdp_config=_Config(cpu_offload=False),
            optimizer_config=_fsdp_optimizer_config(),
            model_config=SimpleNamespace(lora=SimpleNamespace(rank=0)),
            fsdp_strategy="fsdp2",
            seed=20260823,
        )
        strategy.world_size = world_size

        for mode in (BFloat16UpdateMode.STOCHASTIC, BFloat16UpdateMode.KAHAN):
            model = _TinyBFloat16Model()
            fully_shard(model, mesh=mesh, reshard_after_forward=False)
            parameter = model.weight
            assert isinstance(parameter, DTensor)
            optimizer = _optimizer(parameter, mode, seed=20260823)

            _run_fsdp_steps(optimizer, parameter, 5)
            checkpoint_dir = str(Path(checkpoint_root) / mode.value)
            strategy.save_checkpoint(
                model,
                checkpoint_dir,
                node_local_rank=rank,
                optimizer=optimizer,
                client_state={"completed_steps": 5},
            )
            _run_fsdp_steps(optimizer, parameter, 7)
            expected_local = parameter.to_local().detach().clone()

            _run_fsdp_steps(optimizer, parameter, 3)
            _, client_state = strategy.load_checkpoint(model, checkpoint_dir, optimizer=optimizer)
            assert client_state["client_state"] == {"completed_steps": 5}
            _run_fsdp_steps(optimizer, parameter, 7)

            torch.testing.assert_close(parameter.to_local(), expected_local, rtol=0, atol=0)
            assert optimizer.param_groups[0]["bf16_update_step"] == 12

    finally:
        dist.destroy_process_group()


def _replicated_rounding_worker(rank: int, world_size: int, port: int) -> None:
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
        init_method=f"tcp://127.0.0.1:{port}",
    )
    try:
        mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("replicate",))
        replicated = nn.Parameter(
            distribute_tensor(
                torch.full((4_096,), 0.1, dtype=torch.bfloat16),
                mesh,
                (Replicate(),),
                src_data_rank=None,
            )
        )
        replicated_optimizer = _optimizer(replicated, BFloat16UpdateMode.STOCHASTIC, seed=20260823)
        _run_fsdp_steps(replicated_optimizer, replicated, 1)
        gathered = [torch.empty_like(replicated.to_local()) for _ in range(world_size)]
        dist.all_gather(gathered, replicated.to_local())
        for replica in gathered[1:]:
            torch.testing.assert_close(replica, gathered[0], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_two_rank_fsdp2_checkpoint_preserves_low_precision_updates(tmp_path: Path) -> None:
    mp.spawn(
        _fsdp_checkpoint_worker,
        args=(2, get_free_port(), str(tmp_path / "checkpoint")),
        nprocs=2,
        join=True,
    )


def test_two_rank_replicated_parameters_use_identical_rounding() -> None:
    mp.spawn(
        _replicated_rounding_worker,
        args=(2, get_free_port()),
        nprocs=2,
        join=True,
    )
