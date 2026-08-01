"""Exercise model-level EP/FSDP collective schedules on four or twelve GPUs.

Every case uses the production TorchTitan expert-parallel hooks, FSDP2 wrapping,
and grouped MoE path. ``--case`` selects live or replayed routing, checkpointing,
routing concentration, and an optional rank delay. NCCL completion hooks record
operation types and sequence numbers for the rank-local EP and FSDP process
groups without adding collectives. Layer boundaries also snapshot both
process-group sequence counters.

Run the compact topology on one four-GPU node::

    torchrun --nproc-per-node=4 tests/gpu/fault_injection/ep_fsdp_collective_schedule_worker.py

The default case uses concentrated router replay and reentrant checkpointing.
Pass ``--case`` with one of the choices reported by ``--help`` to select another
matrix case.

The worker enables ``TORCH_NCCL_ENABLE_TIMING=1`` before process-group
initialization because ProcessGroupNCCL completion hooks require start and end
event recording.

Pass ``--ep-size 4 --fsdp-size 3`` after the script path when launching a
twelve-rank gang to exercise an EP4/FSDP3 topology.
"""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Callable
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, DeviceMesh
from torch.utils.checkpoint import checkpoint

from skyrl_train.distributed.fsdp_utils import apply_ep, apply_fsdp2, create_device_mesh
from skyrl_train.distributed.utils import init_worker_process_group_with_device
from skyrl_train.models.layers.moe import MoE
from skyrl_train.models.layers.moe_swap import GroupedMoEShim
from skyrl_train.models.router_replay import RouterReplay, set_active_replay
from tests.collective_schedule import (
    NcclCollectiveRecorder,
    RankCollectiveSchedule,
    assert_collective_schedules_match,
)
from tests.collective_schedule_matrix import (
    COLLECTIVE_SCHEDULE_CASES,
    DEFAULT_COLLECTIVE_SCHEDULE_CASE,
    CheckpointMode,
    CollectiveScheduleCase,
    RoutingMode,
    collective_schedule_case,
)


DIM = 64
HIDDEN_DIM = 64
NUM_EXPERTS = 12
TOP_K = 2
NUM_LAYERS = 3
BATCH_SIZE = 1
SEQUENCE_LENGTH = 16
SEED = 1701
DEFAULT_EP_SIZE = 2
NCCL_TIMING_ENV = "TORCH_NCCL_ENABLE_TIMING"
DELAY_BOUNDARY = "layer1:original:enter"
RANK_DELAY_SECONDS = 2.0


class BoundaryDelay:
    def __init__(self, rank: int, delayed_rank: int | None) -> None:
        self._rank = rank
        self._delayed_rank = delayed_rank
        self.was_applied = False

    def apply(self, label: str) -> None:
        if self._rank != self._delayed_rank or label != DELAY_BOUNDARY:
            return
        if self.was_applied:
            raise AssertionError(f"delay boundary {DELAY_BOUNDARY!r} occurred more than once")
        self.was_applied = True
        # The delay is the injected condition: peers enter the next real EP/FSDP
        # collective while this rank remains outside it.
        time.sleep(RANK_DELAY_SECONDS)


class TinyMoEBlock(torch.nn.Module):
    def __init__(self, layer_index: int, checkpoint_mode: CheckpointMode) -> None:
        super().__init__()
        self.layer_index = layer_index
        self.checkpoint_mode = checkpoint_mode
        self.norm = torch.nn.LayerNorm(DIM)
        self.mlp = GroupedMoEShim(
            MoE(
                dim=DIM,
                hidden_dim=HIDDEN_DIM,
                num_experts=NUM_EXPERTS,
                top_k=TOP_K,
                route_norm=True,
                use_grouped_mm=True,
            ),
            returns_tuple=False,
        )
        self._record_boundary: Callable[[str], None] | None = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        phase = (
            "recompute" if self.checkpoint_mode is CheckpointMode.REENTRANT and torch.is_grad_enabled() else "original"
        )
        if self._record_boundary is not None:
            self._record_boundary(f"layer{self.layer_index}:{phase}:enter")
        output = hidden_states + self.mlp(self.norm(hidden_states))
        if self._record_boundary is not None:
            self._record_boundary(f"layer{self.layer_index}:{phase}:exit")
        return output


class TinyMoEModel(torch.nn.Module):
    def __init__(self, checkpoint_mode: CheckpointMode) -> None:
        super().__init__()
        self.checkpoint_mode = checkpoint_mode
        self.layers = torch.nn.ModuleList(TinyMoEBlock(index, checkpoint_mode) for index in range(NUM_LAYERS))
        self.output = torch.nn.Linear(DIM, DIM, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            if self.checkpoint_mode is CheckpointMode.REENTRANT:
                hidden_states = checkpoint(layer, hidden_states, use_reentrant=True, preserve_rng_state=True)
            else:
                hidden_states = layer(hidden_states)
        return self.output(hidden_states)


def _mesh_sizes(world_size: int, ep_size: int, fsdp_size: int | None) -> tuple[int, int]:
    fsdp_size = world_size // ep_size if fsdp_size is None else fsdp_size
    if ep_size * fsdp_size != world_size:
        raise ValueError(f"test requires world_size == EP * FSDP; got {world_size} != {ep_size} * {fsdp_size}")
    experts_per_ep_rank = NUM_EXPERTS // ep_size
    if NUM_EXPERTS % ep_size or experts_per_ep_rank % fsdp_size:
        raise ValueError(f"NUM_EXPERTS={NUM_EXPERTS} must divide evenly over EP={ep_size} and FSDP={fsdp_size}")
    return ep_size, fsdp_size


def _expected_boundaries(checkpoint_mode: CheckpointMode) -> tuple[str, ...]:
    original = tuple(f"layer{layer}:original:{edge}" for layer in range(NUM_LAYERS) for edge in ("enter", "exit"))
    if checkpoint_mode is CheckpointMode.NONE:
        return original
    recompute = tuple(
        f"layer{layer}:recompute:{edge}" for layer in reversed(range(NUM_LAYERS)) for edge in ("enter", "exit")
    )
    return original + recompute


def _replay_targets(
    fsdp_coordinate: int,
    routing_mode: RoutingMode,
    device: torch.device,
) -> list[torch.Tensor]:
    targets = []
    for layer in range(NUM_LAYERS):
        first_expert = (fsdp_coordinate * NUM_LAYERS + layer * TOP_K) % NUM_EXPERTS
        selected = torch.empty(BATCH_SIZE * SEQUENCE_LENGTH, TOP_K, dtype=torch.long, device=device)
        for token in range(BATCH_SIZE * SEQUENCE_LENGTH):
            token_offset = token * TOP_K if routing_mode is RoutingMode.REPLAY_SPREAD else 0
            for slot in range(TOP_K):
                selected[token, slot] = (first_expert + token_offset + slot) % NUM_EXPERTS
        targets.append(selected)
    return targets


def _assert_operation_families(schedule: RankCollectiveSchedule, checkpoint_mode: CheckpointMode) -> None:
    ep_operations = tuple(event.operation.upper() for event in schedule.events["ep"])
    operations_per_layer = 8 if checkpoint_mode is CheckpointMode.REENTRANT else 5
    assert len(ep_operations) == NUM_LAYERS * operations_per_layer, (
        f"expected {operations_per_layer} EP all-to-alls per layer for checkpoint_mode={checkpoint_mode.value}; "
        f"observed {len(ep_operations)} operations"
    )
    assert all("ALLTOALL" in operation for operation in ep_operations), ep_operations

    fsdp_operations = tuple(event.operation.upper() for event in schedule.events["fsdp"])
    assert any("ALLGATHER" in operation for operation in fsdp_operations), fsdp_operations
    assert any("REDUCE_SCATTER" in operation for operation in fsdp_operations), fsdp_operations


def _build_sharded_model(
    device: torch.device,
    device_mesh: DeviceMesh,
    checkpoint_mode: CheckpointMode,
) -> TinyMoEModel:
    torch.manual_seed(SEED)
    model = TinyMoEModel(checkpoint_mode).to(device=device, dtype=torch.bfloat16)
    for module in model.modules():
        if isinstance(module, MoE):
            module.init_weights(0.02)

    fsdp_kwargs = {"mesh": device_mesh["fsdp"], "reshard_after_forward": True}
    sharded_experts = apply_ep(model, device_mesh, ep_comm_backend="torch", fsdp_kwargs=fsdp_kwargs)
    assert sharded_experts == NUM_LAYERS
    apply_fsdp2(
        model,
        fsdp_kwargs,
        {"wrap_policy": {"transformer_layer_cls_to_wrap": ["TinyMoEBlock"]}},
    )
    assert all(isinstance(layer.mlp.moe.experts.w1, DTensor) for layer in model.layers)
    return model


def _install_schedule_recorder(
    model: TinyMoEModel,
    device_mesh: DeviceMesh,
    case: CollectiveScheduleCase,
    rank: int,
) -> tuple[NcclCollectiveRecorder, BoundaryDelay]:
    recorder = NcclCollectiveRecorder(device_mesh, ("ep", "fsdp"))
    delay = BoundaryDelay(rank, case.delayed_rank)

    def record_boundary(label: str) -> None:
        delay.apply(label)
        recorder.record_boundary_snapshot(label)

    for layer in model.layers:
        layer._record_boundary = record_boundary
    return recorder, delay


def _run_training_step(
    model: TinyMoEModel,
    fsdp_coordinate: int,
    device: torch.device,
    routing_mode: RoutingMode,
) -> None:
    torch.manual_seed(SEED + fsdp_coordinate)
    hidden_states = torch.randn(
        BATCH_SIZE,
        SEQUENCE_LENGTH,
        DIM,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    replay = None
    if routing_mode is not RoutingMode.LIVE:
        replay = RouterReplay()
        replay.begin_replay()
        targets = _replay_targets(fsdp_coordinate, routing_mode, device)
        expected_experts = NUM_EXPERTS if routing_mode is RoutingMode.REPLAY_SPREAD else TOP_K
        assert all(torch.unique(layer_targets).numel() == expected_experts for layer_targets in targets)
        replay.set_microbatch_targets(
            targets,
            torch.ones(BATCH_SIZE * SEQUENCE_LENGTH, dtype=torch.bool, device=device),
        )
        set_active_replay(replay)
    try:
        loss = model(hidden_states).float().square().mean()
        loss.backward()
        assert torch.isfinite(loss)
    finally:
        set_active_replay(None)
        if replay is not None:
            replay.clear()

    local_grad_sums = []
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad.to_local() if isinstance(parameter.grad, DTensor) else parameter.grad
        local_grad_sums.append(gradient.float().abs().sum())
    assert local_grad_sums and torch.stack(local_grad_sums).sum().item() > 0


def _finish_local_schedule(
    recorder: NcclCollectiveRecorder,
    rank: int,
    checkpoint_mode: CheckpointMode,
) -> RankCollectiveSchedule:
    torch.cuda.synchronize()
    schedule = recorder.finish(rank=rank)
    assert tuple(boundary.label for boundary in schedule.boundaries) == _expected_boundaries(checkpoint_mode)
    _assert_operation_families(schedule, checkpoint_mode)
    return schedule


def _compare_rank_schedules(schedule: RankCollectiveSchedule, world_size: int) -> None:
    schedules = [schedule for _ in range(world_size)]
    dist.all_gather_object(schedules, schedule)
    assert_collective_schedules_match(schedules, "ep")
    assert_collective_schedules_match(schedules, "fsdp")


def _run_ep_fsdp_collective_schedule_case(
    ep_size: int,
    fsdp_size: int | None,
    case: CollectiveScheduleCase,
) -> None:
    init_worker_process_group_with_device(timeout_seconds=120)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", torch.cuda.current_device())
    ep_size, fsdp_size = _mesh_sizes(world_size, ep_size, fsdp_size)
    device_mesh = create_device_mesh(world_size, fsdp_size=fsdp_size, ep_size=ep_size)
    mesh_dim_names = tuple(device_mesh.mesh_dim_names)
    fsdp_coordinate = tuple(device_mesh.get_coordinate())[mesh_dim_names.index("fsdp")]

    model = _build_sharded_model(device, device_mesh, case.checkpoint_mode)
    dist.barrier()
    recorder, delay = _install_schedule_recorder(model, device_mesh, case, rank)
    _run_training_step(model, fsdp_coordinate, device, case.routing_mode)
    assert delay.was_applied is (rank == case.delayed_rank)
    schedule = _finish_local_schedule(recorder, rank, case.checkpoint_mode)
    _compare_rank_schedules(schedule, world_size)

    if rank == 0:
        print(
            "MODEL_COLLECTIVE_SCHEDULE_OK "
            f"case={case.name} world={world_size} ep={ep_size} fsdp={fsdp_size} layers={NUM_LAYERS} "
            f"ep_ops={len(schedule.events['ep'])} fsdp_ops={len(schedule.events['fsdp'])}"
        )


def _run_collective_schedule_with_managed_process_group(
    ep_size: int,
    fsdp_size: int | None,
    case: CollectiveScheduleCase,
) -> None:
    with patch.dict(os.environ, {NCCL_TIMING_ENV: "1"}):
        try:
            _run_ep_fsdp_collective_schedule_case(ep_size, fsdp_size, case)
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep-size", type=int, default=DEFAULT_EP_SIZE)
    parser.add_argument("--fsdp-size", type=int)
    parser.add_argument(
        "--case",
        choices=tuple(case.name for case in COLLECTIVE_SCHEDULE_CASES),
        default=DEFAULT_COLLECTIVE_SCHEDULE_CASE,
    )
    arguments = parser.parse_args()
    _run_collective_schedule_with_managed_process_group(
        arguments.ep_size,
        arguments.fsdp_size,
        collective_schedule_case(arguments.case),
    )
