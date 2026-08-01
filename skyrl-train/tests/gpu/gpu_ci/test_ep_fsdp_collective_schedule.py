"""Exercise the model-level EP/FSDP collective schedule on four or twelve GPUs.

The tiny model uses the production TorchTitan expert-parallel hooks, FSDP2
wrapping, grouped MoE replay seam, and reentrant activation checkpointing. NCCL
completion hooks record operation types and sequence numbers for the rank-local
EP and FSDP process groups without adding collectives. Layer boundaries also
snapshot both process-group sequence counters.

Run the compact topology on one four-GPU node::

    torchrun --nproc-per-node=4 tests/gpu/gpu_ci/test_ep_fsdp_collective_schedule.py

Set ``SKYRL_TEST_EP_SIZE=4`` and ``SKYRL_TEST_FSDP_SIZE=3`` when launching a
twelve-rank gang to match the production EP4/FSDP3 topology.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
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


pytestmark = [pytest.mark.gpu]

DIM = 64
HIDDEN_DIM = 64
NUM_EXPERTS = 12
TOP_K = 2
NUM_LAYERS = 3
BATCH_SIZE = 1
SEQUENCE_LENGTH = 16
SEED = 1701
EP_SIZE_ENV = "SKYRL_TEST_EP_SIZE"
FSDP_SIZE_ENV = "SKYRL_TEST_FSDP_SIZE"
DEFAULT_EP_SIZE = 2


class TinyMoEBlock(torch.nn.Module):
    def __init__(self, layer_index: int) -> None:
        super().__init__()
        self.layer_index = layer_index
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
        self._record_boundary = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        phase = "recompute" if torch.is_grad_enabled() else "original"
        if self._record_boundary is not None:
            self._record_boundary(f"layer{self.layer_index}:{phase}:enter")
        output = hidden_states + self.mlp(self.norm(hidden_states))
        if self._record_boundary is not None:
            self._record_boundary(f"layer{self.layer_index}:{phase}:exit")
        return output


class TinyCheckpointedMoE(torch.nn.Module):
    _no_split_modules = ["TinyMoEBlock"]

    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(TinyMoEBlock(index) for index in range(NUM_LAYERS))
        self.output = torch.nn.Linear(DIM, DIM, bias=False)
        self.config = SimpleNamespace(tie_word_embeddings=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = checkpoint(layer, hidden_states, use_reentrant=True, preserve_rng_state=True)
        return self.output(hidden_states)


def _mesh_sizes(world_size: int) -> tuple[int, int]:
    ep_size = int(os.environ.get(EP_SIZE_ENV, str(DEFAULT_EP_SIZE)))
    fsdp_size = int(os.environ.get(FSDP_SIZE_ENV, str(world_size // ep_size)))
    if ep_size * fsdp_size != world_size:
        raise ValueError(f"test requires world_size == EP * FSDP; got {world_size} != {ep_size} * {fsdp_size}")
    experts_per_ep_rank = NUM_EXPERTS // ep_size
    if NUM_EXPERTS % ep_size or experts_per_ep_rank % fsdp_size:
        raise ValueError(f"NUM_EXPERTS={NUM_EXPERTS} must divide evenly over EP={ep_size} and FSDP={fsdp_size}")
    return ep_size, fsdp_size


def _expected_boundaries() -> tuple[str, ...]:
    original = tuple(f"layer{layer}:original:{edge}" for layer in range(NUM_LAYERS) for edge in ("enter", "exit"))
    recompute = tuple(
        f"layer{layer}:recompute:{edge}" for layer in reversed(range(NUM_LAYERS)) for edge in ("enter", "exit")
    )
    return original + recompute


def _replay_targets(fsdp_coordinate: int, device: torch.device) -> list[torch.Tensor]:
    targets = []
    for layer in range(NUM_LAYERS):
        first_expert = (fsdp_coordinate * NUM_LAYERS + layer * TOP_K) % NUM_EXPERTS
        selected = torch.empty(BATCH_SIZE * SEQUENCE_LENGTH, TOP_K, dtype=torch.long, device=device)
        for slot in range(TOP_K):
            selected[:, slot] = (first_expert + slot) % NUM_EXPERTS
        targets.append(selected)
    return targets


def _assert_operation_families(schedule: RankCollectiveSchedule) -> None:
    ep_operations = tuple(event.operation.upper() for event in schedule.events["ep"])
    assert len(ep_operations) == NUM_LAYERS * 8, (
        f"each checkpointed MoE layer must issue 3 original-forward, 3 recompute-forward, "
        f"and 2 backward EP all-to-alls; observed {len(ep_operations)} operations"
    )
    assert all("ALLTOALL" in operation for operation in ep_operations), ep_operations

    fsdp_operations = tuple(event.operation.upper() for event in schedule.events["fsdp"])
    assert any("ALLGATHER" in operation for operation in fsdp_operations), fsdp_operations
    assert any("REDUCE_SCATTER" in operation for operation in fsdp_operations), fsdp_operations


def _build_sharded_model(device: torch.device, device_mesh: DeviceMesh) -> TinyCheckpointedMoE:
    torch.manual_seed(SEED)
    model = TinyCheckpointedMoE().to(device=device, dtype=torch.bfloat16)
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


def _install_schedule_recorder(model: TinyCheckpointedMoE, device_mesh: DeviceMesh) -> NcclCollectiveRecorder:
    recorder = NcclCollectiveRecorder(
        {
            "ep": device_mesh.get_group("ep"),
            "fsdp": device_mesh.get_group("fsdp"),
        }
    )
    for layer in model.layers:
        layer._record_boundary = recorder.boundary
    return recorder


def _run_replayed_step(
    model: TinyCheckpointedMoE,
    fsdp_coordinate: int,
    device: torch.device,
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
    replay = RouterReplay()
    replay.begin_replay()
    replay.set_microbatch_targets(
        _replay_targets(fsdp_coordinate, device),
        torch.ones(BATCH_SIZE * SEQUENCE_LENGTH, dtype=torch.bool, device=device),
    )
    set_active_replay(replay)
    try:
        loss = model(hidden_states).float().square().mean()
        loss.backward()
        assert torch.isfinite(loss)
        assert replay.num_layers == NUM_LAYERS
    finally:
        set_active_replay(None)
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
    device_mesh: DeviceMesh,
) -> RankCollectiveSchedule:
    mesh_dim_names = tuple(device_mesh.mesh_dim_names)
    mesh_coordinate = tuple(device_mesh.get_coordinate())

    torch.cuda.synchronize()
    schedule = recorder.finish(
        rank=rank,
        mesh_dim_names=mesh_dim_names,
        mesh_shape=tuple(device_mesh.shape),
        mesh_coordinate=mesh_coordinate,
    )
    assert tuple(boundary.label for boundary in schedule.boundaries) == _expected_boundaries()
    _assert_operation_families(schedule)
    return schedule


def _compare_rank_schedules(schedule: RankCollectiveSchedule, world_size: int) -> None:
    schedules = [None] * world_size
    dist.all_gather_object(schedules, schedule)
    assert_collective_schedules_match(schedules, "ep")
    assert_collective_schedules_match(schedules, "fsdp")


def _run_ep_fsdp_replay_checkpoint_collective_schedule() -> None:
    init_worker_process_group_with_device(timeout_seconds=120)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", torch.cuda.current_device())
    ep_size, fsdp_size = _mesh_sizes(world_size)
    device_mesh = create_device_mesh(world_size, fsdp_size=fsdp_size, ep_size=ep_size)
    mesh_dim_names = tuple(device_mesh.mesh_dim_names)
    fsdp_coordinate = tuple(device_mesh.get_coordinate())[mesh_dim_names.index("fsdp")]

    model = _build_sharded_model(device, device_mesh)
    dist.barrier()
    recorder = _install_schedule_recorder(model, device_mesh)
    _run_replayed_step(model, fsdp_coordinate, device)
    schedule = _finish_local_schedule(recorder, rank, device_mesh)
    _compare_rank_schedules(schedule, world_size)

    if rank == 0:
        print(
            "MODEL_COLLECTIVE_SCHEDULE_OK "
            f"world={world_size} ep={ep_size} fsdp={fsdp_size} layers={NUM_LAYERS} "
            f"ep_ops={len(schedule.events['ep'])} fsdp_ops={len(schedule.events['fsdp'])}"
        )


def test_ep_fsdp_replay_checkpoint_collective_schedule() -> None:
    if not torch.cuda.is_available():
        pytest.skip("NCCL collective schedule contract requires CUDA")

    try:
        _run_ep_fsdp_replay_checkpoint_collective_schedule()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    test_ep_fsdp_replay_checkpoint_collective_schedule()
