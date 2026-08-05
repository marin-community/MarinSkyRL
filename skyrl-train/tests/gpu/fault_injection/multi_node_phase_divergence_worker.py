"""Isolate permanent collective-stall mechanisms in a warmed four-node mesh."""

from __future__ import annotations

import argparse
import os
import signal
from pathlib import Path

import torch
import torch.distributed as dist

from tests.gpu.fault_injection.collective_payloads import (
    MeshCollectives,
    VERIFICATION_DTYPE,
    numel_for_mib,
    warm_ep_and_fsdp_communicators,
)
from tests.gpu.fault_injection.multi_node_mesh import MeshRuntime, multi_node_mesh_runtime
from tests.gpu.fault_injection.multi_node_phase_divergence_protocol import (
    ACTIVE_MARKER,
    AFTER_ENQUEUE_MARKER,
    BEFORE_ENQUEUE_MARKER,
    BLOCKING_WAIT_DISABLED_MARKER,
    BLOCKING_WAIT_FIELD,
    MODEL_OPERATION_MARKER,
    PROCESS_GROUP_TIMEOUT_SECONDS,
    READY_MARKER,
    UNAFFECTED_FSDP_COMPLETION_MARKER,
    UNEXPECTED_COMPLETION_MARKER,
    WARMUP_MARKER,
    FaultMode,
)
from tests.nccl_environment import disable_nccl_communicator_nonblocking
from tests.process_gang import signal_rank_ready_and_wait_for_start


WARMUP_ROUNDS = 3
PAYLOAD_MIB = 1
TARGET_RANK = 1
# torch.cuda._sleep counts device cycles. This is longer than both the process-group
# deadline and controller deadline on supported GPUs and is interrupted by teardown.
CUDA_STALL_CYCLES = 1_000_000_000_000


def _build_and_warm_communicators(runtime: MeshRuntime, values_per_rank: int) -> MeshCollectives:
    collectives = runtime.collective_groups()
    warm_ep_and_fsdp_communicators(
        collectives,
        rounds=WARMUP_ROUNDS,
        ep_values_per_rank=values_per_rank,
        fsdp_values_per_rank=values_per_rank,
    )
    torch.cuda.synchronize(runtime.device)
    print(f"{WARMUP_MARKER} rank={runtime.placement.rank} rounds={WARMUP_ROUNDS}", flush=True)
    return collectives


def _marker(marker: str, *, mode: FaultMode, rank: int, phase: str) -> None:
    print(f"{marker} mode={mode.value} rank={rank} phase={phase}", flush=True)


def _enqueue_fsdp_all_gather(
    collectives: MeshCollectives,
    values_per_rank: int,
    *,
    mode: FaultMode,
    rank: int,
    stall_current_stream: bool = False,
) -> tuple[dist.Work, torch.Tensor, torch.Tensor]:
    input_values = torch.full((values_per_rank,), rank, dtype=VERIFICATION_DTYPE, device=collectives.fsdp.device)
    output_values = torch.empty(
        values_per_rank * len(collectives.fsdp.ranks),
        dtype=VERIFICATION_DTYPE,
        device=collectives.fsdp.device,
    )
    if stall_current_stream:
        torch.cuda._sleep(CUDA_STALL_CYCLES)
    _marker(BEFORE_ENQUEUE_MARKER, mode=mode, rank=rank, phase="fsdp-all-gather")
    work = dist.all_gather_into_tensor(
        output_values,
        input_values,
        group=collectives.fsdp.process_group,
        async_op=True,
    )
    _marker(AFTER_ENQUEUE_MARKER, mode=mode, rank=rank, phase="fsdp-all-gather")
    return work, input_values, output_values


def _wait_for_fsdp_all_gather(
    collectives: MeshCollectives,
    values_per_rank: int,
    *,
    mode: FaultMode,
    rank: int,
    stall_current_stream: bool = False,
) -> None:
    work, input_values, output_values = _enqueue_fsdp_all_gather(
        collectives,
        values_per_rank,
        mode=mode,
        rank=rank,
        stall_current_stream=stall_current_stream,
    )
    work.wait()
    del input_values, output_values


def _hold_after_healthy_fsdp(
    collectives: MeshCollectives,
    values_per_rank: int,
    *,
    mode: FaultMode,
    rank: int,
) -> None:
    _wait_for_fsdp_all_gather(collectives, values_per_rank, mode=mode, rank=rank)
    _marker(UNAFFECTED_FSDP_COMPLETION_MARKER, mode=mode, rank=rank, phase="fsdp-all-gather")
    signal.pause()


def _run_enqueued_cuda_stall(
    runtime: MeshRuntime,
    collectives: MeshCollectives,
    values_per_rank: int,
) -> None:
    rank = runtime.placement.rank
    if TARGET_RANK not in collectives.fsdp.ranks:
        _hold_after_healthy_fsdp(
            collectives,
            values_per_rank,
            mode=FaultMode.ENQUEUED_CUDA_STALL,
            rank=rank,
        )

    _wait_for_fsdp_all_gather(
        collectives,
        values_per_rank,
        mode=FaultMode.ENQUEUED_CUDA_STALL,
        rank=rank,
        stall_current_stream=rank == TARGET_RANK,
    )
    _marker(
        UNEXPECTED_COMPLETION_MARKER,
        mode=FaultMode.ENQUEUED_CUDA_STALL,
        rank=rank,
        phase="fsdp-all-gather",
    )


def _run_pre_enqueue_nonarrival(
    runtime: MeshRuntime,
    collectives: MeshCollectives,
    values_per_rank: int,
) -> None:
    rank = runtime.placement.rank
    if TARGET_RANK not in collectives.fsdp.ranks:
        _hold_after_healthy_fsdp(
            collectives,
            values_per_rank,
            mode=FaultMode.PRE_ENQUEUE_NONARRIVAL,
            rank=rank,
        )

    _marker(
        BEFORE_ENQUEUE_MARKER,
        mode=FaultMode.PRE_ENQUEUE_NONARRIVAL,
        rank=rank,
        phase="fsdp-all-gather",
    )
    if rank == TARGET_RANK:
        signal.pause()
    _wait_for_fsdp_all_gather(
        collectives,
        values_per_rank,
        mode=FaultMode.PRE_ENQUEUE_NONARRIVAL,
        rank=rank,
    )
    _marker(
        UNEXPECTED_COMPLETION_MARKER,
        mode=FaultMode.PRE_ENQUEUE_NONARRIVAL,
        rank=rank,
        phase="fsdp-all-gather",
    )


def _run_model_schedule_divergence(
    runtime: MeshRuntime,
    collectives: MeshCollectives,
    values_per_rank: int,
) -> None:
    rank = runtime.placement.rank
    value = torch.ones((), device=runtime.device, requires_grad=True)

    def model_backward_hook(gradient: torch.Tensor) -> torch.Tensor:
        _marker(MODEL_OPERATION_MARKER, mode=FaultMode.MODEL_SCHEDULE_DIVERGENCE, rank=rank, phase="backward")
        if rank != TARGET_RANK:
            return gradient

        _marker(
            BEFORE_ENQUEUE_MARKER,
            mode=FaultMode.MODEL_SCHEDULE_DIVERGENCE,
            rank=rank,
            phase="ep-all-to-all",
        )
        output = torch.empty(values_per_rank, dtype=VERIFICATION_DTYPE, device=runtime.device)
        source = torch.full_like(output, rank)
        work = dist.all_to_all_single(output, source, group=collectives.ep.process_group, async_op=True)
        _marker(
            AFTER_ENQUEUE_MARKER,
            mode=FaultMode.MODEL_SCHEDULE_DIVERGENCE,
            rank=rank,
            phase="ep-all-to-all",
        )
        work.wait()
        return gradient

    value.register_hook(model_backward_hook)
    value.square().backward()

    if TARGET_RANK not in collectives.fsdp.ranks:
        _hold_after_healthy_fsdp(
            collectives,
            values_per_rank,
            mode=FaultMode.MODEL_SCHEDULE_DIVERGENCE,
            rank=rank,
        )
    _wait_for_fsdp_all_gather(
        collectives,
        values_per_rank,
        mode=FaultMode.MODEL_SCHEDULE_DIVERGENCE,
        rank=rank,
    )
    _marker(
        UNEXPECTED_COMPLETION_MARKER,
        mode=FaultMode.MODEL_SCHEDULE_DIVERGENCE,
        rank=rank,
        phase="fsdp-all-gather",
    )


def _run(runtime: MeshRuntime, control_directory: Path, mode: FaultMode) -> None:
    values_per_rank = numel_for_mib(PAYLOAD_MIB, VERIFICATION_DTYPE)
    collectives = _build_and_warm_communicators(runtime, values_per_rank)
    rank = runtime.placement.rank
    blocking_wait = os.environ.get("NCCL_BLOCKING_WAIT")
    blocking_wait_record = (
        BLOCKING_WAIT_DISABLED_MARKER if blocking_wait is None else f"{BLOCKING_WAIT_FIELD}={blocking_wait}"
    )
    print(
        f"{READY_MARKER} mode={mode.value} rank={rank} backend={dist.get_backend()} "
        f"timeout={PROCESS_GROUP_TIMEOUT_SECONDS} ep_ranks={collectives.ep.ranks} "
        f"fsdp_ranks={collectives.fsdp.ranks} "
        f"async_error_handling={os.environ.get('TORCH_NCCL_ASYNC_ERROR_HANDLING')} {blocking_wait_record}",
        flush=True,
    )
    signal_rank_ready_and_wait_for_start(control_directory, rank)
    _marker(ACTIVE_MARKER, mode=mode, rank=rank, phase="fault")

    if mode is FaultMode.ENQUEUED_CUDA_STALL:
        _run_enqueued_cuda_stall(runtime, collectives, values_per_rank)
    elif mode is FaultMode.PRE_ENQUEUE_NONARRIVAL:
        _run_pre_enqueue_nonarrival(runtime, collectives, values_per_rank)
    else:
        _run_model_schedule_divergence(runtime, collectives, values_per_rank)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-directory", type=Path, required=True)
    parser.add_argument("--fault-mode", type=FaultMode, choices=tuple(FaultMode), required=True)
    arguments = parser.parse_args()
    disable_nccl_communicator_nonblocking(os.environ)
    with multi_node_mesh_runtime(PROCESS_GROUP_TIMEOUT_SECONDS, os.environ) as mesh_runtime:
        _run(mesh_runtime, arguments.control_directory, arguments.fault_mode)


if __name__ == "__main__":
    main()
