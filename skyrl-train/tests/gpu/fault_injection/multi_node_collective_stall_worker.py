"""Isolate permanent collective-stall mechanisms in a warmed four-node mesh."""

from __future__ import annotations

import argparse
import os
import signal
from dataclasses import dataclass
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
from tests.gpu.fault_injection.multi_node_collective_stall_protocol import (
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


@dataclass(frozen=True)
class FaultContext:
    runtime: MeshRuntime
    collectives: MeshCollectives
    values_per_rank: int
    mode: FaultMode

    @property
    def rank(self) -> int:
        return self.runtime.placement.rank


@dataclass(frozen=True)
class PendingAllGather:
    work: dist.Work
    input_values: torch.Tensor
    output_values: torch.Tensor


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


def _marker(context: FaultContext, marker: str, phase: str) -> None:
    print(f"{marker} mode={context.mode.value} rank={context.rank} phase={phase}", flush=True)


def _enqueue_fsdp_all_gather(
    context: FaultContext,
    *,
    stall_current_stream: bool = False,
) -> PendingAllGather:
    input_values = torch.full(
        (context.values_per_rank,),
        context.rank,
        dtype=VERIFICATION_DTYPE,
        device=context.collectives.fsdp.device,
    )
    output_values = torch.empty(
        context.values_per_rank * len(context.collectives.fsdp.ranks),
        dtype=VERIFICATION_DTYPE,
        device=context.collectives.fsdp.device,
    )
    if stall_current_stream:
        torch.cuda._sleep(CUDA_STALL_CYCLES)
    _marker(context, BEFORE_ENQUEUE_MARKER, "fsdp-all-gather")
    work = dist.all_gather_into_tensor(
        output_values,
        input_values,
        group=context.collectives.fsdp.process_group,
        async_op=True,
    )
    _marker(context, AFTER_ENQUEUE_MARKER, "fsdp-all-gather")
    return PendingAllGather(work, input_values, output_values)


def _run_fsdp_all_gather_to_completion(context: FaultContext, *, stall_current_stream: bool = False) -> None:
    pending = _enqueue_fsdp_all_gather(context, stall_current_stream=stall_current_stream)
    pending.work.wait()
    torch.cuda.synchronize(context.collectives.fsdp.device)


def _hold_after_healthy_fsdp(context: FaultContext) -> None:
    _run_fsdp_all_gather_to_completion(context)
    _marker(context, UNAFFECTED_FSDP_COMPLETION_MARKER, "fsdp-all-gather")
    signal.pause()


def _run_model_operation(context: FaultContext) -> None:
    value = torch.ones((), device=context.runtime.device, requires_grad=True)

    def model_backward_hook(gradient: torch.Tensor) -> torch.Tensor:
        _marker(context, MODEL_OPERATION_MARKER, "backward")
        if context.rank != TARGET_RANK:
            return gradient

        _marker(context, BEFORE_ENQUEUE_MARKER, "ep-all-to-all")
        output = torch.empty(
            context.values_per_rank,
            dtype=VERIFICATION_DTYPE,
            device=context.runtime.device,
        )
        source = torch.full_like(output, context.rank)
        work = dist.all_to_all_single(
            output,
            source,
            group=context.collectives.ep.process_group,
            async_op=True,
        )
        _marker(context, AFTER_ENQUEUE_MARKER, "ep-all-to-all")
        work.wait()
        torch.cuda.synchronize(context.runtime.device)
        return gradient

    value.register_hook(model_backward_hook)
    value.square().backward()


def _run_fault(context: FaultContext) -> None:
    if context.mode is FaultMode.MODEL_SCHEDULE_DIVERGENCE:
        _run_model_operation(context)

    if TARGET_RANK not in context.collectives.fsdp.ranks:
        _hold_after_healthy_fsdp(context)

    if context.mode is FaultMode.PRE_ENQUEUE_NONARRIVAL and context.rank == TARGET_RANK:
        _marker(context, BEFORE_ENQUEUE_MARKER, "fsdp-all-gather")
        signal.pause()

    _run_fsdp_all_gather_to_completion(
        context,
        stall_current_stream=context.mode is FaultMode.ENQUEUED_CUDA_STALL and context.rank == TARGET_RANK,
    )
    _marker(context, UNEXPECTED_COMPLETION_MARKER, "fsdp-all-gather")


def _run(runtime: MeshRuntime, control_directory: Path, mode: FaultMode) -> None:
    values_per_rank = numel_for_mib(PAYLOAD_MIB, VERIFICATION_DTYPE)
    collectives = _build_and_warm_communicators(runtime, values_per_rank)
    context = FaultContext(runtime, collectives, values_per_rank, mode)
    blocking_wait = os.environ.get("NCCL_BLOCKING_WAIT")
    blocking_wait_record = (
        BLOCKING_WAIT_DISABLED_MARKER if blocking_wait is None else f"{BLOCKING_WAIT_FIELD}={blocking_wait}"
    )
    print(
        f"{READY_MARKER} mode={mode.value} rank={context.rank} backend={dist.get_backend()} "
        f"timeout={PROCESS_GROUP_TIMEOUT_SECONDS} ep_ranks={collectives.ep.ranks} "
        f"fsdp_ranks={collectives.fsdp.ranks} "
        f"async_error_handling={os.environ.get('TORCH_NCCL_ASYNC_ERROR_HANDLING')} {blocking_wait_record}",
        flush=True,
    )
    signal_rank_ready_and_wait_for_start(control_directory, context.rank)
    _marker(context, ACTIVE_MARKER, "fault")
    _run_fault(context)


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
