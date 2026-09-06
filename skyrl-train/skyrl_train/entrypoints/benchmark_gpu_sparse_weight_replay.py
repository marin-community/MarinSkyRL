"""Standalone synthetic exact GPU replay, never a training/publication backend.

torchrun --standalone --nproc-per-node=2 -m \
    skyrl_train.entrypoints.benchmark_gpu_sparse_weight_replay --total-mib 4 --chunk-mib 4

Both dense and sparse include resident-fixture base/target checks, receiver ACK,
and sender predecessor commit. Report maximum rank wall time; stream phases and
control waits overlap and must not be summed. NCCL intervals include readiness
waits, not just link service. Fixtures and allocations are outside timing, and
their extra resident memory is disclosed. No model loader or actual Qwen deltas
are exercised. A rejected ACK aborts both ranks; reconnect recovery is absent.
Chunks are independent transactions with one chunk resident at a time. Total
MiB measures serial traffic, not a whole-model cache or atomic publication.
"""

import argparse
from dataclasses import asdict
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import socket
import time

import torch
import torch.distributed as dist

from skyrl_train.gpu_sparse_weight_replay import (
    CudaPhases,
    Encoding,
    GpuWorkspace,
    PatchHeader,
    RAW_DTYPES,
    ReplayBaseline,
    packet_header,
    validate_ack,
)
from skyrl_train.sparse_weight_replay import MAX_CACHE_BYTES, MAX_CHUNK_BYTES


def synthetic_target(base: torch.Tensor, fraction: float, seed: int) -> torch.Tensor:
    """Flip exactly one bit in a deterministic GPU permutation of raw elements."""
    target = base.clone()
    generator = torch.Generator(device=base.device).manual_seed(seed)
    indices = torch.randperm(base.numel(), device=base.device, generator=generator)[: round(base.numel() * fraction)]
    target[indices] = torch.bitwise_xor(target[indices], 1)
    return target


def transaction(
    state: ReplayBaseline,
    staging: torch.Tensor,
    expected_base: torch.Tensor,
    target: torch.Tensor,
    workspace: GpuWorkspace,
    preferred: Encoding,
    control,
) -> tuple[torch.Tensor, dict]:
    """Fail together at admission or ACK so a validation error cannot strand NCCL."""
    rank = dist.get_rank()
    phases, stats = CudaPhases(), {}
    torch.cuda.synchronize()
    dist.barrier()
    started = time.perf_counter()
    metadata = [None]
    if rank == 0:
        try:
            provisional = packet_header(
                elements=state.values.numel(),
                width=state.values.element_size(),
                changed=0,
                preferred=Encoding.DENSE,
                base_version=state.version,
                target_version=state.version + 1,
            )
            check_started = time.perf_counter()
            state.validate_base(provisional, expected_base)
            stats["base_integrity_wall_seconds"] = time.perf_counter() - check_started
            header, packing = workspace.pack(
                state.values, target, preferred=preferred, version=state.version, phases=phases
            )
            stats.update(packing)
            metadata[0] = {"header": asdict(header)}
        except ValueError as error:
            metadata[0] = {"error": str(error)}
    control_started = time.perf_counter()
    dist.broadcast_object_list(metadata, src=0, group=control)
    stats["header_control_wall_seconds"] = time.perf_counter() - control_started
    if "error" in metadata[0]:
        raise ValueError(metadata[0]["error"])
    fields = metadata[0]["header"]
    header = PatchHeader(**{**fields, "encoding": Encoding(fields["encoding"])})
    admission = [None]
    if rank == 1:
        try:
            check_started = time.perf_counter()
            state.validate_base(header, expected_base)
            stats["base_integrity_wall_seconds"] = time.perf_counter() - check_started
            admission[0] = {"accepted": True}
        except ValueError as error:
            admission[0] = {"accepted": False, "error": str(error)}
    control_started = time.perf_counter()
    dist.broadcast_object_list(admission, src=1, group=control)
    stats["admission_control_wall_seconds"] = time.perf_counter() - control_started
    if not admission[0]["accepted"]:
        raise ValueError(admission[0]["error"])
    transfer_start = phases.start("observed_nccl_gpu_seconds")
    payloads = workspace.payload(header, target if rank == 0 else staging)
    for payload in payloads:
        dist.broadcast(payload, src=0)
    phases.end("observed_nccl_gpu_seconds", transfer_start)
    ack = [None]
    if rank == 1:
        try:
            stats.update(workspace.apply(header, state.values, staging, phases))
            check_started = time.perf_counter()
            staging = state.install(header, staging, target)
            stats["target_integrity_and_install_wall_seconds"] = time.perf_counter() - check_started
            ack[0] = {"accepted": True, "base_version": header.base_version, "target_version": header.target_version}
        except ValueError as error:
            ack[0] = {"accepted": False, "error": str(error)}
    control_started = time.perf_counter()
    dist.broadcast_object_list(ack, src=1, group=control)
    stats["ack_control_wall_seconds"] = time.perf_counter() - control_started
    if not ack[0]["accepted"]:
        raise ValueError(ack[0]["error"])
    validate_ack(header, ack[0])
    if rank == 0:
        commit_started = phases.start("sender_predecessor_commit_gpu_seconds")
        state.commit_after_ack(header, target, ack[0])
        phases.end("sender_predecessor_commit_gpu_seconds", commit_started)
        if not torch.equal(state.values, target):
            raise AssertionError("Sender predecessor commit differs from target")
    torch.cuda.synchronize()
    # This final rendezvous includes the sender's completed predecessor update.
    dist.barrier()
    stats.update(
        elapsed_seconds=time.perf_counter() - started,
        payload_bytes=header.payload_bytes,
        payload_collectives=len(payloads),
        encoding=header.encoding.value,
        preferred=preferred.value,
        base_version=header.base_version,
        target_version=header.target_version,
        metadata_estimated_json_bytes=sum(
            len(json.dumps(value).encode()) for value in (metadata[0], admission[0], ack[0])
        ),
        **phases.seconds(),
    )
    return staging, stats


def kernel_gate(device: torch.device) -> int:
    """Check exact kernels and malformed addressing before the timed fleet."""
    cases = 0
    for width, dtype in RAW_DTYPES.items():
        elements = 2053  # Partial tail and changes across multiple packing blocks.
        initial = (torch.arange(elements, device=device, dtype=torch.int32) * 1103515245).to(dtype)
        workspace = GpuWorkspace(elements, width, device)
        for preferred in (Encoding.INDEX32, Encoding.BLOCK_LOCAL16):
            # Exact replacements must preserve signed zero and distinct NaN payloads.
            special_base = initial.clone()
            special_base[0], special_base[1] = 0, 0x7FC1 if width == 2 else 0x7FC00001
            target = special_base.clone()
            target[0], target[1] = -(1 << (8 * width - 1)), special_base[1] ^ 1
            header, packing = workspace.pack(special_base, target, preferred=preferred, version=0, phases=CudaPhases())
            staged = torch.empty_like(initial)
            workspace.apply(header, special_base, staged, CudaPhases())
            if packing["changed_elements"] != 2 or not torch.equal(staged, target):
                raise AssertionError("Signed zero or NaN payload changed-bit gate failed")
            cases += 1
            for fraction in (0.0, 0.02, 0.2, 1.0):
                target = synthetic_target(initial, fraction, 17)
                phases = CudaPhases()
                header, _ = workspace.pack(initial, target, preferred=preferred, version=0, phases=phases)
                staged = torch.empty_like(initial)
                if header.encoding == Encoding.DENSE:
                    staged.copy_(target)
                workspace.apply(header, initial, staged, phases)
                state = ReplayBaseline(initial.clone())
                state.validate_base(header, initial)
                state.install(header, staged, target)
                if state.version != 1 or not torch.equal(state.values, target):
                    raise AssertionError("Raw-bit reconstruction/version gate failed")
                cases += 1
            # A duplicate index must be rejected before the installed baseline changes.
            target = synthetic_target(initial, 0.2, 17)
            header, _ = workspace.pack(initial, target, preferred=preferred, version=0, phases=CudaPhases())
            indices = workspace.indices16 if preferred == Encoding.BLOCK_LOCAL16 else workspace.indices32
            indices[1] = indices[0]
            state = ReplayBaseline(initial.clone())
            try:
                workspace.apply(header, state.values, torch.empty_like(initial), CudaPhases())
            except ValueError:
                if state.version != 0 or not torch.equal(state.values, initial):
                    raise AssertionError("Rejected patch mutated the installed baseline")
            else:
                raise AssertionError("Malformed sparse indices were accepted")
            cases += 1
            if preferred == Encoding.BLOCK_LOCAL16:
                header, _ = workspace.pack(initial, target, preferred=preferred, version=0, phases=CudaPhases())
                workspace.offsets[-1] += 1
                try:
                    workspace.apply(header, state.values, torch.empty_like(initial), CudaPhases())
                except ValueError:
                    if state.version != 0 or not torch.equal(state.values, initial):
                        raise AssertionError("Rejected block offsets mutated the installed baseline")
                else:
                    raise AssertionError("Malformed block offsets were accepted")
                cases += 1
            # Target corruption at a valid address is caught by the full bit oracle.
            header, _ = workspace.pack(initial, target, preferred=preferred, version=0, phases=CudaPhases())
            workspace.values[0] ^= 1
            staged = torch.empty_like(initial)
            workspace.apply(header, state.values, staged, CudaPhases())
            try:
                state.install(header, staged, target)
            except ValueError:
                if state.version != 0 or not torch.equal(state.values, initial):
                    raise AssertionError("Corrupted target mutated the installed baseline")
            else:
                raise AssertionError("Corrupted target was accepted")
            cases += 1
    torch.cuda.synchronize()
    return cases


def replay_case(width: int, total_bytes: int, chunk_bytes: int, fraction: float, mode: Encoding, control) -> dict:
    rank = dist.get_rank()
    device = torch.device("cuda", torch.cuda.current_device())
    result = {"rank": rank, "elapsed_seconds": 0.0, "payload_bytes": 0, "transactions": []}
    for offset in range(0, total_bytes, chunk_bytes):
        elements = min(chunk_bytes, total_bytes - offset) // width
        initial = (torch.arange(elements, dtype=torch.int32, device=device) * 1103515245 + offset).to(RAW_DTYPES[width])
        state, staging = ReplayBaseline(initial.clone()), torch.empty_like(initial)
        workspace = GpuWorkspace(elements, width, device)
        expected_base = initial
        for version in (1, 2):
            target = synthetic_target(expected_base, fraction, 17 + version)
            torch.cuda.synchronize()
            staging, stats = transaction(state, staging, expected_base, target, workspace, mode, control)
            result["transactions"].append({"chunk_offset": offset, **stats})
            result["elapsed_seconds"] += stats["elapsed_seconds"]
            result["payload_bytes"] += stats["payload_bytes"]
            expected_base = target
    result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    result["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    result["free_bytes"], result["device_total_bytes"] = torch.cuda.mem_get_info()
    return result


def protocol_gate(device: torch.device, control) -> int:
    """Reject one-rank corruption together, then prove the group still progresses."""
    rank, cases = dist.get_rank(), 0
    for mode in (Encoding.DENSE, Encoding.INDEX32, Encoding.BLOCK_LOCAL16):
        for fault in ("sender_base", "receiver_base", "receiver_version", "transmitted_target"):
            initial = torch.arange(2053, dtype=torch.int32, device=device)
            state = ReplayBaseline(initial.clone())
            target = synthetic_target(initial, 0.2, 17)
            if (fault == "sender_base" and rank == 0) or (fault == "receiver_base" and rank == 1):
                state.values[0] ^= 1
            if fault == "receiver_version" and rank == 1:
                state.version = 2
            if fault == "transmitted_target" and rank == 0:
                # The receiver's independently resident target remains uncorrupted.
                target[0] ^= 1
            before, version = state.values.clone(), state.version
            rejected = False
            try:
                transaction(
                    state, torch.empty_like(initial), initial, target, GpuWorkspace(2053, 4, device), mode, control
                )
            except ValueError:
                rejected = True
            outcomes = [None, None]
            dist.all_gather_object(
                outcomes,
                {"rejected": rejected, "unchanged": state.version == version and torch.equal(state.values, before)},
                group=control,
            )
            if not all(row["rejected"] and row["unchanged"] for row in outcomes):
                raise AssertionError(f"Coordinated rejection failed for {mode.value}/{fault}: {outcomes}")
            cases += 1
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-mib", type=int, default=4)
    parser.add_argument("--chunk-mib", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--topology", choices=("same-node", "inter-node"), default="same-node")
    args = parser.parse_args()
    total_bytes, chunk_bytes = args.total_mib * 1024**2, args.chunk_mib * 1024**2
    if not 0 < total_bytes <= MAX_CACHE_BYTES or not 0 < chunk_bytes <= MAX_CHUNK_BYTES or not 1 <= args.repeats <= 10:
        raise ValueError("Expected total <=256 MiB, chunk <=16 MiB and 1-10 repeats")
    if int(os.environ.get("WORLD_SIZE", 0)) != 2:
        raise ValueError("Replay requires exactly two ranks")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    control = dist.new_group(backend="gloo", timeout=timedelta(seconds=120))
    try:
        rank = dist.get_rank()
        device = torch.cuda.get_device_properties(torch.cuda.current_device())
        devices = [None, None]
        dist.all_gather_object(
            devices,
            {
                "rank": rank,
                "host": socket.gethostname(),
                "gpu": device.name,
                "uuid": str(device.uuid),
                "total_memory_bytes": device.total_memory,
            },
            group=control,
        )
        hosts, uuids = {row["host"] for row in devices}, {row["uuid"] for row in devices}
        if len(uuids) != 2 or len(hosts) != (1 if args.topology == "same-node" else 2):
            raise ValueError("Actual host/device placement differs from declared topology")
        # Optional dependency is provided by the pinned CUDA profile.
        import triton

        source = Path(__file__).parents[1]
        provenance = {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "triton": triton.__version__,
            "devices": devices,
            "topology": args.topology,
            "arguments": vars(args),
            "source_sha256": {
                name: hashlib.sha256(path.read_bytes()).hexdigest()
                for name, path in {
                    "entrypoint": Path(__file__),
                    "codec": source / "gpu_sparse_weight_replay.py",
                    "kernels": source / "gpu_sparse_weight_kernels.py",
                }.items()
            },
            "scope": "synthetic contiguous raw-bit chunks, two sequential versions; one chunk resident, total MiB is serial traffic; no Qwen/model loader/whole-model atomic publication",
            "timing_scope": "max rank verified-ACK wall includes full fixture integrity, count sync, packing, control, NCCL, apply and predecessor commit; rank phases overlap",
            "integrity_scope": "independent resident GPU fixtures, not production checksums; invalid/missing ACK fails closed without reconnect recovery",
            "memory_scope": "includes full baseline/staging/fixture buffers and reusable codec workspace; synthetic setup permutation and allocator peaks are also included",
            "control_scope": "Gloo metadata/ACK estimated JSON bytes only; count/status D2H reported separately; equality scalar syncs included in integrity wall time, total driver traffic not measured",
        }
        if rank == 0:
            print("GPU_SPARSE_REPLAY_PROVENANCE " + json.dumps(provenance), flush=True)
        gate = kernel_gate(torch.device("cuda", torch.cuda.current_device()))
        protocol_cases = protocol_gate(torch.device("cuda", torch.cuda.current_device()), control)
        gates = [None, None]
        dist.all_gather_object(
            gates, {"rank": rank, "kernel_cases": gate, "protocol_cases": protocol_cases}, group=control
        )
        if rank == 0:
            print("GPU_SPARSE_KERNEL_GATE " + json.dumps(gates), flush=True)
        modes = (Encoding.DENSE, Encoding.INDEX32, Encoding.BLOCK_LOCAL16)
        for width in RAW_DTYPES:
            for fraction in (0.0, 0.02, 0.03, 0.2, 1.0):
                # Every mode warms its exact shapes before measured rotations.
                for repeat in range(-1, args.repeats):
                    order = modes[repeat % len(modes) :] + modes[: repeat % len(modes)]
                    for mode in order:
                        torch.cuda.reset_peak_memory_stats()
                        stats = replay_case(width, total_bytes, chunk_bytes, fraction, mode, control)
                        gathered = [None, None]
                        dist.all_gather_object(gathered, stats, group=control)
                        if rank == 0 and repeat >= 0:
                            print(
                                "GPU_SPARSE_REPLAY_RESULT "
                                + json.dumps(
                                    {
                                        "dtype": "bfloat16" if width == 2 else "float32",
                                        "fraction": fraction,
                                        "repeat": repeat,
                                        "mode": mode.value,
                                        "exact_reconstruction": True,
                                        "max_rank_elapsed_seconds": max(row["elapsed_seconds"] for row in gathered),
                                        "ranks": gathered,
                                    }
                                ),
                                flush=True,
                            )
        if rank == 0:
            print("GPU_SPARSE_REPLAY_PASS", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
