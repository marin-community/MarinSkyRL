"""Aggregate-only frozen replay; no models, training, or production publication.

CPU: python -m skyrl_train.entrypoints.benchmark_sparse_weight_replay --total-mib 64
GPU: torchrun --standalone --nproc-per-node=2 -m \
    skyrl_train.entrypoints.benchmark_sparse_weight_replay --nccl --total-mib 64

Dense NCCL starts with resident target bytes. Indexed NCCL includes sender D2H,
CPU encode, payload H2D, broadcast, receiver D2H, CPU decode and full-range H2D.
Both include receiver acknowledgment. Verification of the final GPU bytes is
outside the timed interval. This serial chunk replay excludes model extraction,
loader work, and inter-node links; it cannot establish Snowball publication gains.
The primary time is the maximum rank's acknowledged elapsed time. Phase times
overlap across ranks: receiver metadata wait includes sender encoding, and the
observed collective interval can include waiting for the sender's payload copy.
Neither is pure network service. Do not sum phases across ranks. Metadata size
is a JSON estimate, not Gloo/pickle bytes.
"""

import argparse
from dataclasses import asdict
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import resource
import socket
import time

import numpy as np
import torch
import torch.distributed as dist

from skyrl_train.sparse_weight_replay import (
    EncodedChunk,
    HostBaselineCache,
    MAX_CACHE_BYTES,
    MAX_CHUNK_BYTES,
    WIRE_BYTES,
    decode_chunk,
    encode_chunk,
)


def make_target(base: bytes, dtype: str, fraction: float, seed: int) -> bytes:
    """Flip one bit at independently selected elements, preserving raw dtype width."""
    width = WIRE_BYTES[dtype]
    values = np.frombuffer(base, dtype=f"<u{width}").copy()
    count = round(len(values) * fraction)
    indices = np.random.default_rng(seed).choice(len(values), size=count, replace=False, shuffle=False)
    values[indices] ^= 1
    return values.tobytes()


def rss_high_water_bytes() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def add_times(total: dict, values: dict) -> None:
    for key, value in values.items():
        total[key] = total.get(key, 0) + value


def cpu_replay(cache: HostBaselineCache, keys: list[str], dtype: str, fraction: float, mode: str) -> dict:
    stats = {"payload_bytes": 0, "changed_elements": 0, "encode_seconds": 0.0, "decode_seconds": 0.0}
    for index, key in enumerate(keys):
        base = cache.get(key)
        target = make_target(base, dtype, fraction, index)
        started = time.perf_counter()
        packet, encoding = encode_chunk(base, target, dtype=dtype, base_version=0, target_version=1, mode=mode)
        stats["encode_seconds"] += time.perf_counter() - started
        started = time.perf_counter()
        actual, decoding = decode_chunk(base, packet, installed_version=0)
        stats["decode_seconds"] += time.perf_counter() - started
        if actual != target:
            raise AssertionError("CPU reconstruction differs from independent target bytes")
        stats["payload_bytes"] += packet.payload_bytes
        add_times(stats, encoding)
        add_times(stats, decoding)
    return stats


def timed_copy(destination: torch.Tensor, source: torch.Tensor) -> float:
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    destination.copy_(source, non_blocking=True)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / 1000


def gpu_replay(
    cache: HostBaselineCache, keys: list[str], dtype: str, fraction: float, mode: str, control
) -> list[dict]:
    rank = dist.get_rank()
    stats = {"payload_bytes": 0, "estimated_json_metadata_bytes": 0, "elapsed_seconds": 0.0}
    for index, key in enumerate(keys):
        base = cache.get(key)
        target = make_target(base, dtype, fraction, index)
        # The existing publisher already has the converted tensor on the GPU.
        target_gpu = torch.from_numpy(np.frombuffer(target, dtype=np.uint8).copy()).cuda() if rank == 0 else None
        destination = torch.empty(len(base), dtype=torch.uint8, device="cuda") if rank == 1 else None
        # Fixed-capacity staging is reused for a chunk, not retained per tensor.
        staging = torch.empty(3 * len(base), dtype=torch.uint8, pin_memory=True) if mode == "indexed" else None
        torch.cuda.synchronize()
        dist.barrier()
        started = time.perf_counter()
        if mode == "dense":
            payload = target_gpu if rank == 0 else destination
            transfer_start, transfer_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            transfer_start.record()
            dist.broadcast(payload, 0)
            transfer_end.record()
            transfer_end.synchronize()
            add_times(stats, {"observed_collective_seconds": transfer_start.elapsed_time(transfer_end) / 1000})
            stats["payload_bytes"] += len(base)
        else:
            metadata = [None]
            if rank == 0:
                add_times(stats, {"sender_d2h_seconds": timed_copy(staging[: len(base)], target_gpu)})
                add_times(stats, {"sender_d2h_bytes": len(base)})
                encode_start = time.perf_counter()
                packet, encoding = encode_chunk(
                    base,
                    staging[: len(base)].numpy().tobytes(),
                    dtype=dtype,
                    base_version=0,
                    target_version=1,
                    mode="indexed",
                )
                add_times(stats, {"encode_seconds": time.perf_counter() - encode_start, **encoding})
                header = asdict(packet)
                header.pop("indices")
                header.pop("values")
                header["index_bytes"] = len(packet.indices)
                header["payload_bytes"] = packet.payload_bytes
                metadata[0] = header
            control_start = time.perf_counter()
            dist.broadcast_object_list(metadata, src=0, group=control)
            add_times(stats, {"metadata_control_seconds": time.perf_counter() - control_start})
            header = metadata[0]
            count = header["payload_bytes"]
            stats["estimated_json_metadata_bytes"] += len(json.dumps(header).encode())
            stats["payload_bytes"] += count
            payload = torch.empty(count, dtype=torch.uint8, device="cuda")
            if rank == 0 and count:
                raw = packet.indices + packet.values
                np.copyto(staging[:count].numpy(), np.frombuffer(raw, dtype=np.uint8))
                add_times(stats, {"sender_payload_h2d_seconds": timed_copy(payload, staging[:count])})
                add_times(stats, {"sender_payload_h2d_bytes": count})
            if count:
                transfer_start, transfer_end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                transfer_start.record()
                dist.broadcast(payload, 0)
                transfer_end.record()
                transfer_end.synchronize()
                add_times(stats, {"observed_collective_seconds": transfer_start.elapsed_time(transfer_end) / 1000})
            if rank == 1:
                if count:
                    add_times(stats, {"receiver_payload_d2h_seconds": timed_copy(staging[:count], payload)})
                    add_times(stats, {"receiver_payload_d2h_bytes": count})
                raw = staging[:count].numpy().tobytes()
                index_bytes = header["index_bytes"]
                fields = {key: value for key, value in header.items() if key not in {"index_bytes", "payload_bytes"}}
                packet = EncodedChunk(**fields, indices=raw[:index_bytes], values=raw[index_bytes:])
                decode_start = time.perf_counter()
                reconstructed, decoding = decode_chunk(base, packet, installed_version=0)
                add_times(stats, {"decode_seconds": time.perf_counter() - decode_start, **decoding})
                np.copyto(staging[: len(base)].numpy(), np.frombuffer(reconstructed, dtype=np.uint8))
                add_times(stats, {"receiver_target_h2d_seconds": timed_copy(destination, staging[: len(base)])})
                add_times(stats, {"receiver_target_h2d_bytes": len(base)})
        # Actual GPU completion + receiver acknowledgment included in end-to-end wall.
        torch.cuda.synchronize()
        dist.barrier()
        stats["elapsed_seconds"] += time.perf_counter() - started
        verification = [rank == 0 or destination.cpu().numpy().tobytes() == target]
        dist.broadcast_object_list(verification, src=1, group=control)
        if not verification[0]:
            raise AssertionError("NCCL destination differs from independent target bytes")
        del target_gpu, destination, staging, payload
    stats.update(
        rank=rank,
        rss_high_water_bytes=rss_high_water_bytes(),
        gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
    )
    gathered = [None, None]
    dist.all_gather_object(gathered, stats, group=control)
    return gathered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nccl", action="store_true")
    parser.add_argument("--total-mib", type=int, default=64)
    parser.add_argument("--chunk-mib", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    total_bytes, chunk_bytes = args.total_mib * 1024**2, args.chunk_mib * 1024**2
    if not 0 < total_bytes <= MAX_CACHE_BYTES or not 0 < chunk_bytes <= MAX_CHUNK_BYTES or not 1 <= args.repeats <= 10:
        raise ValueError("Expected total <=256 MiB, chunks <=16 MiB, and 1-10 repeats")
    control, rank = None, 0
    if args.nccl:
        if int(os.environ.get("WORLD_SIZE", 0)) != 2 or int(os.environ.get("LOCAL_WORLD_SIZE", 0)) != 2:
            raise ValueError("NCCL replay requires exactly two ranks on one node")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl", timeout=timedelta(seconds=120))
        control = dist.new_group(backend="gloo", timeout=timedelta(seconds=120))
        rank = dist.get_rank()
    try:
        cache = HostBaselineCache()
        keys = []
        rng = np.random.default_rng(17)
        for offset in range(0, total_bytes, chunk_bytes):
            key = str(offset)
            cache.add(key, rng.bytes(min(chunk_bytes, total_bytes - offset)))
            keys.append(key)
        provenance = {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "module_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "codec_sha256": hashlib.sha256(
                (Path(__file__).parents[1] / "sparse_weight_replay.py").read_bytes()
            ).hexdigest(),
            "cache_bytes": cache.retained_bytes,
            "chunk_bytes": chunk_bytes,
            "rss_high_water_before_bytes": rss_high_water_bytes(),
            "scope": "synthetic serial frozen chunks; no model loader or inter-node transport",
            "timing_scope": "primary=max rank acknowledged elapsed; per-rank phases overlap and must not be summed",
            "metadata_scope": "estimated JSON bytes only; Gloo pickle/framing bytes are not measured",
            "collective_scope": "observed interval may include waiting for sender preparation; not pure network time",
        }
        if args.nccl:
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
            provenance["devices"] = devices
        if rank == 0:
            print("SPARSE_REPLAY_PROVENANCE " + json.dumps(provenance), flush=True)
        for dtype in WIRE_BYTES:
            for fraction in (0.0, 0.02, 0.03, 0.2, 1.0):
                # One unreported warmup of each mode before measuring this case.
                for repeat in range(-1, args.repeats):
                    # Alternate order without changing inputs or using a separate dense codec control.
                    modes = ("dense", "indexed") if repeat % 2 == 0 else ("indexed", "dense")
                    for mode in modes:
                        if args.nccl:
                            torch.cuda.reset_peak_memory_stats()
                            result = gpu_replay(cache, keys, dtype, fraction, mode, control)
                        else:
                            result = cpu_replay(cache, keys, dtype, fraction, mode)
                        if rank == 0 and repeat >= 0:
                            print(
                                "SPARSE_REPLAY_RESULT "
                                + json.dumps(
                                    {
                                        "dtype": dtype,
                                        "fraction": fraction,
                                        "repeat": repeat,
                                        "mode": mode,
                                        "backend": "nccl" if args.nccl else "cpu_codec",
                                        "exact_reconstruction": True,
                                        "max_rank_elapsed_seconds": (
                                            max(row["elapsed_seconds"] for row in result) if args.nccl else None
                                        ),
                                        "results": result,
                                        "rss_high_water_bytes": rss_high_water_bytes(),
                                    }
                                ),
                                flush=True,
                            )
        if rank == 0:
            print("SPARSE_REPLAY_PASS", flush=True)
    finally:
        if args.nccl:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
