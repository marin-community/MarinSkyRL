"""Measure exact replay primitives and CUDA allocator peaks without a model."""

import argparse
import json
import os
import socket
import time

import torch

from skyrl_train.weight_sync.bucket_qualification import mark_measurement_once
from skyrl_train.weight_sync.byte_replay import compare_installed_views
from skyrl_train.weight_sync.readback_diagnostics import persist_readback
from skyrl_train.weight_sync.router_replay import compare_widened_router


BUFFER_BYTES = 1 << 30
SCRATCH_BYTES = 512 << 10
REPETITIONS = 3
CASES = ("count_nonzero", "installed_bytes", "router_widening")


def specification():
    return {
        "schema": "replay_memory_probe_spec_v1",
        "devices": 1,
        "buffer_bytes": BUFFER_BYTES,
        "scratch_bytes": SCRATCH_BYTES,
        "router_shape": [256, 2560],
        "repetitions": REPETITIONS,
        "streams": ["default", "separate_load"],
        "cases": list(CASES),
        "model_loading": False,
        "collectives": False,
        "scope": "Native CUDA primitives and allocator only; no full receiver or transport qualification",
    }


def memory():
    free, total = torch.cuda.mem_get_info()
    return {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "device_free_bytes": free,
        "device_total_bytes": total,
    }


def measure(case, stream, buffers, installed, router_source, router_installed):
    torch.cuda.synchronize()
    before = memory()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    # Match the receiver: allocate scratch on the current default stream, then
    # compare on the selected load stream. Persistent input storage predates this.
    scratch = torch.empty(SCRATCH_BYTES, dtype=torch.bool, device="cuda")
    checkpoints = {"after_scratch": memory()}
    ready = torch.cuda.Event()
    ready.record()
    with torch.cuda.stream(stream):
        stream.wait_event(ready)
        if case == "count_nonzero":
            scratch.zero_()
            checkpoints["after_zero"] = memory()
            mismatches = int(torch.count_nonzero(scratch).item())
            checkpoints["after_count_nonzero"] = memory()
            compared = scratch.numel()
        elif case == "installed_bytes":
            result = compare_installed_views(((buffers[0], installed),), scratch, expected_bytes=BUFFER_BYTES)
            compared, mismatches = result.compared_bytes, result.mismatches
        elif case == "router_widening":
            result = compare_widened_router(router_source, router_installed, scratch)
            compared, mismatches = result.compared_bytes, result.mismatches
        else:
            raise ValueError("Unknown probe case")
    torch.cuda.synchronize()
    after = memory()
    elapsed = time.monotonic() - started
    del scratch
    torch.cuda.synchronize()
    released = memory()
    if mismatches != 0:
        raise ValueError("Primitive returned mismatches for identical installed bytes")
    return {
        "case": case,
        "memory_checkpoints": checkpoints,
        "before": before,
        "after": after,
        "released": released,
        "extra_peak_allocated_bytes": after["peak_allocated_bytes"] - before["allocated_bytes"],
        "extra_peak_reserved_bytes": after["peak_reserved_bytes"] - before["reserved_bytes"],
        "seconds": elapsed,
        "compared_bytes": compared,
        "mismatches": mismatches,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--durable-prefix", required=True)
    args = parser.parse_args()
    if args.preview:
        print(json.dumps(specification(), sort_keys=True))
        return
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Probe requires exactly one native CUDA device")
    torch.cuda.set_device(0)
    if torch.__version__ != "2.11.0+cu129":
        raise ValueError("Probe must use the same frozen Torch/CUDA wheel as Snowball")
    identity = {
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "attempt_uid": os.environ["IRIS_ATTEMPT_UID"],
        "torch": torch.__version__,
        "torch_git": torch.version.git_version,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "device_uuid": str(torch.cuda.get_device_properties(0).uuid),
    }
    persist_readback(args.durable_prefix, "inputs", {"identity": identity, "specification": specification()})
    buffers = tuple(torch.zeros(BUFFER_BYTES, dtype=torch.uint8, device="cuda") for _ in range(2))
    installed = buffers[0].clone()
    router_source = torch.arange(256 * 2560, dtype=torch.float32, device="cuda").to(torch.bfloat16).view(256, 2560)
    router_installed = router_source.to(torch.float32)
    torch.cuda.synchronize()
    marker = mark_measurement_once(args.durable_prefix)
    rows = []
    for stream_name in specification()["streams"]:
        stream = torch.cuda.default_stream() if stream_name == "default" else torch.cuda.Stream()
        for case in CASES:
            for repetition in range(REPETITIONS):
                row = measure(case, stream, buffers, installed, router_source, router_installed)
                row.update(stream=stream_name, repetition=repetition, identity=identity)
                rows.append(row)
                persist_readback(args.durable_prefix, f"{stream_name}-{case}-{repetition}", row)
    receipt = {"identity": identity, "specification": specification(), "marker": marker, "rows": rows}
    receipt["all_peaks_within_1mib"] = all(row["extra_peak_allocated_bytes"] <= 1 << 20 for row in rows)
    durable = persist_readback(args.durable_prefix, "complete", receipt)
    print(
        "REPLAY_MEMORY_PROBE_CAPTURE_PASS "
        + json.dumps(
            {"observations": len(rows), "durable": durable, "all_peaks_within_1mib": receipt["all_peaks_within_1mib"]}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
