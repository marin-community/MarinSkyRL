"""Probe-only timing around the unchanged production token processor."""

import mmap
import os
import re
import struct
import time
from pathlib import Path

from skyrl_train.inference_engines.non_agentic_logits_processor import NonAgenticTokenProcessor


class TimedNonAgenticTokenProcessor(NonAgenticTokenProcessor):
    """Record host call time without changing the processor or synchronizing CUDA.

    Each request has a bounded, process-specific 16-byte memory-mapped counter.
    The driver reads these only after the generation panel has drained. Timing
    covers the original processor call, including CUDA launch overhead but not
    asynchronous kernel completion. Counter-writing overhead is excluded.
    """

    def new_req_logits_processor(self, params):
        process = super().new_req_logits_processor(params)
        if process is None:
            return None
        identity = params.extra_args["probe_request_identity"]
        if not re.fullmatch(r"[a-z_]+-[0-9]{2}", identity):
            raise ValueError("Invalid bounded probe request identity")
        directory = Path(params.extra_args["probe_timing_directory"])
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{identity}-{os.getpid()}.bin"
        with path.open("xb") as stream:
            stream.write(bytes(16))
        with path.open("r+b") as stream:
            counter = mmap.mmap(stream.fileno(), 16)
        calls = 0
        nanoseconds = 0

        def timed(output_ids, logits):
            nonlocal calls, nanoseconds
            start = time.perf_counter_ns()
            try:
                return process(output_ids, logits)
            finally:
                nanoseconds += time.perf_counter_ns() - start
                calls += 1
                counter[:] = struct.pack("<QQ", calls, nanoseconds)

        return timed


def timing_receipts(directory: Path) -> list[dict]:
    """Read small counters after all native requests have drained."""
    paths = sorted(directory.glob("*.bin"))
    if len(paths) > 64 * 8:
        raise ValueError("Probe timing counter count exceeded")
    rows = []
    for path in paths:
        raw = path.read_bytes()
        if len(raw) != 16 or path.read_bytes() != raw:
            raise ValueError("Timing counter changed after the drained boundary")
        calls, nanoseconds = struct.unpack("<QQ", raw)
        rows.append({"file": path.name, "calls": calls, "processor_host_nanoseconds": nanoseconds})
    return rows
