"""One-GPU Grug-shape probe for sparse publication and CPU staging costs.

This intentionally synchronizes between phases. It is a standalone cost probe, not
the publication fast path or a substitute for a distributed end-to-end measurement.
"""

import argparse
import json
import os
from pathlib import Path
import statistics
import time
from urllib.parse import urlparse

import torch


def timed(action):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = action()
    torch.cuda.synchronize()
    return time.perf_counter() - start, result


def run_case(size: int, density: float, repeats: int) -> dict:
    samples = []
    for repeat in range(repeats + 1):
        generator = torch.Generator(device="cuda").manual_seed(713 + repeat)
        baseline = torch.randint(0, 65536, (size,), device="cuda", dtype=torch.int32, generator=generator)
        baseline = baseline.to(torch.int16).view(torch.bfloat16)
        current = baseline.clone()
        changed = round(size * density)
        changed_positions = torch.randperm(size, device="cuda", generator=generator)[:changed]
        old_bits = current.view(torch.int16).index_select(0, changed_positions)
        current.view(torch.int16).index_copy_(0, changed_positions, old_bits ^ 1)
        landing = baseline.clone()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()

        gpu_detect, positions64 = timed(
            lambda: current.view(torch.int16).ne(baseline.view(torch.int16)).nonzero().view(-1)
        )
        gpu_pack, packed = timed(
            lambda: (
                positions64.to(torch.int32),
                current.index_select(0, positions64),
            )
        )
        positions32, values = packed
        gpu_apply, _ = timed(lambda: landing.index_copy_(0, positions32.to(torch.int64), values))
        assert torch.equal(landing.view(torch.int16), current.view(torch.int16))
        gpu_commit, _ = timed(lambda: baseline.copy_(current))
        gpu_peak = torch.cuda.max_memory_allocated() - before

        # The remote candidate keeps the acknowledged image in CPU RAM. Stage the
        # next source image, compare exact BF16 bits, pack, then stage the patch.
        cpu_baseline = landing.to("cpu", non_blocking=False).pin_memory()
        # landing equals current here, so restore the old image for CPU comparison.
        cpu_baseline.view(torch.int16).index_copy_(0, changed_positions.cpu(), old_bits.cpu())
        staged = torch.empty(size, dtype=torch.bfloat16, pin_memory=True)
        d2h, _ = timed(lambda: staged.copy_(current, non_blocking=True))
        start = time.perf_counter()
        cpu_positions64 = staged.view(torch.int16).ne(cpu_baseline.view(torch.int16)).nonzero().view(-1)
        cpu_detect = time.perf_counter() - start
        start = time.perf_counter()
        cpu_positions = cpu_positions64.to(torch.int32).pin_memory()
        cpu_values = staged.index_select(0, cpu_positions64).pin_memory()
        cpu_pack = time.perf_counter() - start
        h2d, device_patch = timed(
            lambda: (
                cpu_positions.to("cuda", non_blocking=True),
                cpu_values.to("cuda", non_blocking=True),
            )
        )
        cpu_landing = cpu_baseline.to("cuda")
        cpu_apply, _ = timed(lambda: cpu_landing.index_copy_(0, device_patch[0].to(torch.int64), device_patch[1]))
        assert torch.equal(cpu_landing.view(torch.int16), current.view(torch.int16))
        start = time.perf_counter()
        cpu_baseline.copy_(staged)
        cpu_commit = time.perf_counter() - start
        sample = {
            "gpu_detect_seconds": gpu_detect,
            "gpu_pack_seconds": gpu_pack,
            "gpu_apply_seconds": gpu_apply,
            "gpu_commit_seconds": gpu_commit,
            "gpu_peak_transient_bytes": gpu_peak,
            "d2h_seconds": d2h,
            "cpu_detect_seconds": cpu_detect,
            "cpu_pack_seconds": cpu_pack,
            "h2d_seconds": h2d,
            "cpu_apply_seconds": cpu_apply,
            "cpu_commit_seconds": cpu_commit,
        }
        if repeat:
            samples.append(sample)
    return {
        "values": size,
        "logical_bytes": size * 2,
        "density": density,
        "changed_values": changed,
        "encoded_bytes": changed * 6 + 8,
        "samples": samples,
        "median": {key: statistics.median(sample[key] for sample in samples) for key in samples[0]},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[6553600, 3276800, 67108864])
    parser.add_argument("--densities", type=float, nargs="+", default=[0.025, 0.23, 0.5])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This probe requires one CUDA GPU")
    torch.set_num_threads(8)
    result = {
        "method": "single H100, sequential synchronized phases; no NIC or NCCL transfer",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "cases": [run_case(size, density, args.repeats) for size in args.sizes for density in args.densities],
    }
    payload = json.dumps(result, indent=2)
    if args.output:
        if args.output.startswith("s3://"):
            import boto3  # noqa: PLC0415 - only a probe writing one small artifact
            from botocore.config import Config  # noqa: PLC0415

            uri = urlparse(args.output)
            client = boto3.client(
                "s3",
                endpoint_url=os.environ["AWS_ENDPOINT_URL"],
                config=Config(s3={"addressing_style": "virtual"}),
            )
            client.put_object(Bucket=uri.netloc, Key=uri.path.lstrip("/"), Body=(payload + "\n").encode())
        else:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(payload + "\n")
    print(payload, flush=True)


if __name__ == "__main__":
    main()
