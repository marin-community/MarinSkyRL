"""Screen a CPU-owned exact XOR/zstd publication against GPU sparse indices.

This probe has one source and one receiver on one device. It times synchronized
local stages and does not measure a network, distributed routing, or useful
overlap. The CPU codec uses raw BF16 words, so NaNs and signed zero are exact.
"""

import argparse
import json
import os
from pathlib import Path
import statistics
import struct
import time
from urllib.parse import urlparse

import numpy as np
import torch
import zstandard


HEADER = struct.Struct("<IIIB")  # base version, target version, changed count, location width


def pack_xor(positions: np.ndarray, xor_words: np.ndarray, base_version: int) -> bytes:
    """Use NeMo-style gap locations and exact XOR words in a versioned packet."""
    positions64 = positions.astype(np.int64, copy=False)
    gaps = np.diff(positions64, prepend=-1) - 1
    location_dtype = np.uint16 if not gaps.size or gaps.max() <= np.iinfo(np.uint16).max else np.uint32
    location_bytes = np.dtype(location_dtype).itemsize
    return (
        HEADER.pack(base_version, base_version + 1, positions.size, location_bytes)
        + gaps.astype(location_dtype).tobytes()
        + xor_words.astype("<u2", copy=False).tobytes()
    )


def encode_xor(current: np.ndarray, baseline: np.ndarray, base_version: int) -> tuple[bytes, int, int]:
    """Encode exact changed BF16 positions and XOR words; return packet and sizes."""
    if current.dtype != np.uint16 or baseline.dtype != np.uint16 or current.shape != baseline.shape:
        raise ValueError("Expected aligned raw BF16 word arrays")
    positions = np.flatnonzero(current != baseline).astype("<i4")
    xor_words = np.bitwise_xor(current[positions], baseline[positions]).astype("<u2")
    raw = pack_xor(positions, xor_words, base_version)
    return zstandard.ZstdCompressor(level=1).compress(raw), positions.size, len(raw)


def decode_xor(packet: bytes, expected_base_version: int) -> tuple[np.ndarray, np.ndarray]:
    """Reject a wrong version before yielding decoded positions and raw XOR words."""
    raw = zstandard.ZstdDecompressor().decompress(packet)
    base_version, target_version, count, location_bytes = HEADER.unpack_from(raw)
    if (base_version, target_version) != (expected_base_version, expected_base_version + 1):
        raise ValueError("Delta base version differs from receiver")
    if location_bytes not in (2, 4) or len(raw) != HEADER.size + count * (location_bytes + 2):
        raise ValueError("Delta length differs from header")
    gaps = np.frombuffer(raw, dtype="<u2" if location_bytes == 2 else "<u4", count=count, offset=HEADER.size)
    positions = np.cumsum(gaps.astype(np.int64) + 1) - 1
    xor_words = np.frombuffer(raw, dtype="<u2", count=count, offset=HEADER.size + count * location_bytes)
    return positions, xor_words


def qualify_cpu() -> dict:
    """Use an independent dense oracle over two updates and raw BF16 edge bits."""
    seed_bits = np.array(
        [0x0000, 0x8000, 0x0001, 0x8001, 0x3F80, 0xBF80, 0x7F80, 0xFF80, 0x7FC0, 0x7FC1, 0x7FFF, 0xFFFF],
        dtype="<u2",
    )
    receiver = seed_bits.copy()
    baseline = seed_bits.copy()
    targets = [
        np.array(
            [0x8000, 0x0000, 0x0002, 0x8002, 0x3F81, 0xBF81, 0x7F80, 0xFF80, 0x7FC1, 0x7FC0, 0x7FFE, 0xFFFE],
            dtype="<u2",
        ),
        np.array(
            [0x0000, 0x8000, 0x0001, 0x8001, 0x3F80, 0xBF80, 0x7F80, 0xFF80, 0x7FFF, 0x7FFF, 0x7FFF, 0xFFFF],
            dtype="<u2",
        ),
    ]
    counts = []
    for base_version, target in enumerate(targets):
        packet, count, _ = encode_xor(target, baseline, base_version)
        try:
            decode_xor(packet, base_version + 1)
        except ValueError:
            pass
        else:
            raise AssertionError("A wrong base version was accepted")
        positions, xor_words = decode_xor(packet, base_version)
        receiver[positions] ^= xor_words
        if not np.array_equal(receiver, target):
            raise AssertionError(f"Receiver bytes differ after update {base_version + 1}")
        baseline[:] = target  # acknowledged commit becomes the next base
        counts.append(count)
    return {"two_update_exact": True, "wrong_base_rejected": True, "changed_counts": counts}


def timed_gpu(action):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = action()
    torch.cuda.synchronize()
    return time.perf_counter() - start, result


def bits_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.view(torch.int16).numpy().view(np.uint16)


def run_gpu_case(size: int, density: float, seed: int) -> dict:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    previous = torch.randn(size, device="cuda", generator=generator).to(torch.bfloat16)
    current = previous.clone()
    chosen = (torch.rand(size, device="cuda", generator=generator) < density).nonzero().view(-1)
    current_bits = current.view(torch.int16)
    current_bits.index_copy_(0, chosen, current_bits.index_select(0, chosen) ^ 1)
    torch.cuda.synchronize()

    gpu_baseline = previous.clone()
    gpu_receiver = previous.clone()
    cpu_baseline = previous.to("cpu").pin_memory()
    cpu_receiver = previous.clone()
    staged = torch.empty(size, dtype=torch.bfloat16, pin_memory=True)
    before_gpu = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    wall_start = time.perf_counter()
    gpu_detect, positions64 = timed_gpu(
        lambda: current.view(torch.int16).ne(gpu_baseline.view(torch.int16)).nonzero().view(-1)
    )
    gpu_pack, packed = timed_gpu(lambda: (positions64.to(torch.int32), current.index_select(0, positions64)))
    gpu_positions, gpu_values = packed
    gpu_apply, _ = timed_gpu(lambda: gpu_receiver.index_copy_(0, gpu_positions.to(torch.int64), gpu_values))
    gpu_commit, _ = timed_gpu(lambda: gpu_baseline.copy_(current))
    gpu_total = time.perf_counter() - wall_start
    if not torch.equal(gpu_receiver.view(torch.int16), current.view(torch.int16)):
        raise AssertionError("GPU sparse receiver differs from dense oracle")

    wall_start = time.perf_counter()
    d2h, _ = timed_gpu(lambda: staged.copy_(current, non_blocking=True))
    started = time.perf_counter()
    current_cpu_bits = staged.view(torch.int16)
    baseline_cpu_bits = cpu_baseline.view(torch.int16)
    positions64 = current_cpu_bits.ne(baseline_cpu_bits).nonzero().view(-1)
    cpu_detect = time.perf_counter() - started
    started = time.perf_counter()
    xor_words = current_cpu_bits[positions64].bitwise_xor(baseline_cpu_bits[positions64]).numpy().view(np.uint16)
    positions = positions64.numpy()
    raw = pack_xor(positions, xor_words, 0)
    cpu_pack = time.perf_counter() - started
    started = time.perf_counter()
    packet = zstandard.ZstdCompressor(level=1).compress(raw)
    compress = time.perf_counter() - started
    started = time.perf_counter()
    received_positions, received_xor = decode_xor(packet, 0)
    decompress = time.perf_counter() - started
    h2d, device_patch = timed_gpu(
        lambda: (
            torch.from_numpy(received_positions.copy()).pin_memory().to("cuda", non_blocking=True),
            torch.from_numpy(received_xor.copy().view(np.int16)).pin_memory().to("cuda", non_blocking=True),
        )
    )
    position_gpu, xor_gpu = device_patch

    def apply_xor():
        destination = cpu_receiver.view(torch.int16)
        index = position_gpu.to(torch.int64)
        destination.index_copy_(0, index, destination.index_select(0, index) ^ xor_gpu)

    cpu_apply, _ = timed_gpu(apply_xor)
    started = time.perf_counter()
    cpu_baseline.copy_(staged)
    cpu_commit = time.perf_counter() - started
    cpu_total = time.perf_counter() - wall_start
    if not torch.equal(cpu_receiver.view(torch.int16), current.view(torch.int16)):
        raise AssertionError("CPU XOR receiver differs from dense oracle")
    if not np.array_equal(bits_numpy(cpu_baseline), bits_numpy(staged)):
        raise AssertionError("Acknowledged CPU baseline did not commit")
    if positions.size != gpu_positions.numel():
        raise AssertionError("CPU and GPU changed counts differ")

    return {
        "values": size,
        "logical_bytes": size * 2,
        "requested_density": density,
        "changed_values": int(positions.size),
        "changed_density": positions.size / size,
        "gpu_index_bytes": int(positions.size * 6 + HEADER.size),
        "cpu_xor_raw_bytes": len(raw),
        "cpu_xor_zstd_bytes": len(packet),
        "gpu_index_seconds": gpu_total,
        "gpu_detect_seconds": gpu_detect,
        "gpu_pack_seconds": gpu_pack,
        "gpu_apply_seconds": gpu_apply,
        "gpu_commit_seconds": gpu_commit,
        "cpu_xor_seconds": cpu_total,
        "d2h_seconds": d2h,
        "cpu_detect_seconds": cpu_detect,
        "cpu_pack_seconds": cpu_pack,
        "compress_seconds": compress,
        "decompress_seconds": decompress,
        "h2d_seconds": h2d,
        "cpu_apply_seconds": cpu_apply,
        "cpu_commit_seconds": cpu_commit,
        "cpu_baseline_bytes": size * 2,
        "gpu_baseline_bytes": size * 2,
        "gpu_transient_peak_bytes": torch.cuda.max_memory_allocated() - before_gpu,
    }


def write_output(payload: str, output: str) -> None:
    if output.startswith("s3://"):
        import boto3  # noqa: PLC0415 - only the GPU probe writes an optional S3 artifact
        from botocore.config import Config  # noqa: PLC0415

        uri = urlparse(output)
        client = boto3.client(
            "s3", endpoint_url=os.environ["AWS_ENDPOINT_URL"], config=Config(s3={"addressing_style": "virtual"})
        )
        client.put_object(Bucket=uri.netloc, Key=uri.path.lstrip("/"), Body=payload.encode())
        return
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--sizes", type=int, nargs="+", default=[3_276_800, 6_553_600, 67_108_864])
    parser.add_argument("--densities", type=float, nargs="+", default=[0.001, 0.005, 0.014, 0.019, 0.024])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = {"qualification": qualify_cpu(), "method": __doc__.strip()}
    if not args.cpu_only:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU screen requires a CUDA device")
        samples = [
            run_gpu_case(size, density, 713 + repeat)
            for size in args.sizes
            for density in args.densities
            for repeat in range(args.repeats)
        ]
        result.update(
            {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "device": torch.cuda.get_device_name(),
                "samples": samples,
                "medians": [
                    {
                        "values": size,
                        "requested_density": density,
                        "gpu_index_seconds": statistics.median(
                            row["gpu_index_seconds"]
                            for row in samples
                            if row["values"] == size and row["requested_density"] == density
                        ),
                        "cpu_xor_seconds": statistics.median(
                            row["cpu_xor_seconds"]
                            for row in samples
                            if row["values"] == size and row["requested_density"] == density
                        ),
                        "cpu_xor_zstd_bytes": statistics.median(
                            row["cpu_xor_zstd_bytes"]
                            for row in samples
                            if row["values"] == size and row["requested_density"] == density
                        ),
                    }
                    for size in args.sizes
                    for density in args.densities
                ],
            }
        )
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        write_output(payload, args.output)
    print(payload, flush=True)


if __name__ == "__main__":
    main()
