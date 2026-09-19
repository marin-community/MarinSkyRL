"""Disposable H100 -> CoreWeave object store -> GB200 exact-patch experiment.

This measures a practical inter-region relay. It does not measure direct NCCL or
same-datacenter InfiniBand. The source and destination use GPU-resident BF16 views.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import time
from pathlib import Path

import fsspec
import torch
from skyrl_train.weight_sync.expert_block.sparse_experiment_codec import Patch, apply, bits, encode


def _filesystem(root: str):
    if root.startswith("s3://"):
        import s3fs

        key = os.environ.get("CW_KEY_ID") or os.environ.get("AWS_ACCESS_KEY_ID")
        secret = os.environ.get("CW_KEY_SECRET") or os.environ.get("AWS_SECRET_ACCESS_KEY")
        if not key or not secret:
            raise RuntimeError("CoreWeave object-store credentials are absent")
        return s3fs.S3FileSystem(key=key, secret=secret, client_kwargs={"endpoint_url": "https://cwobject.com"})
    return fsspec.filesystem("file")


def _path(root: str, size_mib: int, density: float, encoding: str) -> str:
    return f"{root.rstrip('/')}/{size_mib}mib-{density:.4f}-{encoding}.bin"


def _sample(size_mib: int, density: float, device: str):
    count = size_mib * 1024 * 1024 // 2
    changed = max(1, round(count * density))
    baseline = torch.zeros(count, dtype=torch.bfloat16, device=device)
    current = baseline.clone()
    positions = torch.arange(changed, dtype=torch.int64, device=device) * (count - 1) // max(changed - 1, 1)
    current.index_fill_(0, positions, 1)
    return baseline, current, changed


def _bytes(tensor: torch.Tensor) -> bytes:
    return tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()


def _write_one(fs, root: str, size_mib: int, density: float, encoding: str, device: str) -> dict:
    baseline, current, changed = _sample(size_mib, density, device)
    torch.cuda.synchronize() if device == "cuda" else None
    start = time.perf_counter()
    if encoding == "dense":
        patch = None
    else:
        patch = encode(current, baseline, encoding)
    torch.cuda.synchronize() if device == "cuda" else None
    encode_seconds = time.perf_counter() - start
    start = time.perf_counter()
    positions_bytes = b"" if patch is None else _bytes(patch.positions)
    values_bytes = _bytes(current if patch is None else patch.values)
    staging_seconds = time.perf_counter() - start
    metadata = {
        "size_mib": size_mib,
        "density": density,
        "encoding": encoding,
        "numel": current.numel(),
        "changed": changed,
        "positions_bytes": len(positions_bytes),
        "values_bytes": len(values_bytes),
    }
    header = json.dumps(metadata, sort_keys=True).encode()
    payload = struct.pack(">Q", len(header)) + header + positions_bytes + values_bytes
    path = _path(root, size_mib, density, encoding)
    start = time.perf_counter()
    fs.pipe_file(path, payload)
    put_seconds = time.perf_counter() - start
    return {
        **metadata,
        "path": path,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "object_bytes": len(payload),
        "encode_seconds": encode_seconds,
        "staging_seconds": staging_seconds,
        "put_seconds": put_seconds,
        "peak_gpu_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else 0,
    }


def _read_one(fs, root: str, size_mib: int, density: float, encoding: str, device: str) -> dict:
    path = _path(root, size_mib, density, encoding)
    start = time.perf_counter()
    payload = fs.cat_file(path)
    get_seconds = time.perf_counter() - start
    header_bytes = struct.unpack(">Q", payload[:8])[0]
    metadata = json.loads(payload[8 : 8 + header_bytes])
    if (metadata["size_mib"], metadata["density"], metadata["encoding"]) != (size_mib, density, encoding):
        raise RuntimeError("Object metadata differs from requested trial")
    offset = 8 + header_bytes
    positions_end = offset + metadata["positions_bytes"]
    values_end = positions_end + metadata["values_bytes"]
    if values_end != len(payload):
        raise RuntimeError("Object length differs from declared payload")
    baseline, expected, changed = _sample(size_mib, density, device)
    destination = baseline.clone()
    start = time.perf_counter()
    if encoding == "dense":
        values = torch.frombuffer(bytearray(payload[offset:]), dtype=torch.uint8).view(torch.bfloat16).to(device)
    else:
        position_dtype = torch.int32 if encoding == "indices" else torch.uint8
        positions = torch.frombuffer(bytearray(payload[offset:positions_end]), dtype=position_dtype).to(device)
        values = torch.frombuffer(bytearray(payload[positions_end:]), dtype=torch.uint8).view(torch.bfloat16).to(device)
    torch.cuda.synchronize() if device == "cuda" else None
    staging_seconds = time.perf_counter() - start
    start = time.perf_counter()
    if encoding == "dense":
        destination.copy_(values)
    else:
        apply(destination, Patch(encoding, metadata["numel"], positions, values))
    torch.cuda.synchronize() if device == "cuda" else None
    apply_seconds = time.perf_counter() - start
    if not bool(torch.equal(bits(destination), bits(expected))):
        raise RuntimeError("Received weights differ at the raw-bit level")
    if changed != metadata["changed"]:
        raise RuntimeError("Changed-value count differs from producer")
    return {
        **metadata,
        "path": path,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "object_bytes": len(payload),
        "get_seconds": get_seconds,
        "staging_seconds": staging_seconds,
        "apply_seconds": apply_seconds,
        "byte_equal": True,
        "peak_gpu_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", required=True, choices=("write", "read"))
    parser.add_argument("--root", required=True)
    parser.add_argument("--sizes-mib", default="1")
    parser.add_argument("--densities", default="0.01")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    fs = _filesystem(args.root)
    sizes = [int(value) for value in args.sizes_mib.split(",")]
    densities = [float(value) for value in args.densities.split(",")]
    output = {
        "schema_version": 1,
        "role": args.role,
        "root": args.root,
        "device": args.device,
        "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
        "source_commit": os.environ.get("SPARSE_EXPERIMENT_SOURCE_COMMIT"),
        "samples": [],
        "complete": False,
    }
    output_path = Path(os.environ.get("IRIS_OUTPUT_DIR", "/tmp")) / f"sparse-expert-relay-{args.role}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        for size in sizes:
            for density in densities:
                for encoding in ("dense", "indices", "bitmap"):
                    torch.cuda.reset_peak_memory_stats() if args.device == "cuda" else None
                    fn = _write_one if args.role == "write" else _read_one
                    sample = fn(fs, args.root, size, density, encoding, args.device)
                    output["samples"].append(sample)
                    output_path.write_text(json.dumps(output, indent=2, sort_keys=True))
                    print(f"SPARSE_RELAY_SAMPLE {args.role} {size}MiB {density:.4f} {encoding}", flush=True)
        output["complete"] = True
    finally:
        output_path.write_text(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
