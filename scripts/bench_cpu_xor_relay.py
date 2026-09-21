"""Qualify successive acknowledged exact updates between Iris GPU tasks via S3.

The sender and receiver are separate H100 tasks. They generate the same dense
BF16 oracle independently, then the receiver checks every installed raw word.
This is a same-region S3 qualification, not a full trainer or cross-region run.
"""

import argparse
import hashlib
import json
import os
import struct
import time
from urllib.parse import urlparse

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
import numpy as np
import torch
import zstandard

from bench_cpu_xor_delta import decode_xor, pack_xor


EDGE_BITS = np.array(
    [0x0000, 0x8000, 0x0001, 0x8001, 0x3F80, 0xBF80, 0x7F80, 0xFF80, 0x7FC0, 0x7FC1, 0x7FFF, 0xFFFF],
    dtype="<u2",
)
INDEX_HEADER = struct.Struct("<III")
DENSE_HEADER = struct.Struct("<II")


def initial_bits(size: int) -> np.ndarray:
    bits = np.random.default_rng(17017).integers(0, 65536, size, dtype=np.uint16)
    bits[: len(EDGE_BITS)] = EDGE_BITS
    return bits


def next_bits(previous: np.ndarray, version: int, density: float, pattern: str = "lsb") -> np.ndarray:
    target = previous.copy()
    chosen = np.random.default_rng(32000 + version).choice(target.size, round(target.size * density), replace=False)
    xor = (
        np.uint16(1)
        if pattern == "lsb"
        else np.random.default_rng(33000 + version).integers(1, 65536, chosen.size, dtype=np.uint16)
    )
    target[chosen] ^= xor
    target[version - 1] ^= np.uint16(0x8000)  # exercise signed zero/NaN edge words
    return target


def gpu_bf16(bits: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(bits.view(np.int16)).to("cuda").view(torch.bfloat16)


def sha256(bits: np.ndarray) -> str:
    return hashlib.sha256(bits.tobytes()).hexdigest()


class Store:
    def __init__(self, root: str) -> None:
        uri = urlparse(root)
        if uri.scheme != "s3" or not uri.netloc or not uri.path.strip("/"):
            raise ValueError("Root must be an s3://bucket/prefix URL")
        self.bucket = uri.netloc
        self.prefix = uri.path.strip("/")
        self.client = boto3.client(
            "s3", endpoint_url=os.environ["AWS_ENDPOINT_URL"], config=Config(s3={"addressing_style": "virtual"})
        )
        self.last_success_get_seconds = 0.0

    def key(self, name: str) -> str:
        return f"{self.prefix}/{name}"

    def put(self, name: str, data: bytes) -> None:
        self.client.put_object(Bucket=self.bucket, Key=self.key(name), Body=data)

    def get(self, name: str) -> bytes:
        return self.client.get_object(Bucket=self.bucket, Key=self.key(name))["Body"].read()

    def wait(self, name: str, timeout: float = 300) -> bytes:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                started = time.perf_counter()
                data = self.get(name)
                self.last_success_get_seconds = time.perf_counter() - started
                return data
            except ClientError as error:
                if error.response.get("Error", {}).get("Code") not in ("NoSuchKey", "404"):
                    raise
            time.sleep(0.2)
        raise TimeoutError(f"Timed out waiting for {name}")

    def result(self, role: str, mode: str, pattern: str, rows: list[dict], size: int, density: float) -> None:
        payload = {
            "role": role,
            "mode": mode,
            "pattern": pattern,
            "device": torch.cuda.get_device_name(),
            "values": size,
            "density": density,
            "rows": rows,
        }
        self.put(f"{role}-result.json", (json.dumps(payload, indent=2) + "\n").encode())
        print(json.dumps(payload), flush=True)


def sender(store: Store, mode: str, pattern: str, size: int, density: float, updates: int) -> None:
    initial = initial_bits(size)
    baseline = torch.from_numpy(initial.view(np.int16).copy()).pin_memory()
    gpu_baseline = gpu_bf16(initial) if mode == "gpu_index" else None
    staged = torch.empty(size, dtype=torch.int16, pin_memory=True)
    current_bits = initial
    store.wait("receiver-ready.json")
    rows = []
    for version in range(1, updates + 1):
        current_bits = next_bits(current_bits, version, density, pattern)
        source = gpu_bf16(current_bits)
        torch.cuda.synchronize()
        d2h = detect = pack = compress = 0.0
        if mode == "gpu_index":
            started = time.perf_counter()
            positions = source.view(torch.int16).ne(gpu_baseline.view(torch.int16)).nonzero().view(-1)
            torch.cuda.synchronize()
            detect = time.perf_counter() - started
            started = time.perf_counter()
            values = source.view(torch.int16).index_select(0, positions)
            locations_cpu = positions.to(torch.int32).cpu().numpy()
            values_cpu = values.cpu().numpy()
            torch.cuda.synchronize()
            d2h = time.perf_counter() - started
            started = time.perf_counter()
            packet = (
                INDEX_HEADER.pack(version - 1, version, positions.numel())
                + locations_cpu.tobytes()
                + values_cpu.tobytes()
            )
            pack = time.perf_counter() - started
            changed = int(positions.numel())
            raw_bytes = len(packet)
        else:
            started = time.perf_counter()
            staged.copy_(source.view(torch.int16), non_blocking=True)
            torch.cuda.synchronize()
            d2h = time.perf_counter() - started
            if mode == "dense":
                started = time.perf_counter()
                packet = DENSE_HEADER.pack(version - 1, version) + staged.numpy().tobytes()
                pack = time.perf_counter() - started
                changed = size
                raw_bytes = len(packet)
            else:
                started = time.perf_counter()
                positions = staged.ne(baseline).nonzero().view(-1)
                detect = time.perf_counter() - started
                started = time.perf_counter()
                xor = staged[positions].bitwise_xor(baseline[positions]).numpy().view(np.uint16)
                raw = pack_xor(positions.numpy(), xor, version - 1)
                pack = time.perf_counter() - started
                started = time.perf_counter()
                packet = zstandard.ZstdCompressor(level=1).compress(raw)
                compress = time.perf_counter() - started
                changed = int(positions.numel())
                raw_bytes = len(raw)
        started = time.perf_counter()
        store.put(f"delta-{version}.bin", packet)
        put = time.perf_counter() - started
        started = time.perf_counter()
        ack = json.loads(store.wait(f"ack-{version}.json"))
        ack_wait = time.perf_counter() - started
        ack_get = store.last_success_get_seconds
        if ack != {"version": version, "sha256": sha256(current_bits), "exact": True}:
            raise AssertionError(f"Receiver did not acknowledge exact target {version}: {ack}")
        started = time.perf_counter()
        if mode == "gpu_index":
            gpu_baseline.copy_(source)
            torch.cuda.synchronize()
        else:
            baseline.copy_(staged)
        commit = time.perf_counter() - started
        rows.append(
            {
                "version": version,
                "changed": changed,
                "raw_bytes": raw_bytes,
                "encoded_bytes": len(packet),
                "target_sha256": ack["sha256"],
                "d2h_seconds": d2h,
                "detect_seconds": detect,
                "pack_seconds": pack,
                "compress_seconds": compress,
                "put_seconds": put,
                "ack_wait_seconds": ack_wait,
                "ack_get_seconds": ack_get,
                "commit_seconds": commit,
            }
        )
    store.result("sender", mode, pattern, rows, size, density)


def receiver(store: Store, mode: str, pattern: str, size: int, density: float, updates: int) -> None:
    oracle = initial_bits(size)
    installed = gpu_bf16(oracle)
    store.put("receiver-ready.json", b"{}")
    rows = []
    for version in range(1, updates + 1):
        started = time.perf_counter()
        packet = store.wait(f"delta-{version}.bin")
        wait_get = time.perf_counter() - started
        get = store.last_success_get_seconds
        started = time.perf_counter()
        if mode == "cpu_xor":
            positions, values = decode_xor(packet, version - 1)
        elif mode == "gpu_index":
            base, target, count = INDEX_HEADER.unpack_from(packet)
            if (base, target) != (version - 1, version) or len(packet) != INDEX_HEADER.size + count * 6:
                raise ValueError("GPU index packet has wrong version or length")
            positions = np.frombuffer(packet, dtype="<i4", count=count, offset=INDEX_HEADER.size)
            values = np.frombuffer(packet, dtype="<i2", count=count, offset=INDEX_HEADER.size + count * 4)
        else:
            base, target = DENSE_HEADER.unpack_from(packet)
            if (base, target) != (version - 1, version) or len(packet) != DENSE_HEADER.size + size * 2:
                raise ValueError("Dense packet has wrong version or length")
            values = np.frombuffer(packet, dtype="<i2", count=size, offset=DENSE_HEADER.size)
        decode = time.perf_counter() - started
        started = time.perf_counter()
        destination = installed.view(torch.int16)
        values_gpu = torch.from_numpy(values.copy().view(np.int16)).pin_memory().to("cuda", non_blocking=True)
        if mode == "dense":
            destination.copy_(values_gpu)
        else:
            location_gpu = torch.from_numpy(positions.copy()).pin_memory().to("cuda", non_blocking=True)
            if mode == "cpu_xor":
                destination.index_copy_(0, location_gpu, destination.index_select(0, location_gpu) ^ values_gpu)
            else:
                destination.index_copy_(0, location_gpu.to(torch.int64), values_gpu)
        torch.cuda.synchronize()
        apply = time.perf_counter() - started
        oracle = next_bits(oracle, version, density, pattern)
        if not torch.equal(installed.view(torch.int16), gpu_bf16(oracle).view(torch.int16)):
            raise AssertionError(f"Installed raw BF16 words differ at version {version}")
        ack = {"version": version, "sha256": sha256(oracle), "exact": True}
        started = time.perf_counter()
        store.put(f"ack-{version}.json", json.dumps(ack).encode())
        ack_put = time.perf_counter() - started
        rows.append(
            {
                "version": version,
                "encoded_bytes": len(packet),
                "target_sha256": ack["sha256"],
                "wait_get_seconds": wait_get,
                "get_seconds": get,
                "decode_seconds": decode,
                "apply_seconds": apply,
                "ack_put_seconds": ack_put,
                "exact": True,
            }
        )
    store.result("receiver", mode, pattern, rows, size, density)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("sender", "receiver"), required=True)
    parser.add_argument("--mode", choices=("dense", "gpu_index", "cpu_xor"), default="cpu_xor")
    parser.add_argument("--pattern", choices=("lsb", "random_xor"), default="lsb")
    parser.add_argument("--root", required=True)
    parser.add_argument("--size", type=int, default=67_108_864)
    parser.add_argument("--density", type=float, default=0.019)
    parser.add_argument("--updates", type=int, default=2)
    args = parser.parse_args()
    store = Store(args.root)
    (sender if args.role == "sender" else receiver)(
        store, args.mode, args.pattern, args.size, args.density, args.updates
    )


if __name__ == "__main__":
    main()
