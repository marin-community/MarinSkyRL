"""Bounded two-rank NCCL transport selection probe across CoreWeave zones."""

import argparse
import glob
import hashlib
import json
import os
import socket
import time
from datetime import timedelta
from pathlib import Path
from urllib.parse import urlparse

import boto3
import torch
import torch.distributed as dist
from botocore.config import Config

PORT = 49393
STAGING_BYTES = 128 * 2**20


def store(root: str):
    uri = urlparse(root)
    if uri.scheme != "s3" or not uri.netloc or not uri.path.strip("/"):
        raise ValueError("Expected s3://bucket/prefix")
    client = boto3.client(
        "s3", endpoint_url=os.environ["AWS_ENDPOINT_URL"], config=Config(s3={"addressing_style": "virtual"})
    )
    return client, uri.netloc, uri.path.strip("/")


def put_json(client, bucket: str, prefix: str, name: str, value: dict) -> None:
    client.put_object(
        Bucket=bucket, Key=f"{prefix}/{name}", Body=(json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    )


def wait_json(client, bucket: str, prefix: str, name: str, timeout_seconds: int = 600) -> dict:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            response = client.get_object(Bucket=bucket, Key=f"{prefix}/{name}")
            return json.loads(response["Body"].read())
        except client.exceptions.NoSuchKey:
            time.sleep(1)
    raise TimeoutError(f"Timed out waiting for {name} after {timeout_seconds}s")


def primary_ip() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.connect(("10.0.0.1", 9))
        return sock.getsockname()[0]


def stage_cost() -> dict:
    gpu = torch.empty(STAGING_BYTES, dtype=torch.uint8, device="cuda")
    gpu.random_()
    pinned = torch.empty(STAGING_BYTES, dtype=torch.uint8, pin_memory=True)
    pageable = torch.empty(STAGING_BYTES, dtype=torch.uint8)
    result = {
        "bytes": STAGING_BYTES,
        "gpu_to_pinned_seconds": [],
        "pinned_to_gpu_seconds": [],
        "gpu_to_pageable_seconds": [],
    }
    for _ in range(5):
        torch.cuda.synchronize()
        started = time.perf_counter()
        pinned.copy_(gpu, non_blocking=True)
        torch.cuda.synchronize()
        result["gpu_to_pinned_seconds"].append(time.perf_counter() - started)
        started = time.perf_counter()
        gpu.copy_(pinned, non_blocking=True)
        torch.cuda.synchronize()
        result["pinned_to_gpu_seconds"].append(time.perf_counter() - started)
        started = time.perf_counter()
        pageable.copy_(gpu)
        torch.cuda.synchronize()
        result["gpu_to_pageable_seconds"].append(time.perf_counter() - started)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("receiver", "sender"), required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--host")
    args = parser.parse_args()
    client, bucket, prefix = store(args.root)
    rank = 0 if args.role == "receiver" else 1
    torch.cuda.set_device(0)
    result = {
        "role": args.role,
        "rank": rank,
        "hostname": socket.gethostname(),
        "primary_ip": primary_ip(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "nccl": torch.cuda.nccl.version(),
        "device": torch.cuda.get_device_name(0),
        "nccl_env": {
            name: os.environ.get(name)
            for name in ("NCCL_DEBUG", "NCCL_DEBUG_SUBSYS", "NCCL_SOCKET_IFNAME", "NCCL_NET", "NCCL_IB_DISABLE")
        },
    }
    try:
        result["staging"] = stage_cost()
        put_json(client, bucket, prefix, f"{args.role}-prepared.json", {"host": result["primary_ip"]})
        if rank == 0:
            wait_json(client, bucket, prefix, "sender-prepared.json")
            host = result["primary_ip"]
            put_json(client, bucket, prefix, "receiver-ready.json", {"host": host, "port": PORT})
        else:
            host = args.host or wait_json(client, bucket, prefix, "receiver-ready.json")["host"]
        started = time.perf_counter()
        dist.init_process_group(
            "nccl", init_method=f"tcp://{host}:{PORT}", rank=rank, world_size=2, timeout=timedelta(seconds=90)
        )
        result["init_seconds"] = time.perf_counter() - started
        tensor = (
            torch.arange(2**20, dtype=torch.int32, device="cuda").to(torch.uint8)
            if rank == 1
            else torch.zeros(2**20, dtype=torch.uint8, device="cuda")
        )
        started = time.perf_counter()
        dist.broadcast(tensor, src=1)
        torch.cuda.synchronize()
        result["broadcast_seconds"] = time.perf_counter() - started
        result["sha256"] = hashlib.sha256(bytes(tensor.cpu().tolist())).hexdigest()
        result["success"] = True
    except (RuntimeError, OSError, TimeoutError) as error:
        result["success"] = False
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        logs = []
        for path in glob.glob("/tmp/cross-zone-nccl.*.log"):
            data = Path(path).read_bytes()[-128_000:]
            name = f"{args.role}-{Path(path).name}"
            client.put_object(Bucket=bucket, Key=f"{prefix}/{name}", Body=data)
            logs.append(name)
        result["nccl_log_keys"] = logs
        put_json(client, bucket, prefix, f"{args.role}-result.json", result)
        output_dir = os.environ.get("IRIS_OUTPUT_DIR")
        if output_dir:
            Path(output_dir, f"{args.role}-result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result), flush=True)
    if not result["success"]:
        raise RuntimeError(result["error"])


if __name__ == "__main__":
    main()
