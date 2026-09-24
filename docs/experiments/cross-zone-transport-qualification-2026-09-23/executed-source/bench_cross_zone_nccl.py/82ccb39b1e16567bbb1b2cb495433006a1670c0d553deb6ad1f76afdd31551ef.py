"""Checksum-verified two-rank NCCL broadcast, timed through receiver acknowledgement."""

import argparse
import glob
import hashlib
import json
import os
import shutil
import socket
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist


def primary_ip() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as connection:
        connection.connect(("10.0.0.1", 9))
        return connection.getsockname()[0]


def digest(tensors: list[torch.Tensor]) -> str:
    value = hashlib.sha256()
    for tensor in tensors:
        value.update(tensor.cpu().numpy().tobytes())
    return value.hexdigest()


def save(result: dict) -> None:
    directory = Path(os.environ["IRIS_OUTPUT_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{result['role']}-result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("sender", "receiver"), required=True)
    parser.add_argument("--host", help="Receiver's physical IPv4 address; required on sender")
    parser.add_argument("--port", type=int, default=49394)
    parser.add_argument("--payload-mib", type=int, default=1)
    parser.add_argument("--manifest", type=Path, help="Ordered chunk-size JSON; overrides --payload-mib")
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    if args.role == "sender" and not args.host:
        parser.error("--host is required for sender")
    rank = 1 if args.role == "sender" else 0
    sizes = json.loads(args.manifest.read_text()) if args.manifest else [args.payload_mib * 2**20]
    if not sizes or any(not isinstance(size, int) or size < 1 for size in sizes):
        parser.error("Manifest must be a nonempty list of positive byte sizes")
    torch.cuda.set_device(0)
    result = {
        "role": args.role,
        "rank": rank,
        "hostname": socket.gethostname(),
        "primary_ip": primary_ip(),
        "torch": torch.__version__,
        "nccl": torch.cuda.nccl.version(),
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "bytes": sum(sizes),
        "sizes": sizes,
        "repeats": args.repeats,
        "env": {name: os.environ.get(name) for name in (
            "NCCL_NET", "NCCL_IB_DISABLE", "NCCL_IB_HCA", "NCCL_SOCKET_IFNAME",
            "NCCL_P2P_DISABLE", "NCCL_SHM_DISABLE", "NCCL_DEBUG", "NCCL_DEBUG_SUBSYS",
        )},
        "samples": [],
    }
    save(result)
    try:
        host = args.host if rank else result["primary_ip"]
        started = time.monotonic()
        dist.init_process_group(
            "nccl", init_method=f"tcp://{host}:{args.port}", rank=rank, world_size=2,
            timeout=timedelta(seconds=300),
        )
        result["setup_seconds"] = time.monotonic() - started
        for repeat in range(args.repeats):
            if rank:
                tensors = [torch.randint(0, 256, (size,), dtype=torch.uint8, device="cuda") for size in sizes]
                expected = digest(tensors)
            else:
                tensors = [torch.empty(size, dtype=torch.uint8, device="cuda") for size in sizes]
                expected = None
            torch.cuda.synchronize()
            started = time.monotonic()
            for tensor in tensors:
                dist.broadcast(tensor, src=1)
            torch.cuda.synchronize()
            actual = digest(tensors)
            gathered = [None, None]
            dist.all_gather_object(gathered, actual)
            elapsed = time.monotonic() - started
            if gathered[0] != gathered[1] or (expected and actual != expected):
                raise ValueError(f"SHA-256 mismatch: {gathered}")
            result["samples"].append({
                "repeat": repeat, "sha256": actual, "acknowledged_seconds": elapsed,
                "receiver_sha256": gathered[0], "sender_sha256": gathered[1],
            })
            save(result)
        result["success"] = True
    except Exception as error:
        result["success"] = False
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        for path in glob.glob("/tmp/cross-zone-qual-nccl.*.log"):
            shutil.copyfile(path, Path(os.environ["IRIS_OUTPUT_DIR"]) / f"{args.role}-{Path(path).name}")
        save(result)
        print(json.dumps({"role": args.role, "success": result.get("success"), "samples": len(result["samples"])}), flush=True)


if __name__ == "__main__":
    main()
