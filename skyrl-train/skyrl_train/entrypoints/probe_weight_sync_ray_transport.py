"""Diagnose the native Ray custom-group route without loading a model."""

import argparse
import hashlib
import json
import os
import re
import socket
import time
from datetime import timedelta
from pathlib import Path

import ray
import torch
import torch.distributed as dist

from skyrl_train.distributed.utils import get_free_port, init_custom_process_group
from skyrl_train.io.io import write_bytes_atomic
from skyrl_train.weight_sync.readback_diagnostics import ENVIRONMENT_KEYS, network_log_readback

ENV_KEYS = ENVIRONMENT_KEYS
SCRATCH_BYTES = 1024 * 1024


def validate_worlds(rows: list[dict]) -> None:
    if sorted(row["custom_rank"] for row in rows) != [0, 1]:
        raise ValueError("Custom group must cover ranks zero and one")
    if any(row["default_rank"] != 0 or row["default_world"] != 1 or row["custom_world"] != 2 for row in rows):
        raise ValueError("Expected separate singleton default worlds and one two-rank custom world")
    if rows[0]["environment"] != rows[1]["environment"]:
        raise ValueError("Communicator environments differ")
    if any(row["environment"]["VLLM_BATCH_INVARIANT"] not in (None, "0") for row in rows):
        raise ValueError("Timing requires batch invariance off")


class TransportRank:
    def __init__(self, rank: int, backend: str, output: str):
        self.rank, self.backend = rank, backend
        self.output = Path(output) / f"rank-{rank}.json"
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.rows = []
        self.group = None
        self.device = torch.device("cuda", 0) if backend == "nccl" else torch.device("cpu")
        if backend == "nccl":
            torch.cuda.set_device(self.device)
        self.record(
            "before_default_group",
            requested_default_rank=0,
            requested_default_world=1,
            environment={key: os.environ.get(key) for key in ENV_KEYS},
            torchelastic_agent_store=os.environ.get("TORCHELASTIC_USE_AGENT_STORE"),
        )
        # Native workers own independent default worlds before the cross-world group.
        dist.init_process_group(
            backend,
            init_method=f"tcp://127.0.0.1:{get_free_port()}",
            rank=0,
            world_size=1,
            timeout=timedelta(seconds=20),
            device_id=self.device if backend == "nccl" else None,
        )
        self.record("default_group_ready")

    def record(self, stage: str, **values) -> dict:
        row = {
            "stage": stage,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "monotonic": time.monotonic(),
            "custom_rank": self.rank,
            **values,
        }
        self.rows.append(row)
        self.output.write_text(json.dumps(self.rows, sort_keys=True))
        return row

    def describe(self) -> dict:
        return self.record(
            "pre_custom_group",
            default_rank=dist.get_rank(),
            default_world=dist.get_world_size(),
            custom_world=2,
            environment={key: os.environ.get(key) for key in ENV_KEYS},
            address=ray.util.get_node_ip_address(),
            port=get_free_port(),
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            torchelastic_agent_store=os.environ.get("TORCHELASTIC_USE_AGENT_STORE"),
        )

    def initialize(self, address: str, port: int) -> dict:
        started = time.monotonic()
        self.record("custom_group_enter", address=address, port=port, world_size=2)
        self.group = init_custom_process_group(
            backend=self.backend,
            init_method=f"tcp://{address}:{port}",
            world_size=2,
            rank=self.rank,
            group_name="weight-sync-ray-probe",
            timeout=timedelta(seconds=20),
        )
        return self.record("custom_group_ready", seconds=time.monotonic() - started)

    def transfer(self, size: int, repeat: int) -> dict:
        tensor = torch.empty(size, dtype=torch.uint8, device=self.device)
        if self.rank == 0:
            for offset in range(0, size, SCRATCH_BYTES):
                count = min(SCRATCH_BYTES, size - offset)
                pattern = ((torch.arange(count, device=self.device) + offset + repeat) % 251).to(torch.uint8)
                tensor[offset : offset + count].copy_(pattern)
            del pattern
        if self.backend == "nccl":
            torch.cuda.synchronize()
        started = time.monotonic()
        dist.broadcast(tensor, src=0, group=self.group)
        if self.backend == "nccl":
            torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        digest = hashlib.sha256()
        for offset in range(0, size, SCRATCH_BYTES):
            block = tensor[offset : offset + SCRATCH_BYTES].cpu()
            expected = ((torch.arange(block.numel()) + offset + repeat) % 251).to(torch.uint8)
            if not torch.equal(block, expected):
                raise ValueError("Native broadcast changed payload bytes")
            digest.update(block.numpy().tobytes())
        return self.record("payload", bytes=size, repeat=repeat, seconds=elapsed, sha256=digest.hexdigest())

    def finish(self) -> dict:
        if self.group is not None:
            dist.destroy_process_group(self.group)
        dist.destroy_process_group()
        return self.record("groups_destroyed", network_log=network_log_readback())


def run_probe(output: str, backend: str, sizes: tuple[int, ...], repeats: int = 3, measurement_started=None) -> dict:
    if backend not in ("gloo", "nccl") or any(size <= 0 or size > 512 * 1024**2 for size in sizes):
        raise ValueError("Unsupported backend or payload allocation")
    actors = []
    result = {
        "backend": backend,
        "sizes": list(sizes),
        "repeats": repeats,
        "worlds": [],
        "initialization": [],
        "payloads": [],
        "cleanup": [],
        "error": None,
        "scope": "one-host, two Ray actors; no model or learner",
    }
    try:
        actor_type = ray.remote(num_cpus=1, num_gpus=1 if backend == "nccl" else 0)(TransportRank)
        actors = [actor_type.remote(rank, backend, output) for rank in (0, 1)]
        result["worlds"] = ray.get([actor.describe.remote() for actor in actors], timeout=60)
        validate_worlds(result["worlds"])
        sender = result["worlds"][0]
        result["initialization"] = ray.get(
            [actor.initialize.remote(sender["address"], sender["port"]) for actor in actors], timeout=45
        )
        if measurement_started is not None:
            measurement_started(result)
        for size in sizes:
            for repeat in range(repeats):
                rows = ray.get([actor.transfer.remote(size, repeat) for actor in actors], timeout=45)
                if rows[0]["sha256"] != rows[1]["sha256"]:
                    raise ValueError("Sender and receiver payload hashes differ")
                result["payloads"].extend(rows)
        result["cleanup"] = ray.get([actor.finish.remote() for actor in actors], timeout=20)
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        for actor in actors:
            ray.kill(actor, no_restart=True)
        Path(output).mkdir(parents=True, exist_ok=True)
        result["rank_events"] = {}
        for path in Path(output).glob("rank-*.json"):
            if path.stat().st_size > 1024 * 1024:
                raise ValueError("Rank diagnostic exceeds bounded receipt size")
            result["rank_events"][path.name] = json.loads(path.read_text())
        Path(output, "receipt.json").write_text(json.dumps(result, sort_keys=True))
    return result


def attempt_receipt_prefix(prefix: str, attempt_uid: str) -> str:
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,128}", attempt_uid):
        raise ValueError("Native attempt UID must identify immutable diagnostic receipts")
    return prefix.rstrip("/") + "/attempts/" + attempt_uid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--durable-prefix", required=True)
    args = parser.parse_args()
    prefix = attempt_receipt_prefix(args.durable_prefix, os.environ.get("IRIS_ATTEMPT_UID", ""))
    attempt = {"iris_task_id": os.environ.get("IRIS_TASK_ID"), "iris_attempt_uid": os.environ["IRIS_ATTEMPT_UID"]}

    def measurement_started(state):
        marker = {
            **attempt,
            "measurement_started": True,
            "worlds": state["worlds"],
            "initialization": state["initialization"],
        }
        write_bytes_atomic(prefix + "/measurement-started.json", json.dumps(marker, sort_keys=True).encode())

    ray.init(address="local", num_cpus=4, num_gpus=2, include_dashboard=False)
    try:
        result = run_probe(
            args.output, "nccl", (32 * 1024**2, 128 * 1024**2, 512 * 1024**2), measurement_started=measurement_started
        )
        result.update(attempt)
        write_bytes_atomic(prefix + "/receipt.json", json.dumps(result, sort_keys=True).encode())
        if result["error"] is not None:
            raise RuntimeError(result["error"])
        print("RAY_NATIVE_WEIGHT_SYNC_TRANSPORT_PASS ranks=2 payloads=18", flush=True)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
