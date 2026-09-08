"""Opt-in two-GPU NCCL environment comparison; no model or training is loaded."""

import argparse
import hashlib
import inspect
import json
import os
import socket
import sys
import time
from datetime import timedelta
from pathlib import Path

from tests.process_gang import ProcessGangTimeoutError, launch_process_gang

CASES = ("baseline", "receiver_only", "symmetric")
PAYLOAD_BYTES = (4096, 4 * 1024 * 1024, 32 * 1024 * 1024)
OVERRIDE_KEYS = (
    "VLLM_ALLREDUCE_USE_SYMM_MEM",
    "CUBLAS_WORKSPACE_CONFIG",
    "NCCL_LAUNCH_MODE",
    "NCCL_COLLNET_ENABLE",
    "NCCL_NVLS_ENABLE",
    "NCCL_P2P_NET_DISABLE",
    "NCCL_MIN_NCHANNELS",
    "NCCL_MAX_NCHANNELS",
    "NCCL_PROTO",
    "NCCL_ALGO",
    "NCCL_NTHREADS",
    "NCCL_SOCKET_NTHREADS",
    "VLLM_USE_AOT_COMPILE",
)
MARKER = "WEIGHT_SYNC_ENV_EVENT "


def event(stage: str, rank: int, **values) -> None:
    print(
        MARKER + json.dumps({"stage": stage, "rank": rank, "monotonic_seconds": time.monotonic(), **values}), flush=True
    )


def worker(case: str, output: Path, port: int) -> None:
    # GPU-only imports stay in the opt-in worker; the receipt/timeout auditor is CPU usable.
    import torch
    import torch.distributed as dist
    from vllm.model_executor.layers.batch_invariant import override_envs_for_invariance

    from skyrl_train.distributed.utils import init_custom_process_group

    rank = int(os.environ["LOCAL_RANK"])
    assert torch.cuda.device_count() == 2 and "H100" in torch.cuda.get_device_name(rank)
    event("imported", rank, torch_version=torch.__version__, nccl_version=torch.cuda.nccl.version())
    torch.cuda.set_device(rank)
    if case == "symmetric" or (case == "receiver_only" and rank == 1):
        override_envs_for_invariance()
    event(
        "environment",
        rank,
        values={key: os.environ.get(key) for key in OVERRIDE_KEYS},
        override_source_sha256=hashlib.sha256(inspect.getsource(override_envs_for_invariance).encode()).hexdigest(),
    )
    device = torch.device("cuda", rank)
    # Separate default worlds match the cross-world sync contract. The trainer is
    # device-bound; the receiver is not. Real trainer TP/DP widths are not reproduced.
    event("default_group_start", rank)
    kwargs = {"device_id": device} if rank == 0 else {}
    dist.init_process_group(
        "nccl",
        init_method=f"file://{output / f'default-{rank}'}",
        rank=0,
        world_size=1,
        timeout=timedelta(seconds=20),
        **kwargs,
    )
    event("default_group_end", rank)
    event("custom_group_start", rank)
    group = init_custom_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=2,
        group_name="skyrl-env-probe",
        timeout=timedelta(seconds=20),
    )
    event("custom_group_end", rank)
    for size in PAYLOAD_BYTES:
        pattern = torch.arange(size // 2, dtype=torch.int32, device=device).remainder_(128).to(torch.bfloat16)
        expected = hashlib.sha256(pattern.view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
        tensor = pattern if rank == 0 else torch.zeros_like(pattern)
        event("broadcast_start", rank, payload_bytes=size)
        dist.broadcast(tensor, 0, group=group)
        torch.cuda.synchronize()
        actual = hashlib.sha256(tensor.view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
        event("broadcast_end", rank, payload_bytes=size, sha256=actual, expected_sha256=expected)
        assert actual == expected, "received tensor differs from deterministic payload"
    dist.destroy_process_group(group)
    dist.destroy_process_group()
    event("completed", rank)


def audit_output(text: str, returncode: int | None) -> dict:
    events = [json.loads(line.split(MARKER, 1)[1]) for line in text.splitlines() if MARKER in line]
    completed = {e["rank"] for e in events if e["stage"] == "completed"}
    broadcasts = [e for e in events if e["stage"] == "broadcast_end"]
    expected = {(rank, size) for rank in (0, 1) for size in PAYLOAD_BYTES}
    observed = {(e["rank"], e["payload_bytes"]) for e in broadcasts}
    passed = (
        returncode == 0
        and completed == {0, 1}
        and observed == expected
        and len(broadcasts) == 6
        and all(e["sha256"] == e["expected_sha256"] for e in broadcasts)
    )
    return {"passed": passed, "returncode": returncode, "events": events}


def run_case(case: str, output: Path, *, command: list[str] | None = None, timeout_seconds: float = 120) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ)
    for key in (*OVERRIDE_KEYS, "VLLM_BATCH_INVARIANT"):
        environment.pop(key, None)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    if command is None:
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            str(Path(__file__).resolve()),
            "--worker",
            case,
            "--output",
            str(output),
            "--port",
            str(port),
        ]
    with launch_process_gang(
        command=command,
        working_directory=Path.cwd(),
        environment=environment,
        control_directory=output,
        log_path=output / "workers.log",
        reap_timeout_seconds=5,
        process_description="weight sync environment probe",
    ) as gang:
        try:
            result = gang.wait(timeout_seconds)
            receipt = audit_output(result.output, result.returncode)
            receipt["worker_output"] = result.output
        except ProcessGangTimeoutError as error:
            receipt = audit_output(gang.output(), gang.process.returncode)
            receipt.update(passed=False, timeout=True, error=str(error))
        receipt.update(case=case, leader_pid=gang.process.pid, reaped=gang.process.poll() is not None)
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", choices=CASES)
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    if args.worker:
        try:
            worker(args.worker, args.output.resolve(), args.port)
        except BaseException as error:
            event("error", int(os.environ["LOCAL_RANK"]), error_type=type(error).__name__, error=str(error))
            raise
    else:
        receipts = []
        for case in CASES:
            receipt = run_case(case, args.output.resolve() / case)
            receipts.append(receipt)
            print("WEIGHT_SYNC_ENV_CASE " + json.dumps(receipt), flush=True)
        print("WEIGHT_SYNC_ENV_COMPARISON " + json.dumps(receipts), flush=True)
