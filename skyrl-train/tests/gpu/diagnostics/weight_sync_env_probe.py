"""Opt-in two-GPU NCCL environment comparison; no model or training is loaded."""

import argparse
import hashlib
import inspect
import json
import os
import sys
import time
from contextlib import contextmanager
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
EVENT_PATH: Path | None = None


def event(stage: str, rank: int, **values) -> None:
    line = MARKER + json.dumps({"stage": stage, "rank": rank, "monotonic_seconds": time.monotonic(), **values}) + "\n"
    # Each worker has its own authoritative stream; stdout is only a duplicate.
    if EVENT_PATH is not None:
        with EVENT_PATH.open("a") as stream:
            stream.write(line)
    os.write(sys.stdout.fileno(), line.encode())


@contextmanager
def held_rendezvous_store():
    """Own the listening socket continuously across child process startup."""
    import torch.distributed as dist

    store = dist.TCPStore("127.0.0.1", 0, 2, True, timeout=timedelta(seconds=20), wait_for_workers=False)
    try:
        yield store.port
    finally:
        del store


def prepare_cross_world_rendezvous() -> None:
    """Match Ray workers: a fresh custom TCP endpoint has no torchrun agent store."""
    os.environ["TORCHELASTIC_USE_AGENT_STORE"] = "False"


def worker(case: str, output: Path, port: int, backend: str = "nccl") -> None:
    # GPU-only imports stay in the opt-in worker; the receipt/timeout auditor is CPU usable.
    import torch
    import torch.distributed as dist
    from skyrl_train.distributed.utils import init_custom_process_group

    global EVENT_PATH
    rank = int(os.environ["LOCAL_RANK"])
    EVENT_PATH = output / f"rank-{rank}.jsonl"
    override_hash = None
    if backend == "nccl":
        from vllm.model_executor.layers.batch_invariant import override_envs_for_invariance

        assert torch.cuda.device_count() == 2 and "H100" in torch.cuda.get_device_name(rank)
        torch.cuda.set_device(rank)
        if case == "symmetric" or (case == "receiver_only" and rank == 1):
            override_envs_for_invariance()
        override_hash = hashlib.sha256(inspect.getsource(override_envs_for_invariance).encode()).hexdigest()
    event(
        "imported",
        rank,
        backend=backend,
        torch_version=torch.__version__,
        nccl_version=torch.cuda.nccl.version() if backend == "nccl" else None,
    )
    event(
        "environment",
        rank,
        values={key: os.environ.get(key) for key in OVERRIDE_KEYS},
        override_source_sha256=override_hash,
    )
    device = torch.device("cuda", rank) if backend == "nccl" else torch.device("cpu")
    # Separate default worlds match the cross-world sync contract. The trainer is
    # device-bound; the receiver is not. Real trainer TP/DP widths are not reproduced.
    event("default_group_start", rank)
    kwargs = {"device_id": device} if rank == 0 and backend == "nccl" else {}
    dist.init_process_group(
        backend,
        init_method=f"file://{output / f'default-{rank}'}",
        rank=0,
        world_size=1,
        timeout=timedelta(seconds=20),
        **kwargs,
    )
    event("default_group_end", rank)
    prepare_cross_world_rendezvous()
    event("custom_group_start", rank)
    # Explicit clients join the parent-held server; no free-port handoff remains.
    store = dist.TCPStore("127.0.0.1", port, 2, False, timeout=timedelta(seconds=20))
    group = init_custom_process_group(
        backend=backend,
        store=store,
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
        if backend == "nccl":
            torch.cuda.synchronize()
        actual = hashlib.sha256(tensor.view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
        event("broadcast_end", rank, payload_bytes=size, sha256=actual, expected_sha256=expected)
        assert actual == expected, "received tensor differs from deterministic payload"
    dist.destroy_process_group(group)
    dist.destroy_process_group()
    event("completed", rank)


def audit_output(text: str, returncode: int | None) -> dict:
    events, parse_errors = [], []
    for segment in text.split(MARKER)[1:]:
        try:
            value, _ = json.JSONDecoder().raw_decode(segment.lstrip())
            if not isinstance(value, dict) or "rank" not in value or "stage" not in value:
                raise ValueError("event must contain rank and stage")
            events.append(value)
        except (ValueError, TypeError) as error:
            parse_errors.append(str(error))
    completed = {e["rank"] for e in events if e["stage"] == "completed"}
    broadcasts = [e for e in events if e["stage"] == "broadcast_end"]
    expected = {(rank, size) for rank in (0, 1) for size in PAYLOAD_BYTES}
    observed = {(e["rank"], e.get("payload_bytes")) for e in broadcasts}
    passed = (
        returncode == 0
        and not parse_errors
        and completed == {0, 1}
        and observed == expected
        and len(broadcasts) == 6
        and all(e.get("sha256") is not None and e.get("sha256") == e.get("expected_sha256") for e in broadcasts)
    )
    return {"passed": passed, "returncode": returncode, "events": events, "parse_errors": parse_errors}


def run_case(
    case: str, output: Path, *, command: list[str] | None = None, timeout_seconds: float = 120, backend: str = "nccl"
) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ)
    for key in (*OVERRIDE_KEYS, "VLLM_BATCH_INVARIANT"):
        environment.pop(key, None)
    with held_rendezvous_store() as port:
        return _run_case_owned(case, output, port, command, timeout_seconds, backend, environment)


def _run_case_owned(case, output, port, command, timeout_seconds, backend, environment):
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
            "--backend",
            backend,
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
            raw, returncode = result.output, result.returncode
            timed_out = False
        except ProcessGangTimeoutError:
            raw, returncode = gang.output(), gang.process.returncode
            timed_out = True
        # Persist raw material before parsing. Audit failure cannot lose a case.
        rank_streams = {path.name: path.read_text() for path in sorted(output.glob("rank-*.jsonl"))}
        receipt = {
            "case": case,
            "backend": backend,
            "passed": False,
            "returncode": returncode,
            "worker_output": raw,
            "rank_streams": rank_streams,
            "timeout": timed_out,
        }
        (output / "raw-receipt.json").write_text(json.dumps(receipt, indent=2))
        try:
            receipt.update(audit_output("".join(rank_streams.values()) if rank_streams else raw, returncode))
        except Exception as error:
            receipt.update(passed=False, audit_error=f"{type(error).__name__}: {error}")
        if timed_out:
            receipt["passed"] = False
        receipt.update(case=case, leader_pid=gang.process.pid, reaped=gang.process.poll() is not None)
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2))
    return receipt


def run_comparison(output: Path, backend: str = "nccl") -> list[dict]:
    receipts = []
    for case in CASES:
        try:
            receipt = run_case(case, output / case, backend=backend)
        except Exception as error:
            receipt = {
                "case": case,
                "backend": backend,
                "passed": False,
                "orchestration_error": f"{type(error).__name__}: {error}",
            }
        receipts.append(receipt)
        print("WEIGHT_SYNC_ENV_CASE " + json.dumps(receipt), flush=True)
    print("WEIGHT_SYNC_ENV_COMPARISON " + json.dumps(receipts), flush=True)
    return receipts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", choices=CASES)
    parser.add_argument("--port", type=int)
    parser.add_argument("--backend", choices=("nccl", "gloo"), default="nccl")
    args = parser.parse_args()
    if args.worker:
        try:
            worker(args.worker, args.output.resolve(), args.port, args.backend)
        except BaseException as error:
            event("error", int(os.environ["LOCAL_RANK"]), error_type=type(error).__name__, error=str(error))
            raise
    else:
        run_comparison(args.output.resolve(), args.backend)
