"""Opt-in ProcessGroupNCCL failure-contract tests.

Run on an otherwise idle node with at least four GPUs:

    uv run --isolated --extra dev --extra vllm \
        pytest -s tests/gpu/fault_injection/nccl_failure_contract.py

The controller launches disposable torchrun gangs that are expected to fail.
This file intentionally does not match pytest's default ``test_*.py`` pattern.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist


WORLD_SIZE = 4
COLLECTIVE_TIMEOUT_SECONDS = 8
PROCESS_TIMEOUT_SECONDS = 45
SKYRL_TRAIN_ROOT = Path(__file__).parents[3]


def _worker(mode: str) -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(seconds=COLLECTIVE_TIMEOUT_SECONDS),
        device_id=device,
    )
    print(f"FAULT_INJECTION_READY mode={mode} rank={rank}", flush=True)

    if mode == "subgroup-divergence":
        from skyrl_train.distributed.fsdp_utils import create_device_mesh

        mesh = create_device_mesh(WORLD_SIZE, fsdp_size=2, ep_size=2)
        group = mesh["ep"].get_group() if rank in (0, 3) else mesh["fsdp"].get_group()
        dist.all_reduce(torch.ones(1, device=device), group=group)
    elif mode == "collective-order":
        tensor = torch.full((1024,), rank, dtype=torch.float32, device=device)
        if rank % 2 == 0:
            dist.all_reduce(tensor)
            dist.broadcast(tensor, src=0)
        else:
            dist.broadcast(tensor, src=0)
            dist.all_reduce(tensor)
    elif mode == "late-rank":
        if rank == 0:
            time.sleep(COLLECTIVE_TIMEOUT_SECONDS * 2)
        dist.all_reduce(torch.ones(1, device=device))
    else:  # pragma: no cover - argparse constrains worker invocations
        raise ValueError(f"unknown fault mode: {mode}")

    print(f"FAULT_INJECTION_UNEXPECTED_COMPLETION mode={mode} rank={rank}", flush=True)
    dist.destroy_process_group()


def _run_fault(mode: str) -> tuple[int, float, str]:
    env = os.environ.copy()
    env.update(
        {
            "SKYRL_WORKER_NCCL_TIMEOUT_IN_S": str(COLLECTIVE_TIMEOUT_SECONDS),
            "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
            "TORCH_NCCL_ENABLE_MONITORING": "1",
            "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC": "5",
            "TORCH_NCCL_DUMP_ON_TIMEOUT": "1",
            "TORCH_NCCL_TRACE_BUFFER_SIZE": "20000",
        }
    )
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={WORLD_SIZE}",
        str(Path(__file__).resolve()),
        "--worker",
        mode,
    ]
    started_at = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=SKYRL_TRAIN_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=PROCESS_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        output, _ = process.communicate()
        pytest.fail(
            f"{mode} did not tear down within {PROCESS_TIMEOUT_SECONDS}s; output:\n{output}",
            pytrace=False,
        )
    return process.returncode, time.monotonic() - started_at, output


@pytest.mark.parametrize("mode", ["subgroup-divergence", "collective-order", "late-rank"])
def test_nccl_fault_terminates_torchrun_gang(mode: str) -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < WORLD_SIZE:
        pytest.skip(f"requires {WORLD_SIZE} CUDA devices")

    returncode, elapsed, output = _run_fault(mode)

    assert f"FAULT_INJECTION_READY mode={mode}" in output
    assert "FAULT_INJECTION_UNEXPECTED_COMPLETION" not in output
    assert returncode != 0, output
    assert elapsed < PROCESS_TIMEOUT_SECONDS
    assert any(signal_text in output.lower() for signal_text in ("timed out", "timeout", "watchdog")), output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--worker",
        choices=("subgroup-divergence", "collective-order", "late-rank"),
        required=True,
    )
    _worker(parser.parse_args().worker)
