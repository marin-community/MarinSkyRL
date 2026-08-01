"""Opt-in matrix for healthy model-level EP/FSDP collective schedules.

Run on an otherwise idle node with at least four GPUs::

    uv run --isolated --extra dev --extra ep \
        pytest -s tests/gpu/fault_injection/ep_fsdp_collective_matrix.py

Each case receives a fresh torchrun gang. The matrix covers live and replayed
routing, spread and concentrated replay targets, reentrant activation
checkpointing, and one controlled rank delay at a model-layer boundary.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from tests.collective_schedule_matrix import COLLECTIVE_SCHEDULE_CASES, CollectiveScheduleCase
from tests.torchrun_process import kill_and_reap_torchrun, read_process_output


WORLD_SIZE = 4
RUN_TIMEOUT_SECONDS = 240
REAP_TIMEOUT_SECONDS = 10
COMMUNICATOR_NONBLOCKING_VARIABLES = (
    "TORCH_NCCL_USE_COMM_NONBLOCKING",
    "TORCH_NCCL_NONBLOCKING_TIMEOUT",
)
SKYRL_TRAIN_ROOT = Path(__file__).parents[3]
WORKER_PATH = SKYRL_TRAIN_ROOT / "tests/gpu/gpu_ci/test_ep_fsdp_collective_schedule.py"
REQUIRES_FOUR_CUDA_DEVICES = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < WORLD_SIZE,
    reason=f"requires {WORLD_SIZE} CUDA devices",
)


@dataclass(frozen=True)
class RunResult:
    returncode: int
    output: str


def _run_case(case: CollectiveScheduleCase) -> RunResult:
    environment = os.environ.copy()
    for variable in COMMUNICATOR_NONBLOCKING_VARIABLES:
        environment.pop(variable, None)
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={WORLD_SIZE}",
        str(WORKER_PATH),
        "--case",
        case.name,
    ]
    with tempfile.TemporaryDirectory(prefix=f"skyrl-collective-matrix-{case.name}-") as temporary_dir:
        log_path = Path(temporary_dir) / "torchrun.log"
        with log_path.open("w") as log_file:
            process = subprocess.Popen(
                command,
                cwd=SKYRL_TRAIN_ROOT,
                env=environment,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                returncode = process.wait(timeout=RUN_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                reaped = kill_and_reap_torchrun(process, REAP_TIMEOUT_SECONDS)
                pytest.fail(
                    f"collective schedule case {case.name!r} did not finish within {RUN_TIMEOUT_SECONDS}s; "
                    f"process group reaped={reaped}; output:\n{read_process_output(log_path)}",
                    pytrace=False,
                )
        return RunResult(returncode, read_process_output(log_path))


@pytest.mark.parametrize("case", COLLECTIVE_SCHEDULE_CASES, ids=lambda case: case.name)
@REQUIRES_FOUR_CUDA_DEVICES
def test_model_collective_schedule_survives_matrix_case(case: CollectiveScheduleCase) -> None:
    result = _run_case(case)

    assert result.returncode == 0, result.output
    assert result.output.count(f"MODEL_COLLECTIVE_SCHEDULE_OK case={case.name}") == 1, result.output
