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
from pathlib import Path

import pytest
import torch

from tests.collective_schedule_matrix import COLLECTIVE_SCHEDULE_CASES, CollectiveScheduleCase
from tests.torchrun_process import (
    NCCL_COMMUNICATOR_NONBLOCKING_VARIABLES,
    TorchrunResult,
    TorchrunTimeoutError,
    launch_torchrun,
)


WORLD_SIZE = 4
RUN_TIMEOUT_SECONDS = 240
REAP_TIMEOUT_SECONDS = 10
SKYRL_TRAIN_ROOT = Path(__file__).parents[3]
WORKER_PATH = SKYRL_TRAIN_ROOT / "tests/gpu/gpu_ci/test_ep_fsdp_collective_schedule.py"
REQUIRES_FOUR_CUDA_DEVICES = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < WORLD_SIZE,
    reason=f"requires {WORLD_SIZE} CUDA devices",
)


def _run_case(case: CollectiveScheduleCase) -> TorchrunResult:
    environment = os.environ.copy()
    for variable in NCCL_COMMUNICATOR_NONBLOCKING_VARIABLES:
        environment.pop(variable, None)
    with launch_torchrun(
        script=WORKER_PATH,
        arguments=("--case", case.name),
        world_size=WORLD_SIZE,
        working_directory=SKYRL_TRAIN_ROOT,
        environment=environment,
        temporary_prefix=f"skyrl-collective-matrix-{case.name}-",
        reap_timeout_seconds=REAP_TIMEOUT_SECONDS,
    ) as gang:
        try:
            return gang.wait(RUN_TIMEOUT_SECONDS)
        except TorchrunTimeoutError as error:
            pytest.fail(f"collective schedule case {case.name!r} did not finish: {error}", pytrace=False)


@pytest.mark.parametrize("case", COLLECTIVE_SCHEDULE_CASES, ids=lambda case: case.name)
@REQUIRES_FOUR_CUDA_DEVICES
def test_model_collective_schedule_survives_matrix_case(case: CollectiveScheduleCase) -> None:
    result = _run_case(case)

    assert result.returncode == 0, result.output
    assert result.output.count(f"MODEL_COLLECTIVE_SCHEDULE_OK case={case.name}") == 1, result.output
