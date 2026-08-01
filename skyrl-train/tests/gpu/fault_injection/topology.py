"""Shared topology for the single-node NCCL fault suites."""

from pathlib import Path

import pytest
import torch


WORLD_SIZE = 4
REAP_TIMEOUT_SECONDS = 10
SKYRL_TRAIN_ROOT = Path(__file__).parents[3]
REQUIRES_FOUR_CUDA_DEVICES = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < WORLD_SIZE,
    reason=f"requires {WORLD_SIZE} CUDA devices",
)
