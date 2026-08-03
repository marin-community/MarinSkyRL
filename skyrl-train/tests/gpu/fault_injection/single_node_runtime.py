"""Runtime settings shared by single-node, four-GPU fault-injection suites."""

import pytest
import torch


WORLD_SIZE = 4
REAP_TIMEOUT_SECONDS = 10
REQUIRES_FOUR_CUDA_DEVICES = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < WORLD_SIZE,
    reason=f"requires {WORLD_SIZE} CUDA devices",
)
