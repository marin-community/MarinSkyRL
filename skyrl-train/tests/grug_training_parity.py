"""Shared contract for PyTorch parity with the committed Levanter oracle."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

FP32_OUTPUT_ATOL = 2e-5
FP32_OUTPUT_RTOL = 2e-5
FP32_GRAD_ATOL = 5e-5
FP32_GRAD_RTOL = 5e-5
ROUTE_MARGIN_MULTIPLIER = 10
ORACLE_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "grug_training_oracle"


@dataclass(frozen=True)
class GrugTrainingOracle:
    root: Path
    manifest: dict[str, Any]
    observations: dict[str, np.ndarray]


def load_grug_training_oracle() -> GrugTrainingOracle:
    """Load the small committed oracle into memory."""

    manifest = json.loads((ORACLE_FIXTURE_DIR / "manifest.json").read_text())
    with np.load(ORACLE_FIXTURE_DIR / "observations.npz") as archive:
        observations = {name: archive[name] for name in archive.files}
    return GrugTrainingOracle(root=ORACLE_FIXTURE_DIR, manifest=manifest, observations=observations)


def assert_close(label: str, actual: torch.Tensor, expected: np.ndarray, *, gradient: bool = False) -> None:
    """Apply the committed FP32 output or gradient tolerance policy."""

    expected_tensor = torch.from_numpy(expected).to(device=actual.device, dtype=actual.dtype)
    difference = (actual - expected_tensor).abs().float()
    atol = FP32_GRAD_ATOL if gradient else FP32_OUTPUT_ATOL
    rtol = FP32_GRAD_RTOL if gradient else FP32_OUTPUT_RTOL
    print(f"Grug parity {label}: max_abs={difference.max().item():.8g} mean_abs={difference.mean().item():.8g}")
    torch.testing.assert_close(
        actual,
        expected_tensor,
        atol=atol,
        rtol=rtol,
        msg=lambda message: (
            f"{label}: max_abs={difference.max().item():.8g}, mean_abs={difference.mean().item():.8g}\n{message}"
        ),
    )


def assert_exact_routes(label: str, actual: torch.Tensor, expected: np.ndarray) -> None:
    np.testing.assert_array_equal(actual.detach().cpu().numpy(), expected, err_msg=label)


def assert_route_margin(label: str, route_margin: float) -> None:
    """Require the oracle route decision to be insensitive to output tolerance."""

    minimum = ROUTE_MARGIN_MULTIPLIER * FP32_OUTPUT_ATOL
    if route_margin <= minimum:
        raise AssertionError(f"{label}: route margin {route_margin:.8g} must exceed {minimum:.8g}")
