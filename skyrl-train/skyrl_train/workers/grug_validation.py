"""Test-only snapshot types shared by the FSDP2 and Megatron Grug policy workers."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GrugValidationSnapshot:
    """One policy rank's loaded Grug state as gathered by ``grug_validation_snapshot``."""

    rank: int
    attention_backend: str
    weights: dict[str, torch.Tensor]


@dataclass(frozen=True)
class GrugValidationFingerprintSnapshot:
    """One policy rank's selected Grug tensors represented by compact fingerprints."""

    rank: int
    fingerprints: dict[str, dict[str, object]]
