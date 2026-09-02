"""Test-only snapshot type shared by the FSDP2 and Megatron Grug policy workers."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GrugValidationSnapshot:
    """One policy rank's loaded Grug state as gathered by ``grug_validation_snapshot``."""

    rank: int
    attention_backend: str
    weights: dict[str, torch.Tensor]
