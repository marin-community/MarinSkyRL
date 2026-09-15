"""Shared Iris settings for Tinker CPU orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

CLUSTER = "cw-rno2a"
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
REPLICAS = 1
MAX_RETRIES = 0
PREEMPTIBLE = False


@dataclass(frozen=True)
class ResourceShape:
    cpu: float
    memory: str
    disk: str


EVALUATION_RESOURCES = ResourceShape(cpu=2.0, memory="8GB", disk="20GB")
SFT_RESOURCES = ResourceShape(cpu=4.0, memory="32GB", disk="50GB")
OPD_RESOURCES = ResourceShape(cpu=2.0, memory="8GB", disk="20GB")
