"""Shared parsing and canonical fields for SkyRL training metrics."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


_ANSI_PATTERN = re.compile(r"\x1b\[[0-9;]*m")
_MIRROR_PREFIX = re.compile(r"WANDB_MIRROR\s+kind=(?P<kind>\S+)\s+step=(?P<step>\d+)\s+metrics=")

REWARD_KEYS = ("reward/avg_raw_reward", "loss/avg_final_rewards")
POLICY_LOSS_KEYS = ("policy/policy_loss", "policy_loss")
GRAD_NORM_KEYS = ("policy/raw_grad_norm", "raw_grad_norm")
ENTROPY_KEYS = ("policy/policy_entropy", "policy_entropy")


def metric_value(metrics: dict[str, Any], *names: str) -> Any | None:
    """Return the first present metric from an ordered alias list."""
    for name in names:
        if name in metrics:
            return metrics[name]
    return None


@dataclass(frozen=True)
class TrainingMetricRecord:
    """One structured ``WANDB_MIRROR`` training event."""

    step: int
    metrics: dict[str, Any]

    @property
    def reward(self) -> Any | None:
        return metric_value(self.metrics, *REWARD_KEYS)

    @property
    def policy_loss(self) -> Any | None:
        return metric_value(self.metrics, *POLICY_LOSS_KEYS)

    @property
    def grad_norm(self) -> Any | None:
        return metric_value(self.metrics, *GRAD_NORM_KEYS)

    @property
    def entropy(self) -> Any | None:
        return metric_value(self.metrics, *ENTROPY_KEYS)


@dataclass(frozen=True)
class TrainingMetricsParseResult:
    records: tuple[TrainingMetricRecord, ...]
    malformed_lines: int


def parse_training_metrics_result(log_content: str, *, kind: str = "train") -> TrainingMetricsParseResult:
    """Parse all JSON ``WANDB_MIRROR`` events of ``kind`` from a SkyRL log."""
    records: list[TrainingMetricRecord] = []
    malformed_lines = 0
    for raw_line in log_content.splitlines():
        line = _ANSI_PATTERN.sub("", raw_line)
        match = _MIRROR_PREFIX.search(line)
        if match is None or match.group("kind") != kind:
            continue
        try:
            metrics = json.loads(line[match.end() :].strip())
        except json.JSONDecodeError:
            malformed_lines += 1
            continue
        if not isinstance(metrics, dict):
            malformed_lines += 1
            continue
        step = int(match.group("step"))
        metrics.setdefault("trainer/global_step", step)
        records.append(TrainingMetricRecord(step=step, metrics=metrics))
    return TrainingMetricsParseResult(tuple(records), malformed_lines)


def parse_training_metrics(log_content: str, *, kind: str = "train") -> list[TrainingMetricRecord]:
    """Parse structured training events, omitting malformed event lines."""
    return list(parse_training_metrics_result(log_content, kind=kind).records)
