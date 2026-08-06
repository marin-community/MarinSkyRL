"""Shared parsing and canonical fields for SkyRL training metrics."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


_ANSI_PATTERN = re.compile(r"\x1b\[[0-9;]*m")
_MIRROR_PREFIX = re.compile(r"WANDB_MIRROR\s+kind=(?P<kind>\S+)\s+step=(?P<step>\d+)\s+metrics=")
_USE_TIS_PATTERN = re.compile(
    r"(?:\btrainer\.algorithm\.use_tis\s*=|\buse_tis[\"']?\s*[:=])\s*(?P<enabled>true|false)\b",
    re.IGNORECASE,
)

REWARD_KEYS = ("reward/avg_raw_reward", "loss/avg_final_rewards")
POLICY_LOSS_KEYS = ("policy/policy_loss", "policy_loss")
GRAD_NORM_KEYS = ("policy/raw_grad_norm", "raw_grad_norm")
ENTROPY_KEYS = ("policy/policy_entropy", "policy_entropy")
TIS_EXACT_MATCH_KEYS = (
    "generate/tis/exact_match_fraction",
    "tis/exact_match_fraction",
    "policy/tis/exact_match_fraction",
)
TIS_LOG_RATIO_ABS_MEAN_KEYS = ("policy/tis/log_ratio_abs_mean", "tis/log_ratio_abs_mean")
TIS_IMPORTANCE_RATIO_MEAN_KEYS = (
    "policy/tis/imp_ratio_mean",
    "tis/imp_ratio_mean",
    "policy/rollout_train_prob_diff_mean",
)
POLICY_LOG_RATIO_ABS_MEAN_KEYS = ("policy/log_ratio_abs_mean", "log_ratio_abs_mean")
POLICY_LOG_RATIO_ABS_P99_KEYS = ("policy/log_ratio_abs_p99", "log_ratio_abs_p99")
POLICY_LOG_RATIO_ABS_MAX_KEYS = ("policy/log_ratio_abs_max", "log_ratio_abs_max")


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


@dataclass(frozen=True)
class TrainingMetricsParseResult:
    records: tuple[TrainingMetricRecord, ...]
    malformed_lines: int


def training_metrics_parse_error(malformed_lines: int) -> str | None:
    if malformed_lines == 0:
        return None
    return f"{malformed_lines} WANDB_MIRROR train lines failed JSON parse"


def strip_ansi(text: str) -> str:
    return _ANSI_PATTERN.sub("", text)


def parse_tis_enabled(log_content: str) -> bool | None:
    """Return the last resolved TIS setting printed in a training log."""
    matches = tuple(_USE_TIS_PATTERN.finditer(strip_ansi(log_content)))
    if not matches:
        return None
    return matches[-1].group("enabled").lower() == "true"


def parse_training_metrics_result(log_content: str) -> TrainingMetricsParseResult:
    """Parse all JSON training ``WANDB_MIRROR`` events from a SkyRL log."""
    records: list[TrainingMetricRecord] = []
    malformed_lines = 0
    for raw_line in log_content.splitlines():
        line = strip_ansi(raw_line)
        match = _MIRROR_PREFIX.search(line)
        if match is None or match.group("kind") != "train":
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
