"""Write reproducible local artifacts for a SkyRL behavior analysis."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from .statistics import (
    MatchedRewardStatistics,
    ContextBin,
    TemporalBin,
    context_summary,
    matched_reward_statistics,
    mean_reward,
    temporal_summary,
)
from .traces import load_trace_records

TEMPORAL_SUMMARY_PATH = Path("Q2_temporal/temporal_summary.json")
COMPARISON_PATH = Path("Q1_behavioral_delta/comparison.json")
OVERLAY_PATH = Path("Q3_temporal_overlay/overlay.json")
ROLLOUT_CONTEXT_PATH = Path("Q4_solve_rate_by_context/rollout_context_summary.json")
EVALUATION_CONTEXT_PATH = Path("Q4_solve_rate_by_context/evaluation_context_summary.json")
TRAINING_METRICS_PATH = Path("Q2_skyrl_metrics")


@dataclass(frozen=True)
class TemporalOverlay:
    rollout_bins: dict[str, TemporalBin]
    baseline_mean_reward: float | None
    post_mean_reward: float | None
    comparison: MatchedRewardStatistics


@dataclass(frozen=True)
class EvaluationContextSummary:
    baseline: dict[str, ContextBin]
    post: dict[str, ContextBin]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, default=asdict, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_training_metrics(training_log_dir: Path, output_dir: Path) -> None:
    command = [
        sys.executable,
        "-m",
        "infra.rl_cleanup.parse_skyrl_metrics",
        str(training_log_dir),
        str(output_dir),
    ]
    subprocess.run(command, check=True)


def _write_index(
    output_dir: Path,
    comparison: MatchedRewardStatistics | None,
    training_log_dir: Path | None,
) -> Path:
    lines = ["# RL Behavioral Analysis", "", "## Q1", ""]
    if comparison is None:
        lines.append("No baseline/post evaluation pair was provided.")
    elif comparison.invalid_for_comparison:
        lines.append("The evaluation task sets are invalid for before/after conclusions: zero common tasks.")
    else:
        lines.append(f"Matched evaluation tasks: {comparison.common_task_count}.")
    lines += ["", "## Q2", "", f"- [Temporal rollout summary]({TEMPORAL_SUMMARY_PATH})"]
    if training_log_dir is not None:
        lines.append(f"- [Training metrics]({TRAINING_METRICS_PATH}/)")
    lines += ["", "## Q3", ""]
    if comparison is None:
        lines.append("No baseline/post evaluation pair was provided.")
    else:
        lines.append(f"- [Evaluation overlay]({OVERLAY_PATH})")
    lines += [
        "",
        "## Q4",
        "",
        f"- [Rollout context summary]({ROLLOUT_CONTEXT_PATH})",
    ]
    if comparison is not None:
        lines.append(f"- [Evaluation context summary]({EVALUATION_CONTEXT_PATH})")
    index_path = output_dir / "INDEX.md"
    index_path.write_text("\n".join([*lines, ""]), encoding="utf-8")
    return index_path


def analyze_local_run(
    *,
    rollout_dir: Path,
    baseline_dir: Path | None,
    post_dir: Path | None,
    training_log_dir: Path | None,
    output_dir: Path,
    bin_hours: float,
) -> Path:
    """Analyze local SkyRL artifacts and return the generated ``INDEX.md`` path."""
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rollouts = load_trace_records(rollout_dir)
    temporal = temporal_summary(rollouts, bin_hours)
    _write_json(output_dir / TEMPORAL_SUMMARY_PATH, temporal)
    _write_json(output_dir / ROLLOUT_CONTEXT_PATH, context_summary(rollouts))

    comparison = None
    if baseline_dir is not None and post_dir is not None:
        baseline = load_trace_records(baseline_dir)
        post = load_trace_records(post_dir)
        comparison = matched_reward_statistics(baseline, post)
        _write_json(
            output_dir / COMPARISON_PATH,
            comparison,
        )
        _write_json(
            output_dir / OVERLAY_PATH,
            TemporalOverlay(
                rollout_bins=temporal.bins,
                baseline_mean_reward=mean_reward(baseline),
                post_mean_reward=mean_reward(post),
                comparison=comparison,
            ),
        )
        _write_json(
            output_dir / EVALUATION_CONTEXT_PATH,
            EvaluationContextSummary(baseline=context_summary(baseline), post=context_summary(post)),
        )
    if training_log_dir is not None:
        _run_training_metrics(training_log_dir, output_dir / TRAINING_METRICS_PATH)

    plan = {
        "rollout_dir": str(rollout_dir),
        "baseline_dir": str(baseline_dir) if baseline_dir else None,
        "post_dir": str(post_dir) if post_dir else None,
        "training_log_dir": str(training_log_dir) if training_log_dir else None,
        "bin_hours": bin_hours,
    }
    _write_json(output_dir / "pipeline_plan.json", plan)
    return _write_index(output_dir, comparison, training_log_dir)
