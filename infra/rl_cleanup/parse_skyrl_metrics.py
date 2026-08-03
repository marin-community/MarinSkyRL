#!/usr/bin/env python3
"""
Parse SkyRL training metrics from console logs and per-trial result.json files.

Scans log files for metric dictionary blocks and vLLM inference engine stats,
and optionally parses per-trial result.json files for turn count analysis.

Outputs:
- A CSV table with all metrics per step
- A CSV table with vLLM engine metrics (aggregated across engines)
- A CSV table with per-trial statistics (from result.json)
- A markdown report with summary statistics
- A reward/errors vs steps plot

Usage:
    # Log serialization is detected automatically for standard and agentic runs:
    python parse_skyrl_metrics.py <log_folder> <output_folder>
    python parse_skyrl_metrics.py /path/to/logs /path/to/results --trace_jobs_dir /path/to/trace_jobs

    # Checkpoint selection optionally intersects metrics with exports on disk:
    python parse_skyrl_metrics.py <log_file_or_dir> <output_folder> \
        --run_dir $WORK/rl_ckpts/<RUN_NAME> --save_every 20
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, TextIO

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from infra.rl_metrics import parse_training_metrics_result, strip_ansi, training_metrics_parse_error

# Harbor writes one TimingInfo block per phase on every trial result, in execution order.
TRIAL_PHASES = ("environment_setup", "agent_setup", "agent_execution", "verifier")

# Which trainer span contains which, as `span: containing span`. `None` marks a span that no
# other span contains, which covers both `step` and the checkpoint, export, evaluation and
# reference-update work that callbacks run between steps. The four trainers under `skyrl-train/`
# that emit `timing/step` do not share one tree, so this declares the deepest of them: a span
# whose container a run did not record resolves to the nearest container it did. That resolution
# only walks up, so a trainer that nests a span declared here as a root goes uncorrected.
#
# Containment is nesting, not a promise about wall clock. A child that runs alongside its parent
# makes the children sum past it, and the summary carries the excess as its own row.
TIMING_PARENTS: dict[str, str | None] = {
    "step": None,
    "generate": "step",
    "wait_for_generation_buffer": "step",
    "postprocess_generator_output": "step",
    "convert_to_training_input": "step",
    "run_training": "step",
    "fwd_logprobs_values_reward": "run_training",
    "apply_reward_kl_penalty": "run_training",
    "compute_advantages_and_returns": "run_training",
    "train_critic_and_policy": "run_training",
    "critic_train": "train_critic_and_policy",
    "policy_train": "train_critic_and_policy",
    "policy_critic_overlap_train": "train_critic_and_policy",
    "sync_weights": "step",
    "init_weight_sync_state": None,
    "save_checkpoints": None,
    "cleanup_old_checkpoints": "save_checkpoints",
    "save_hf_model": None,
    "eval": None,
    "update_ref_with_policy": None,
}

UNATTRIBUTED_ROW = "unattributed"
OVERLAP_ROW = "overlap"
TIMING_PREFIX = "timing/"
STEP_SPAN = "step"


@dataclass(frozen=True)
class TimingSpan:
    """One row of the step-time breakdown, measured against the span that contains it."""

    name: str
    within: str | None
    mean_seconds: float
    share_of_within: float | None
    steps: int


@dataclass(frozen=True)
class TrialPhase:
    """One row of the trial-phase breakdown, measured against the trial wall clock."""

    name: str
    median_seconds: float | None
    mean_seconds: float | None
    total_seconds: float
    share_of_trial: float | None
    trials: int | None


@dataclass(frozen=True)
class TrialPhaseSummary:
    """The trial-phase rows and the trial wall clock their shares divide."""

    phases: list[TrialPhase]
    measured_seconds: float
    measured_trials: int
    total_trials: int


@dataclass(frozen=True)
class CheckpointSelection:
    rewards: dict[int, float]
    ema: dict[int, float]
    best_step: int | None
    available_exports: list[int]
    cap_step: int | None
    eligible: list[int]
    reason: str


@dataclass(frozen=True)
class CheckpointInventory:
    available_exports: list[int]
    cap_step: int | None


class MetricSerialization(StrEnum):
    WANDB_JSON = "wandb-json"
    PYTHON_DICT = "python-dict"


@dataclass(frozen=True)
class ProcessedLog:
    name: str
    metrics: list[dict[str, Any]]
    vllm_metrics: list[dict[str, Any]]
    serialization: MetricSerialization


def extract_metrics_blocks(log_content: str) -> list[dict[str, Any]]:
    """
    Extract metric dictionary blocks from log content.

    Looks for blocks that start with {'async/staleness_max': and end with
    'trainer/global_step': N}
    """
    # Strip ANSI codes first
    content = strip_ansi(log_content)

    # Remove the Ray actor prefix from each line
    # Pattern: (skyrl_entrypoint pid=XXXXX) or similar
    lines = content.split("\n")
    cleaned_lines = []
    for line in lines:
        # Remove Ray actor prefix
        match = re.match(r"\([^)]+\)\s*(.*)", line)
        if match:
            cleaned_lines.append(match.group(1))
        else:
            cleaned_lines.append(line)

    content = "\n".join(cleaned_lines)

    # Find all metric blocks
    # They start with {'async/... and end with 'trainer/global_step': N}
    pattern = r"\{'async/[^}]+?'trainer/global_step':\s*\d+\}"

    metrics_list = []

    for match in re.finditer(pattern, content, re.DOTALL):
        block = match.group(0)

        # Parse the dictionary-like string
        metrics = parse_metrics_block(block)
        if metrics:
            metrics_list.append(metrics)

    return metrics_list


def parse_metrics_block(block: str) -> dict[str, Any] | None:
    """
    Parse a metrics block string into a dictionary.

    The block looks like:
    {'async/staleness_max': 0,
     'async/staleness_mean': '0.0000',
     ...
     'trainer/global_step': 1}
    """
    try:
        # Clean up the block for parsing
        # Replace single quotes with double quotes for JSON
        block = block.replace("'", '"')

        # Handle trailing commas (not valid JSON)
        block = re.sub(r",\s*}", "}", block)

        metrics = json.loads(block)

        # Convert string numbers to floats
        for key, value in metrics.items():
            if isinstance(value, str):
                try:
                    metrics[key] = float(value)
                except ValueError:
                    pass

        return metrics
    except json.JSONDecodeError as e:
        # Try alternative parsing
        try:
            # Use ast.literal_eval for Python dict syntax
            import ast

            metrics = ast.literal_eval(block.replace('"', "'"))

            # Convert string numbers to floats
            for key, value in metrics.items():
                if isinstance(value, str):
                    try:
                        metrics[key] = float(value)
                    except ValueError:
                        pass

            return metrics
        except Exception:
            print(f"Warning: Could not parse metrics block: {e}")
            return None


def extract_wandb_json_metrics(log_content: str) -> list[dict[str, Any]]:
    """
    Extract per-step training metrics from JSON ``WANDB_MIRROR`` events.

    Current standard and agentic SkyRL runs emit one JSON event per train step:
        (skyrl_entrypoint pid=...)<ANSI> ... WANDB_MIRROR kind=train step=N metrics={...}<ANSI>
    All present keys are retained; the pass@k suffix remains dependent on
    n_samples_per_prompt.

    Returns a list of dicts (one per train step), each carrying every key in the JSON dict
    (e.g. trainer/global_step, reward/avg_raw_reward, reward/avg_pass_at_*, policy/policy_entropy,
    policy/raw_grad_norm, policy/policy_loss, policy/ppo_clip_ratio, policy/log_ratio_abs_*,
    policy/n_tokens_dp_gt_*pct, loss/avg_raw_advantages, timing/*, ...).
    """
    result = parse_training_metrics_result(log_content)
    if parse_error := training_metrics_parse_error(result.malformed_lines):
        print(f"  Warning: {parse_error}")
    return [record.metrics for record in result.records]


def _checkpoint_rewards(metrics: list[dict[str, Any]]) -> dict[int, float]:
    rewards: dict[int, float] = {}
    for metric in metrics:
        step = metric.get("trainer/global_step")
        reward = metric.get("reward/avg_raw_reward")
        if step is None or reward is None:
            continue
        try:
            normalized_step = int(step)
            normalized_reward = float(reward)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid checkpoint reward metric: step={step!r}, reward={reward!r}") from exc
        rewards.setdefault(normalized_step, normalized_reward)
    return rewards


def _trailing_five_ema(rewards: dict[int, float]) -> dict[int, float]:
    alpha = 1 / 3
    ema: dict[int, float] = {}
    previous = rewards[min(rewards)]
    for step in sorted(rewards):
        previous = alpha * rewards[step] + (1 - alpha) * previous
        ema[step] = previous
    return ema


def _checkpoint_inventory(run_dir: Path | None) -> CheckpointInventory:
    """Return exported checkpoint steps and the durable save cap, when available."""
    if run_dir is None:
        return CheckpointInventory([], None)

    available: list[int] = []
    exports_dir = run_dir / "exports"
    if exports_dir.is_dir():
        for child in exports_dir.iterdir():
            match = re.match(r"global_step_(\d+)$", child.name)
            if match and child.is_dir():
                available.append(int(match.group(1)))
        available.sort()

    cap_step: int | None = None
    cap_file = run_dir / "latest_ckpt_global_step.txt"
    if cap_file.is_file():
        try:
            cap_step = int(cap_file.read_text().strip())
        except (ValueError, OSError) as exc:
            raise ValueError(f"Could not read a checkpoint step from {cap_file}: {exc}") from exc
    return CheckpointInventory(available, cap_step)


def _eligible_checkpoint_steps(
    ema: dict[int, float], available: list[int], cap_step: int | None, save_every: int
) -> list[int]:
    candidates = available if available else sorted(ema)
    return [
        step
        for step in candidates
        if step in ema
        and step % save_every == 0
        and step >= 2 * save_every
        and (cap_step is None or step <= cap_step)
    ]


def select_best_checkpoint(
    metrics: list[dict[str, Any]],
    run_dir: Path | None = None,
    save_every: int = 20,
) -> CheckpointSelection:
    """Select a checkpoint from parsed agentic or standard GRPO metrics.

    When ``run_dir`` is provided, candidates are intersected with this layout:
        <run_dir>/exports/global_step_<N>/policy/<weights>
        <run_dir>/latest_ckpt_global_step.txt

    Without ``run_dir``, saved-aligned steps in the parsed metrics are candidates and no cap is applied.

    Trailing-5 EMA selection over reward/avg_raw_reward:
      - reward keyed by trainer/global_step, first-seen wins
      - trailing-5 EMA: alpha = 1/3; EMA_n = alpha*r_n + (1-alpha)*EMA_{n-1}, EMA_1 = r_1
      - eligible saved-aligned steps = multiples of save_every, excluding the FIRST save
        (s >= 2*save_every), pick max EMA among them
      - selection is CAPPED at latest_ckpt_global_step.txt

    Returns the chosen step, EMA table, eligibility information, and diagnostics.
    """
    rewards = _checkpoint_rewards(metrics)
    if not rewards:
        return CheckpointSelection(rewards, {}, None, [], None, [], "No rewards found in parsed metrics")

    ema = _trailing_five_ema(rewards)
    inventory = _checkpoint_inventory(run_dir)
    available = inventory.available_exports
    cap_step = inventory.cap_step
    eligible = _eligible_checkpoint_steps(ema, available, cap_step, save_every)

    if not eligible:
        reason = (
            "No saved-aligned checkpoint eligible (after cap + exports intersection). "
            f"available_exports={available}, cap_step={cap_step}, save_every={save_every}"
        )
        return CheckpointSelection(rewards, ema, None, available, cap_step, eligible, reason)

    best = max(eligible, key=lambda s: ema[s])
    scope = f"exports <= cap_step={cap_step}" if cap_step is not None else "steps"
    reason = f"highest trailing-5 EMA ({ema[best]:.4f}) among saved-aligned {scope}"
    return CheckpointSelection(rewards, ema, best, available, cap_step, eligible, reason)


def print_best_checkpoint(selection: CheckpointSelection, save_every: int) -> None:
    """Pretty-print the best-checkpoint selector output (EMA table + chosen step)."""
    print("\n" + "=" * 60)
    print("BEST-CHECKPOINT SELECTOR (GRPO, trailing-5 EMA)")
    print("=" * 60)

    rewards = selection.rewards
    ema = selection.ema
    available = selection.available_exports
    cap_step = selection.cap_step
    eligible = set(selection.eligible)
    best = selection.best_step

    print(f"  save_every (hf_save_interval): {save_every}")
    print(f"  available exports: {available}")
    print(f"  cap (latest_ckpt_global_step.txt): {cap_step}")
    print()
    print(f"  {'step':>6} | {'reward':>10} | {'EMA':>10} | export? | eligible?")
    print("  " + "-" * 56)
    for s in sorted(ema):
        has_export = "  yes  " if (not available or s in available) else "  no   "
        elig = " yes" if s in eligible else ""
        star = " <-- BEST" if s == best else ""
        print(f"  {s:>6} | {rewards.get(s, float('nan')):>10.4f} | {ema[s]:>10.4f} | {has_export} |{elig}{star}")
    print()
    if best is not None:
        print(f"  CHOSEN STEP: {best}  ({selection.reason})")
        print(f"  export path: exports/global_step_{best}/policy/")
    else:
        print(f"  NO STEP CHOSEN: {selection.reason}")


def _write_best_checkpoint_report(output: TextIO, selection: CheckpointSelection) -> None:
    output.write("## Best Checkpoint (trailing-5 EMA of reward/avg_raw_reward)\n\n")
    best = selection.best_step
    if best is not None:
        output.write(f"**Chosen step: `{best}`** — {selection.reason}\n\n")
        output.write(f"Export path: `exports/global_step_{best}/policy/`\n\n")
    else:
        output.write(f"No step chosen: {selection.reason}\n\n")
    output.write(f"- available exports: `{selection.available_exports}`\n")
    output.write(f"- cap (latest_ckpt_global_step.txt): `{selection.cap_step}`\n\n")
    ema = selection.ema
    rewards = selection.rewards
    eligible = set(selection.eligible)
    if ema:
        output.write("| Step | Reward | EMA | Eligible |\n|------|--------|-----|----------|\n")
        for step in sorted(ema):
            output.write(
                f"| {step} | {rewards.get(step, float('nan')):.4f} | {ema[step]:.4f} | "
                f"{'yes' if step in eligible else ''} |\n"
            )
        output.write("\n")


def extract_batch_errors(log_content: str) -> dict[int, dict[str, float]]:
    """
    Extract per-step batch error statistics from log content.

    Parses "Exception breakdown" and "Batch generation complete" lines,
    groups them by training step (using "Step N:" markers), and returns
    averaged error counts per step.

    Returns:
        {step_number: {"AgentTimeoutError": avg_per_batch,
                        "ContextLengthExceededError": avg_per_batch,
                        "total_batches": N, "total_failed": M, ...}}
    """
    content = strip_ansi(log_content)

    # Remove Ray actor prefix from each line
    lines = content.split("\n")
    cleaned_lines = []
    for line in lines:
        match = re.match(r"\([^)]+\)\s*(.*)", line)
        cleaned_lines.append(match.group(1) if match else line)

    # Walk through lines, track current step, collect events
    step_marker_re = re.compile(r"Step (\d+):")
    exception_re = re.compile(r"Exception breakdown: (\{.*\})")
    batch_re = re.compile(
        r"Batch generation complete: (\d+)/(\d+) successful, "
        r"(\d+) failed instances, (\d+) masked"
    )

    # Events before step 1's marker belong to step 1
    current_step = 1
    # {step: {"batches": [...], "exceptions": [...]}}
    step_events: dict[int, dict[str, list]] = defaultdict(lambda: {"batches": [], "exceptions": []})

    for line in cleaned_lines:
        sm = step_marker_re.search(line)
        if sm:
            current_step = int(sm.group(1))
            continue

        em = exception_re.search(line)
        if em:
            try:
                import ast

                exc_dict = ast.literal_eval(em.group(1))
                step_events[current_step]["exceptions"].append(exc_dict)
            except Exception:
                pass
            continue

        bm = batch_re.search(line)
        if bm:
            step_events[current_step]["batches"].append(
                {
                    "successful": int(bm.group(1)),
                    "total": int(bm.group(2)),
                    "failed": int(bm.group(3)),
                    "masked": int(bm.group(4)),
                }
            )

    # Aggregate per step
    result = {}
    for step, events in step_events.items():
        batches = events["batches"]
        exceptions = events["exceptions"]
        n_batches = len(batches)
        if n_batches == 0:
            continue

        # Sum up all exception types across batches in this step
        exc_totals: dict[str, int] = defaultdict(int)
        for exc in exceptions:
            for exc_type, count in exc.items():
                exc_totals[exc_type] += count

        total_failed = sum(b["failed"] for b in batches)
        total_masked = sum(b["masked"] for b in batches)
        total_successful = sum(b["successful"] for b in batches)
        total_instances = sum(b["total"] for b in batches)

        agg: dict[str, float] = {
            "batch_errors/total_batches": n_batches,
            "batch_errors/total_instances": total_instances,
            "batch_errors/total_successful": total_successful,
            "batch_errors/total_failed": total_failed,
            "batch_errors/total_masked": total_masked,
        }
        for exc_type, total in exc_totals.items():
            agg[f"batch_errors/avg_{exc_type}"] = total / n_batches
            agg[f"batch_errors/total_{exc_type}"] = total

        result[step] = agg

    return result


def find_trace_jobs_dir(log_folder: Path) -> Path | None:
    """
    Auto-discover the trace_jobs directory relative to the log folder.

    Expected experiment structure:
        <experiment_root>/logs/          <- log_folder
        <experiment_root>/<run_name>/trace_jobs/  <- what we're looking for

    Returns the trace_jobs path if found, else None.
    """
    parent = log_folder.parent  # experiment root
    for child in parent.iterdir():
        if child.is_dir() and child.name != "logs":
            candidate = child / "trace_jobs"
            if candidate.is_dir():
                return candidate
            # Also check one level deeper (run_name/run_name/trace_jobs)
            for grandchild in child.iterdir():
                if grandchild.is_dir():
                    candidate = grandchild / "trace_jobs"
                    if candidate.is_dir():
                        return candidate
    return None


def resolve_trace_jobs_dir(log_folder: Path, requested_dir: str | None) -> Path | None:
    """Resolve an adjacent trace directory, rejecting a missing explicit path."""
    if requested_dir:
        candidate = Path(requested_dir)
        if not candidate.is_dir():
            raise ValueError(f"Explicit trace_jobs directory does not exist: {candidate}")
        return candidate
    return find_trace_jobs_dir(log_folder)


def span_duration(span: Mapping[str, Any] | None) -> float | None:
    """Return the seconds between `started_at` and `finished_at`, or None when they cannot be read.

    Harbor writes that pair on each phase's TimingInfo block and on the trial result itself, so
    the same arithmetic measures a phase and the trial that contains it.
    """
    if not isinstance(span, Mapping):
        return None
    started_at = span.get("started_at")
    finished_at = span.get("finished_at")
    if not started_at or not finished_at:
        return None
    try:
        return (datetime.fromisoformat(finished_at) - datetime.fromisoformat(started_at)).total_seconds()
    except (TypeError, ValueError):
        return None


def parse_result_files(trace_jobs_dir: Path) -> list[dict[str, Any]]:
    """
    Parse all result.json files in trace_jobs directory.

    Extracts per-trial:
      - task_name, trial_name
      - n_episodes (turn count)
      - exception_type (or None if no exception)
      - reward (or None)
      - n_input_tokens, n_output_tokens
      - a duration for each phase in TRIAL_PHASES, and for the trial that contains them

    Robust to individual missing or malformed files.

    Returns list of dicts, one per successfully parsed trial.
    """
    results = []
    task_dirs = [d for d in trace_jobs_dir.iterdir() if d.is_dir()]
    n_skipped = 0

    for task_dir in task_dirs:
        result_path = task_dir / "result.json"
        if not result_path.exists():
            n_skipped += 1
            continue

        try:
            with open(result_path, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            n_skipped += 1
            continue

        trial = {
            "task_name": data.get("task_name", ""),
            "trial_name": data.get("trial_name", ""),
        }

        # Turn count from agent metadata
        agent_result = data.get("agent_result") or {}
        metadata = agent_result.get("metadata") or {}
        trial["n_episodes"] = metadata.get("n_episodes")
        trial["n_input_tokens"] = agent_result.get("n_input_tokens")
        trial["n_output_tokens"] = agent_result.get("n_output_tokens")

        # Exception info
        exc_info = data.get("exception_info") or {}
        trial["exception_type"] = exc_info.get("exception_type")

        # Reward
        verifier_result = data.get("verifier_result") or {}
        rewards = verifier_result.get("rewards") or {}
        trial["reward"] = rewards.get("reward")

        # Timing
        trial["trial_duration"] = span_duration(data)
        for phase_name in TRIAL_PHASES:
            trial[f"{phase_name}_duration"] = span_duration(data.get(phase_name))

        results.append(trial)

    if n_skipped > 0:
        print(f"  Warning: Skipped {n_skipped} missing/malformed result.json files")

    return results


def summarize_trial_phases(df: pd.DataFrame) -> TrialPhaseSummary | None:
    """
    Summarize each harbor phase against the trial wall clock that contains it.

    Shares divide summed phase time by summed trial duration, and both sides of that division
    come from the trials that recorded their own start and finish: a trial that timed a phase
    and then never finished, which is what a run cut short by a sandbox quota produces, would
    otherwise reach the numerator without reaching the denominator. Medians, means and totals
    still cover every trial that timed the phase, and each phase carries its own trial count.

    A trailing `unattributed` row carries the trial time no phase covers. Harbor times each
    phase independently and does not hold them to the trial's own span, so where they sum past
    it an `overlap` row carries the excess instead.

    Returns None when no phase was measurable.
    """
    measured = df[df["trial_duration"].notna()]
    measured_seconds = float(measured["trial_duration"].sum())

    phases: list[TrialPhase] = []
    attributed_seconds = 0.0
    for name in TRIAL_PHASES:
        durations = df[f"{name}_duration"].dropna()
        if durations.empty:
            continue
        phase_seconds = float(measured[f"{name}_duration"].sum())
        attributed_seconds += phase_seconds
        phases.append(
            TrialPhase(
                name=name,
                median_seconds=float(durations.median()),
                mean_seconds=float(durations.mean()),
                total_seconds=float(durations.sum()),
                share_of_trial=phase_seconds / measured_seconds if measured_seconds > 0 else None,
                trials=len(durations),
            )
        )

    if not phases:
        return None

    if measured_seconds > 0:
        remainder = measured_seconds - attributed_seconds
        if remainder < 0:
            phases.append(TrialPhase(OVERLAP_ROW, None, None, -remainder, None, None))
        else:
            phases.append(TrialPhase(UNATTRIBUTED_ROW, None, None, remainder, remainder / measured_seconds, None))

    return TrialPhaseSummary(phases, measured_seconds, len(measured), len(df))


def compute_trial_stats(trials: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Compute aggregate statistics from parsed trial results.

    Returns a dict with:
      - Overall turn count stats (mean, median, min, max, std)
      - Turn count by exception type
      - Turn count by reward outcome (success vs failure)
      - Exception type distribution
      - Per-phase durations and their share of trial wall clock
    """
    if not trials:
        return {}

    df = pd.DataFrame(trials)
    stats: dict[str, Any] = {"total_trials": len(df)}

    phase_durations = summarize_trial_phases(df)
    if phase_durations:
        stats["phase_durations"] = phase_durations

    # Turn count stats (filter out None)
    turns = df["n_episodes"].dropna()
    if len(turns) > 0:
        stats["turn_count"] = {
            "mean": float(turns.mean()),
            "median": float(turns.median()),
            "min": int(turns.min()),
            "max": int(turns.max()),
            "std": float(turns.std()),
            "count": len(turns),
        }
    else:
        stats["turn_count"] = None

    # Exception distribution
    exc_counts = df["exception_type"].value_counts(dropna=False).to_dict()
    # Rename NaN key to "Success"
    stats["exception_distribution"] = {}
    for k, v in exc_counts.items():
        key = k if isinstance(k, str) and k else "No exception"
        stats["exception_distribution"][key] = int(v)

    # Turn count by exception type
    if len(turns) > 0:
        grouped = df.dropna(subset=["n_episodes"]).groupby(df["exception_type"].fillna("No exception"))["n_episodes"]
        stats["turns_by_exception"] = {}
        for exc_type, group in grouped:
            stats["turns_by_exception"][exc_type] = {
                "mean": float(group.mean()),
                "median": float(group.median()),
                "count": len(group),
            }

    # Reward stats
    rewards = df["reward"].dropna()
    if len(rewards) > 0:
        stats["reward"] = {
            "mean": float(rewards.mean()),
            "success_rate": float((rewards > 0).mean()),
            "count": len(rewards),
        }

        # Turn count for successful vs failed trials
        has_both = df.dropna(subset=["n_episodes", "reward"])
        if len(has_both) > 0:
            successful = has_both[has_both["reward"] > 0]["n_episodes"]
            failed = has_both[has_both["reward"] == 0]["n_episodes"]
            stats["turns_by_outcome"] = {}
            if len(successful) > 0:
                stats["turns_by_outcome"]["success"] = {
                    "mean": float(successful.mean()),
                    "median": float(successful.median()),
                    "count": len(successful),
                }
            if len(failed) > 0:
                stats["turns_by_outcome"]["failure"] = {
                    "mean": float(failed.mean()),
                    "median": float(failed.median()),
                    "count": len(failed),
                }

    return stats


def extract_vllm_metrics(log_content: str) -> list[dict[str, Any]]:
    """
    Extract vLLM stat logger metrics from log content.

    Looks for lines like:
    (AsyncVLLMInferenceEngine pid=287294, ip=10.128.26.194) INFO 02-08 00:56:50 [loggers.py:248]
    Engine 000: Avg prompt throughput: 23.1 tokens/s, Avg generation throughput: 0.0 tokens/s,
    Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.2%, Prefix cache hit rate: 0.0%
    """
    # Strip ANSI codes first
    content = strip_ansi(log_content)

    # Pattern to match vLLM stat logger output
    # Captures: pid, ip, date, time, prompt_throughput, gen_throughput, running, waiting, kv_cache, prefix_cache
    pattern = re.compile(
        r"\(AsyncVLLMInferenceEngine pid=(\d+), ip=([^\)]+)\).*?"
        r"INFO (\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}).*?"
        r"Engine \d+: "
        r"Avg prompt throughput: ([\d.]+) tokens/s, "
        r"Avg generation throughput: ([\d.]+) tokens/s, "
        r"Running: (\d+) reqs, "
        r"Waiting: (\d+) reqs, "
        r"GPU KV cache usage: ([\d.]+)%, "
        r"Prefix cache hit rate: ([\d.]+)%",
        re.MULTILINE,
    )

    metrics_list = []
    for match in pattern.finditer(content):
        pid, ip, date, time_str, prompt_tp, gen_tp, running, waiting, kv_cache, prefix_cache = match.groups()

        metrics_list.append(
            {
                "pid": int(pid),
                "ip": ip,
                "date": date,
                "time": time_str,
                "datetime_str": f"{date} {time_str}",
                "prompt_throughput_tokens_per_sec": float(prompt_tp),
                "generation_throughput_tokens_per_sec": float(gen_tp),
                "running_requests": int(running),
                "waiting_requests": int(waiting),
                "gpu_kv_cache_usage_pct": float(kv_cache),
                "prefix_cache_hit_rate_pct": float(prefix_cache),
            }
        )

    return metrics_list


def aggregate_vllm_metrics(metrics: list[dict[str, Any]], window_seconds: int = 5) -> list[dict[str, Any]]:
    """
    Aggregate vLLM metrics across engines by time window.

    Each inference engine reports independently. This function groups metrics
    by timestamp and aggregates them.
    """
    if not metrics:
        return []

    # Group by datetime_str (already 1-second resolution)
    by_time = defaultdict(list)
    for m in metrics:
        by_time[m["datetime_str"]].append(m)

    aggregated = []
    for time_str, engine_metrics in sorted(by_time.items()):
        n_engines = len(engine_metrics)

        # Aggregate metrics
        agg = {
            "datetime_str": time_str,
            "n_engines_reporting": n_engines,
            "unique_ips": len(set(m["ip"] for m in engine_metrics)),
            # Sum across engines
            "total_prompt_throughput_tokens_per_sec": sum(
                m["prompt_throughput_tokens_per_sec"] for m in engine_metrics
            ),
            "total_generation_throughput_tokens_per_sec": sum(
                m["generation_throughput_tokens_per_sec"] for m in engine_metrics
            ),
            "total_running_requests": sum(m["running_requests"] for m in engine_metrics),
            "total_waiting_requests": sum(m["waiting_requests"] for m in engine_metrics),
            # Average across engines
            "avg_prompt_throughput_per_engine": sum(m["prompt_throughput_tokens_per_sec"] for m in engine_metrics)
            / n_engines,
            "avg_generation_throughput_per_engine": sum(
                m["generation_throughput_tokens_per_sec"] for m in engine_metrics
            )
            / n_engines,
            "avg_running_requests_per_engine": sum(m["running_requests"] for m in engine_metrics) / n_engines,
            "avg_waiting_requests_per_engine": sum(m["waiting_requests"] for m in engine_metrics) / n_engines,
            "avg_gpu_kv_cache_usage_pct": sum(m["gpu_kv_cache_usage_pct"] for m in engine_metrics) / n_engines,
            "avg_prefix_cache_hit_rate_pct": sum(m["prefix_cache_hit_rate_pct"] for m in engine_metrics) / n_engines,
            # Min/Max for understanding variance
            "min_running_requests": min(m["running_requests"] for m in engine_metrics),
            "max_running_requests": max(m["running_requests"] for m in engine_metrics),
            "min_generation_throughput": min(m["generation_throughput_tokens_per_sec"] for m in engine_metrics),
            "max_generation_throughput": max(m["generation_throughput_tokens_per_sec"] for m in engine_metrics),
        }
        aggregated.append(agg)

    return aggregated


def generate_vllm_summary(vllm_metrics: list[dict[str, Any]], aggregated: list[dict[str, Any]]) -> dict[str, Any]:
    """Generate summary statistics for vLLM metrics."""
    if not aggregated:
        return {}

    summary = {
        "total_samples": len(vllm_metrics),
        "aggregated_time_points": len(aggregated),
        "avg_engines_reporting": sum(a["n_engines_reporting"] for a in aggregated) / len(aggregated),
        # Cluster-wide throughput
        "avg_total_prompt_throughput": sum(a["total_prompt_throughput_tokens_per_sec"] for a in aggregated)
        / len(aggregated),
        "avg_total_generation_throughput": sum(a["total_generation_throughput_tokens_per_sec"] for a in aggregated)
        / len(aggregated),
        "max_total_generation_throughput": max(a["total_generation_throughput_tokens_per_sec"] for a in aggregated),
        # Utilization indicators
        "avg_total_running_requests": sum(a["total_running_requests"] for a in aggregated) / len(aggregated),
        "avg_total_waiting_requests": sum(a["total_waiting_requests"] for a in aggregated) / len(aggregated),
        "max_total_running_requests": max(a["total_running_requests"] for a in aggregated),
        "max_total_waiting_requests": max(a["total_waiting_requests"] for a in aggregated),
        # Cache stats
        "avg_kv_cache_usage_pct": sum(a["avg_gpu_kv_cache_usage_pct"] for a in aggregated) / len(aggregated),
        "avg_prefix_cache_hit_rate_pct": sum(a["avg_prefix_cache_hit_rate_pct"] for a in aggregated) / len(aggregated),
        # Per-engine stats
        "avg_running_per_engine": sum(a["avg_running_requests_per_engine"] for a in aggregated) / len(aggregated),
        "avg_generation_throughput_per_engine": sum(a["avg_generation_throughput_per_engine"] for a in aggregated)
        / len(aggregated),
    }

    return summary


def process_log_file(log_path: Path) -> ProcessedLog:
    """Parse current JSON and retained Python-dictionary training logs."""
    with open(log_path, "r", errors="replace") as f:
        content = f.read()

    vllm_metrics = extract_vllm_metrics(content)

    metrics = extract_wandb_json_metrics(content)
    serialization = MetricSerialization.WANDB_JSON
    if not metrics:
        metrics = extract_metrics_blocks(content)
        serialization = MetricSerialization.PYTHON_DICT

    batch_errors = extract_batch_errors(content)
    for metric in metrics:
        step = metric.get("trainer/global_step")
        if step is not None and step in batch_errors:
            metric.update(batch_errors[step])

    # Extract a short name from the filename
    name = log_path.stem

    # If the stem is already short and descriptive (e.g. "900s_225703"), use it directly.
    # Otherwise try to extract version + job ID from long launcher-generated names.
    if len(name) <= 30:
        short_name = name
    else:
        version_match = re.search(r"_(v\d+_[a-z]+)", name)
        job_id_match = re.search(r"_(\d{6})\.", str(log_path))

        if version_match and job_id_match:
            short_name = f"{version_match.group(1)}_{job_id_match.group(1)}"
        elif job_id_match:
            short_name = f"job_{job_id_match.group(1)}"
        else:
            short_name = name[-30:]

    return ProcessedLog(
        name=short_name,
        metrics=metrics,
        vllm_metrics=vllm_metrics,
        serialization=serialization,
    )


def create_summary_statistics(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Create summary statistics for each metric category."""
    summaries = {}

    # Group columns by category
    categories = defaultdict(list)
    for col in df.columns:
        if col in ["log_file", "global_step"]:
            continue
        if "/" in col:
            category = col.split("/")[0]
            categories[category].append(col)
        else:
            categories["other"].append(col)

    # Create summary for each category
    for category, columns in categories.items():
        if not columns:
            continue

        # Select only numeric columns
        numeric_cols = [c for c in columns if df[c].dtype in ["float64", "int64"]]
        if not numeric_cols:
            continue

        summary = df[numeric_cols].agg(["mean", "std", "min", "max", "count"]).T
        summary.columns = ["Mean", "Std", "Min", "Max", "Count"]
        summaries[category] = summary

    return summaries


def resolve_timing_parent(name: str, recorded: set[str]) -> str | None:
    """Return the nearest container of `name` that this run recorded, or None when it has none."""
    parent = TIMING_PARENTS[name]
    while parent is not None and parent not in recorded:
        parent = TIMING_PARENTS[parent]
    return parent


def summarize_timing_spans(df: pd.DataFrame) -> list[TimingSpan]:
    """Summarize recorded timing spans using the declared containment tree.

    Each span's share uses its nearest recorded parent. The `unattributed` row carries step
    time that no direct child covers. Where direct children sum past the step, an `overlap`
    row carries the excess instead.

    Rows run depth-first from `step`, longest sibling first, then the spans that run outside
    `step`, then any column `TIMING_PARENTS` does not declare.
    """
    recorded = {column.removeprefix(TIMING_PREFIX) for column in df.columns if column.startswith(TIMING_PREFIX)}
    if STEP_SPAN not in recorded:
        return []

    def mean_seconds(name: str) -> float:
        return float(df[f"{TIMING_PREFIX}{name}"].mean())

    def summarize(name: str, parent: str | None) -> TimingSpan:
        seconds = df[f"{TIMING_PREFIX}{name}"]
        # A step on which a span did not run is a step on which it took no time, which is how the
        # remainder below counts it too, so a parent's children and its remainder sum to the parent.
        share = (
            float((seconds.fillna(0.0) / df[f"{TIMING_PREFIX}{parent}"]).mean()) if parent is not None else None
        )
        return TimingSpan(name, parent, float(seconds.mean()), share, int(seconds.count()))

    children: dict[str | None, list[str]] = defaultdict(list)
    for name in recorded & set(TIMING_PARENTS):
        if name != STEP_SPAN:
            children[resolve_timing_parent(name, recorded)].append(name)
    for siblings in children.values():
        siblings.sort(key=lambda name: (-mean_seconds(name), name))

    rows = [summarize(STEP_SPAN, None)]

    def append_subtree(parent: str) -> None:
        for child in children[parent]:
            rows.append(summarize(child, parent))
            append_subtree(child)

    append_subtree(STEP_SPAN)

    step_seconds = df[f"{TIMING_PREFIX}{STEP_SPAN}"]
    covered = df[[f"{TIMING_PREFIX}{child}" for child in children[STEP_SPAN]]].fillna(0.0).sum(axis=1)
    unattributed = step_seconds - covered
    mean_unattributed = float(unattributed.mean())
    steps = int(step_seconds.count())
    if mean_unattributed < 0:
        # The children sum past the step, so this trainer ran one of them alongside the step
        # rather than inside it. The excess is neither idle step time nor a share of anything.
        rows.append(TimingSpan(OVERLAP_ROW, STEP_SPAN, -mean_unattributed, None, steps))
    else:
        share = float((unattributed / step_seconds).mean())
        rows.append(TimingSpan(UNATTRIBUTED_ROW, STEP_SPAN, mean_unattributed, share, steps))

    for root in children[None]:
        rows.append(summarize(root, None))
        append_subtree(root)

    undeclared = sorted(recorded - set(TIMING_PARENTS), key=lambda name: (-mean_seconds(name), name))
    for name in undeclared:
        rows.append(summarize(name, None))

    return rows


def _timing_within_label(span: TimingSpan) -> str:
    if span.within is not None:
        return f"`{span.within}`"
    if span.name not in TIMING_PARENTS:
        return "not declared"
    return "—" if span.name == STEP_SPAN else "outside `step`"


def generate_markdown_report(
    all_data: dict[str, list[dict[str, Any]]],
    output_path: Path,
    df: pd.DataFrame,
    vllm_data: dict[str, dict[str, Any]] | None = None,
    trial_stats: dict[str, Any] | None = None,
    selection: CheckpointSelection | None = None,
) -> None:
    """Generate a markdown report with summary statistics."""

    with open(output_path, "w") as f:
        f.write("# SkyRL Training Metrics Analysis\n\n")
        f.write(f"Generated from {len(all_data)} log files\n\n")

        # Overall summary
        f.write("## Overview\n\n")
        f.write(
            "| Log File | Total Steps | Metric Blocks | Final Reward (mean) | Final Reward (max) | Total Time (s) |\n"
        )
        f.write(
            "|----------|-------------|---------------|---------------------|-------------------|----------------|\n"
        )

        for log_name, metrics in all_data.items():
            if not metrics:
                continue

            steps = len(metrics)
            global_steps = [m.get("trainer/global_step", 0) for m in metrics]
            total_steps = max(global_steps) if global_steps else 0
            rewards = [m.get("reward/avg_raw_reward", 0) for m in metrics]
            mean_reward = sum(rewards) / len(rewards) if rewards else 0
            max_reward = max(rewards) if rewards else 0
            total_time = sum(m.get("timing/step", 0) for m in metrics)

            f.write(
                f"| {log_name} | {total_steps} | {steps} | {mean_reward:.4f} | {max_reward:.4f} | {total_time:.1f} |\n"
            )

        f.write("\n")

        # Detailed statistics by category
        summaries = create_summary_statistics(df)

        for category, summary in summaries.items():
            f.write(f"## {category.title()} Metrics\n\n")
            f.write(summary.to_markdown())
            f.write("\n\n")

        # Per-log progression
        f.write("## Training Progression (stability signals)\n\n")

        for log_name, metrics in all_data.items():
            if not metrics:
                continue

            f.write(f"### {log_name}\n\n")

            pass_key = _find_pass_at_key(metrics)
            pass_label = pass_key.split("/")[-1] if pass_key else "pass@k"
            f.write(
                f"| Step | Epoch | Reward | {pass_label} | Entropy | GradNorm | PPOClip | "
                f"PolicyLoss | logRatio_mean | Step Time (s) |\n"
            )
            f.write(
                "|------|-------|--------|--------|---------|----------|---------|"
                "------------|---------------|---------------|\n"
            )

            for m in metrics:
                step = m.get("trainer/global_step", 0)
                epoch = m.get("trainer/epoch", "")
                reward = m.get("reward/avg_raw_reward", float("nan"))
                passk = m.get(pass_key, float("nan")) if pass_key else float("nan")
                entropy = m.get("policy/policy_entropy", float("nan"))
                grad_norm = m.get("policy/raw_grad_norm", float("nan"))
                clip_ratio = m.get("policy/ppo_clip_ratio", float("nan"))
                policy_loss = m.get("policy/policy_loss", float("nan"))
                log_ratio_mean = m.get("policy/log_ratio_abs_mean", float("nan"))
                step_time = m.get("timing/step", float("nan"))
                f.write(
                    f"| {step} | {epoch} | {reward:.4f} | {passk:.4f} | {entropy:.4f} | "
                    f"{grad_norm:.4f} | {clip_ratio:.5f} | {policy_loss:.5f} | "
                    f"{log_ratio_mean:.5f} | {step_time:.1f} |\n"
                )

            f.write("\n")

        if selection is not None:
            _write_best_checkpoint_report(f, selection)

        # Timing breakdown
        f.write("## Timing Analysis\n\n")

        timing_spans = summarize_timing_spans(df)
        if timing_spans:
            f.write("### Average Time Breakdown\n\n")
            f.write(
                "Each span is measured against the span that contains it, so siblings sum toward their\n"
                "parent rather than toward the run. The `unattributed` row is the part of a step that no\n"
                "child span covers, and an `overlap` row in its place means a child ran alongside the\n"
                "step rather than inside it. Spans listed as outside `step` are real wall clock between\n"
                "steps and are not part of one. An average covers only the steps on which its span ran,\n"
                "while a share counts a step the span skipped as zero. A short run averages in one-time\n"
                "costs: the first step of a fully asynchronous run pays a rollout-buffer fill that later\n"
                "steps do not.\n\n"
            )

            f.write("| Span | Within | Avg (s) | Avg % of Within | Steps |\n")
            f.write("|------|--------|---------|-----------------|-------|\n")
            for span in timing_spans:
                share = f"{span.share_of_within * 100:.1f}%" if span.share_of_within is not None else "—"
                f.write(
                    f"| {span.name} | {_timing_within_label(span)} | "
                    f"{span.mean_seconds:.1f} | {share} | {span.steps} |\n"
                )

            f.write("\n")

        # Comparison across logs
        if len(all_data) > 1:
            f.write("## Cross-Log Comparison\n\n")

            comparison_metrics = [
                ("reward/avg_raw_reward", "Avg Reward"),
                ("reward/avg_pass_at_8", "Pass@8"),
                ("timing/step", "Step Time (s)"),
                ("timing/wait_for_generation_buffer", "Gen Wait Time (s)"),
                ("generate/avg_num_tokens", "Avg Tokens"),
                ("async/staleness_mean", "Staleness"),
            ]

            f.write("| Log | " + " | ".join(name for _, name in comparison_metrics) + " |\n")
            f.write("|-----|" + "|".join(["------" for _ in comparison_metrics]) + "|\n")

            for log_name, metrics in all_data.items():
                if not metrics:
                    continue

                row = [log_name]
                for metric_key, _ in comparison_metrics:
                    values = [m.get(metric_key, 0) for m in metrics]
                    mean_val = sum(values) / len(values) if values else 0
                    row.append(f"{mean_val:.4f}")

                f.write("| " + " | ".join(row) + " |\n")

            f.write("\n")

        # vLLM Inference Engine Analysis
        if vllm_data:
            f.write("## vLLM Inference Engine Analysis\n\n")
            f.write("Metrics from vLLM stat loggers (V1LoggingStatLoggerFixed).\n\n")
            f.write("> **Note**: Ray deduplicates similar log messages with `[repeated Nx across cluster]`,\n")
            f.write("> so we typically capture stats from one engine per timestamp. The stats shown are\n")
            f.write("> **per-engine** values. Multiply by num_inference_engines for cluster-wide estimates.\n\n")

            f.write("### Summary by Log (Per-Engine Stats)\n\n")
            f.write(
                "| Log | Avg Running/Engine | Avg Waiting/Engine | Avg Gen Throughput/Engine | Avg KV Cache % | Avg Prefix Hit % |\n"
            )
            f.write(
                "|-----|-------------------|-------------------|--------------------------|----------------|------------------|\n"
            )

            for log_name, data in vllm_data.items():
                summary = data.get("summary", {})
                if not summary:
                    continue

                f.write(f"| {log_name} ")
                f.write(f"| {summary.get('avg_running_per_engine', 0):.1f} ")
                f.write(f"| {summary.get('avg_total_waiting_requests', 0):.1f} ")
                f.write(f"| {summary.get('avg_generation_throughput_per_engine', 0):.1f} tok/s ")
                f.write(f"| {summary.get('avg_kv_cache_usage_pct', 0):.1f}% ")
                f.write(f"| {summary.get('avg_prefix_cache_hit_rate_pct', 0):.1f}% |\n")

            f.write("\n")

            # Utilization analysis
            f.write("### Utilization Analysis (Per-Engine)\n\n")
            f.write("Key indicators of inference engine utilization:\n\n")
            f.write("- **Running requests/engine**: Concurrent requests being processed by each engine\n")
            f.write("- **Waiting requests**: Requests queued (0 = engine not saturated, has spare capacity)\n")
            f.write("- **Generation throughput**: Decode tokens/sec per engine\n")
            f.write("  - 8B model on H100 can do **1000+ tok/s** when saturated\n")
            f.write("  - If seeing <300 tok/s with 0 waiting, engine is **starved for requests**\n\n")

            for log_name, data in vllm_data.items():
                summary = data.get("summary", {})
                if not summary:
                    continue

                f.write(f"#### {log_name}\n\n")

                avg_running = summary.get("avg_running_per_engine", 0)
                max_running = summary.get("max_total_running_requests", 0)
                avg_waiting = summary.get("avg_total_waiting_requests", 0)
                max_waiting = summary.get("max_total_waiting_requests", 0)
                avg_gen_tp = summary.get("avg_generation_throughput_per_engine", 0)
                max_gen_tp = summary.get("max_total_generation_throughput", 0)

                f.write(f"- **Running requests/engine**: avg={avg_running:.1f}, max={max_running}\n")
                f.write(f"- **Waiting requests**: avg={avg_waiting:.1f}, max={max_waiting}\n")
                f.write(f"- **Generation throughput/engine**: avg={avg_gen_tp:.1f} tok/s, max={max_gen_tp:.1f} tok/s\n")
                f.write(f"- **KV cache usage**: avg={summary.get('avg_kv_cache_usage_pct', 0):.1f}%\n")
                f.write(f"- **Prefix cache hit rate**: avg={summary.get('avg_prefix_cache_hit_rate_pct', 0):.1f}%\n")

                # Utilization assessment
                if avg_waiting == 0 and avg_running < 5:
                    f.write(
                        f"- ⚠️ **Underutilized**: Engines starved for requests (0 waiting, avg {avg_running:.1f} running)\n"
                    )
                    f.write("  - Bottleneck is likely upstream (environment execution, not inference)\n")
                elif avg_waiting > 0:
                    f.write("- ✅ **Well-utilized**: Engines saturated (waiting > 0)\n")
                elif avg_gen_tp < 300:
                    f.write(
                        f"- ⚠️ **Low throughput**: {avg_gen_tp:.0f} tok/s << expected 1000+ tok/s for saturated 8B model\n"
                    )
                else:
                    f.write("- ℹ️ **Moderate utilization**\n")

                f.write("\n")

        # Trial-level analysis from result.json
        if trial_stats:
            f.write("## Trial-Level Analysis (from result.json)\n\n")
            f.write(f"Total trials parsed: {trial_stats.get('total_trials', 0)}\n\n")

            phase_summary = trial_stats.get("phase_durations")
            if phase_summary:
                f.write("### Phase Duration Breakdown\n\n")
                f.write(
                    "Harbor times four phases on a trial result. A phase carries its own trial count\n"
                    "because a multi-step trial records `agent_execution` and `verifier` per step rather\n"
                    "than on the trial itself.\n\n"
                )
                if phase_summary.measured_seconds > 0:
                    f.write(
                        "Shares are of the "
                        f"{phase_summary.measured_seconds:.1f} s of wall clock recorded by the "
                        f"{phase_summary.measured_trials} of {phase_summary.total_trials} trials that\n"
                        "timed their own start and finish. `unattributed` is trial time that no phase\n"
                        "covers, and an `overlap` row in its place means the phases sum past the trials\n"
                        "containing them. Medians, means and totals cover every trial that timed the\n"
                        "phase, including trials that never finished and so carry no share.\n\n"
                    )
                else:
                    f.write("No trial recorded its own start and finish, so phase shares are unavailable.\n\n")

                f.write("| Phase | Median (s) | Mean (s) | Total (s) | % of Trial Time | Trials |\n")
                f.write("|-------|------------|----------|-----------|-----------------|--------|\n")
                for phase in phase_summary.phases:
                    median = f"{phase.median_seconds:.1f}" if phase.median_seconds is not None else "—"
                    mean = f"{phase.mean_seconds:.1f}" if phase.mean_seconds is not None else "—"
                    share = f"{phase.share_of_trial * 100:.1f}%" if phase.share_of_trial is not None else "—"
                    trials = str(phase.trials) if phase.trials is not None else "—"
                    f.write(f"| {phase.name} | {median} | {mean} | {phase.total_seconds:.1f} | {share} | {trials} |\n")
                f.write("\n")

            tc = trial_stats.get("turn_count")
            if tc:
                f.write("### Turn Count Statistics\n\n")
                f.write("| Metric | Value |\n")
                f.write("|--------|-------|\n")
                f.write(f"| Mean | {tc['mean']:.1f} |\n")
                f.write(f"| Median | {tc['median']:.1f} |\n")
                f.write(f"| Std | {tc['std']:.1f} |\n")
                f.write(f"| Min | {tc['min']} |\n")
                f.write(f"| Max | {tc['max']} |\n")
                f.write(f"| Count | {tc['count']} |\n\n")

            exc_dist = trial_stats.get("exception_distribution", {})
            if exc_dist:
                f.write("### Exception Distribution\n\n")
                f.write("| Exception Type | Count | % |\n")
                f.write("|---------------|-------|---|\n")
                total = sum(exc_dist.values())
                for exc_type, count in sorted(exc_dist.items(), key=lambda x: x[1], reverse=True):
                    pct = count / total * 100 if total else 0
                    f.write(f"| {exc_type} | {count} | {pct:.1f}% |\n")
                f.write("\n")

            turns_by_exc = trial_stats.get("turns_by_exception", {})
            if turns_by_exc:
                f.write("### Turn Count by Exception Type\n\n")
                f.write("| Exception Type | Mean Turns | Median Turns | Count |\n")
                f.write("|---------------|-----------|-------------|-------|\n")
                for exc_type, stats in sorted(turns_by_exc.items(), key=lambda x: x[1]["mean"], reverse=True):
                    f.write(f"| {exc_type} | {stats['mean']:.1f} | {stats['median']:.1f} | {stats['count']} |\n")
                f.write("\n")

            turns_by_outcome = trial_stats.get("turns_by_outcome", {})
            if turns_by_outcome:
                f.write("### Turn Count by Outcome\n\n")
                f.write("| Outcome | Mean Turns | Median Turns | Count |\n")
                f.write("|---------|-----------|-------------|-------|\n")
                for outcome, stats in turns_by_outcome.items():
                    f.write(f"| {outcome.title()} | {stats['mean']:.1f} | {stats['median']:.1f} | {stats['count']} |\n")
                f.write("\n")

            reward_stats = trial_stats.get("reward")
            if reward_stats:
                f.write("### Reward Summary\n\n")
                f.write(f"- Mean reward: {reward_stats['mean']:.4f}\n")
                f.write(f"- Success rate: {reward_stats['success_rate']:.1%}\n")
                f.write(f"- Trials with reward data: {reward_stats['count']}\n\n")


def _find_pass_at_key(metrics_list: list[dict[str, Any]]) -> str | None:
    """Return the first reward/avg_pass_at_<k> key present (k is n_samples-dependent)."""
    for m in metrics_list:
        for key in m:
            if re.match(r"reward/avg_pass_at_\d+$", key):
                return key
    return None


def generate_reward_plot(all_data: dict[str, list[dict[str, Any]]], output_path: Path) -> None:
    """Plot reward and available stability, TIS, and batch-error metrics."""
    has_logratio = any(any("policy/log_ratio_abs" in k for k in m) for metrics in all_data.values() for m in metrics)
    has_batch_errors = any(any(k.startswith("batch_errors/") for k in m) for metrics in all_data.values() for m in metrics)
    n_panels = 2 + int(has_logratio) + int(has_batch_errors)
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 4 * n_panels), sharex=True)
    ax_reward = axes[0]
    ax_collapse = axes[1]
    ax_lr = axes[2] if has_logratio else None
    ax_errors = axes[2 + int(has_logratio)] if has_batch_errors else None

    for log_name, metrics in all_data.items():
        if not metrics:
            continue
        steps = [m.get("trainer/global_step", i) for i, m in enumerate(metrics)]
        rewards = [m.get("reward/avg_raw_reward", float("nan")) for m in metrics]
        ents = [m.get("policy/policy_entropy", float("nan")) for m in metrics]
        gns = [m.get("policy/raw_grad_norm", float("nan")) for m in metrics]
        if not steps:
            continue
        single = len(steps) == 1
        marker = "o" if single else None
        ms = 8 if single else None

        # Reward panel: EMA solid + raw faint
        raw_series = pd.Series(rewards, index=steps)
        ema_series = raw_series.ewm(span=5).mean()
        color = ax_reward.plot(steps, ema_series.values, label=log_name, linewidth=2, marker=marker, markersize=ms)[
            0
        ].get_color()
        ax_reward.plot(steps, rewards, color=color, alpha=0.2, linewidth=1)

        # Collapse panel: entropy (left axis) + grad_norm (right axis, dashed)
        ax_collapse.plot(
            steps, ents, color=color, linewidth=2, label=f"{log_name} entropy", marker=marker, markersize=ms
        )
        ax_gn = getattr(ax_collapse, "_twin", None)
        if ax_gn is None:
            ax_gn = ax_collapse.twinx()
            ax_collapse._twin = ax_gn
        ax_gn.plot(steps, gns, color=color, linewidth=1.5, linestyle="--", alpha=0.7, label=f"{log_name} grad_norm")

        # TIS / log-ratio panel
        if ax_lr is not None:
            lr_mean = [m.get("policy/log_ratio_abs_mean", float("nan")) for m in metrics]
            lr_p99 = [m.get("policy/log_ratio_abs_p99", float("nan")) for m in metrics]
            lr_max = [m.get("policy/log_ratio_abs_max", float("nan")) for m in metrics]
            ax_lr.plot(
                steps, lr_mean, color=color, linewidth=2, label=f"{log_name} |logr| mean", marker=marker, markersize=ms
            )
            ax_lr.plot(
                steps, lr_p99, color=color, linewidth=1, linestyle=":", alpha=0.8, label=f"{log_name} |logr| p99"
            )
            ax_lr.plot(
                steps, lr_max, color=color, linewidth=1, linestyle="--", alpha=0.5, label=f"{log_name} |logr| max"
            )

        if ax_errors is not None:
            timeouts = [m.get("batch_errors/avg_AgentTimeoutError", 0) for m in metrics]
            context_errors = [m.get("batch_errors/avg_ContextLengthExceededError", 0) for m in metrics]
            ax_errors.plot(steps, timeouts, color=color, linewidth=2, label=f"{log_name} timeouts")
            ax_errors.plot(
                steps,
                context_errors,
                color=color,
                linewidth=1.5,
                linestyle="--",
                label=f"{log_name} context length",
            )

    ax_reward.set_ylabel("Avg Raw Reward")
    ax_reward.set_title("Average Reward vs Training Step (EMA solid, raw faint)")
    ax_reward.legend(loc="best", fontsize="small")
    ax_reward.grid(True, alpha=0.3)

    ax_collapse.set_ylabel("Policy Entropy (solid)")
    ax_collapse.set_title("Entropy & Grad Norm (collapse signals)")
    ax_collapse.grid(True, alpha=0.3)
    if getattr(ax_collapse, "_twin", None) is not None:
        ax_collapse._twin.set_ylabel("Raw Grad Norm (dashed)")
    ax_collapse.legend(loc="upper left", fontsize="small")

    if ax_lr is not None:
        ax_lr.set_ylabel("|log ratio| (TIS)")
        ax_lr.set_title("Token Importance Sampling: |log ratio| mean / p99 / max")
        ax_lr.grid(True, alpha=0.3)
        ax_lr.legend(loc="best", fontsize="small")

    if ax_errors is not None:
        ax_errors.set_ylabel("Errors / Batch")
        ax_errors.set_title("Agent Timeout and Context-Length Errors")
        ax_errors.grid(True, alpha=0.3)
        ax_errors.legend(loc="best", fontsize="small")

    axes[-1].set_xlabel("Training Step")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved training reward/collapse plot to: {output_path}")


def generate_turn_count_plot(trials: list[dict[str, Any]], output_path: Path) -> None:
    """Generate a turn count distribution plot from per-trial result.json data."""
    df = pd.DataFrame(trials)
    turns = df["n_episodes"].dropna()
    if len(turns) == 0:
        print("  No turn count data available for plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: histogram of turn counts
    ax = axes[0]
    ax.hist(turns, bins=min(50, int(turns.max()) + 1), edgecolor="black", alpha=0.7)
    ax.axvline(turns.mean(), color="red", linestyle="--", label=f"Mean: {turns.mean():.1f}")
    ax.axvline(turns.median(), color="orange", linestyle="--", label=f"Median: {turns.median():.1f}")
    ax.set_xlabel("Turn Count (n_episodes)")
    ax.set_ylabel("Number of Trials")
    ax.set_title("Turn Count Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: turn count by exception type (box plot)
    ax2 = axes[1]
    has_turns = df.dropna(subset=["n_episodes"]).copy()
    has_turns["exc"] = has_turns["exception_type"].fillna("No exception")

    # Only show exception types with >= 5 samples
    exc_counts = has_turns["exc"].value_counts()
    top_types = exc_counts[exc_counts >= 5].index.tolist()
    plot_df = has_turns[has_turns["exc"].isin(top_types)]

    if len(plot_df) > 0 and len(top_types) > 1:
        # Sort by median turn count
        medians = plot_df.groupby("exc")["n_episodes"].median().sort_values()
        plot_df["exc"] = pd.Categorical(plot_df["exc"], categories=medians.index, ordered=True)
        plot_df.boxplot(column="n_episodes", by="exc", ax=ax2, vert=True)
        ax2.set_xlabel("Exception Type")
        ax2.set_ylabel("Turn Count")
        ax2.set_title("Turn Count by Exception Type")
        fig.suptitle("")  # Remove auto-generated title from boxplot
        ax2.tick_params(axis="x", rotation=30)
    else:
        ax2.text(0.5, 0.5, "Insufficient data\nfor breakdown", ha="center", va="center", transform=ax2.transAxes)
        ax2.set_title("Turn Count by Exception Type")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved turn count plot to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Parse SkyRL training metrics from console logs")
    parser.add_argument(
        "log_folder",
        type=str,
        help="Path to a FOLDER of logs (globbed by --pattern) or a single log file. "
        "NOT a list of files: pass ONE folder, stage the .out chain links into it first.",
    )
    parser.add_argument("output_folder", type=str, help="Path to output folder for results")
    parser.add_argument("--pattern", type=str, default="*.out", help="Glob pattern for log files (default: *.out)")
    parser.add_argument(
        "--trace_jobs_dir",
        type=str,
        default=None,
        help="Path to trace_jobs directory for per-trial analysis. "
        "Auto-discovered from log_folder if not specified.",
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="RL run dir for best-checkpoint selection: "
        "<run_dir>/exports/global_step_<N>/policy + latest_ckpt_global_step.txt",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=20,
        help="hf_save_interval; checkpoint alignment for best-checkpoint EMA (default: 20)",
    )

    args = parser.parse_args()
    log_folder = Path(args.log_folder)
    output_folder = Path(args.output_folder)

    if not log_folder.exists():
        print(f"Error: Log path does not exist: {log_folder}")
        sys.exit(1)

    # Create output folder
    output_folder.mkdir(parents=True, exist_ok=True)

    # Accept either a directory (glob by --pattern) or a single log file.
    if log_folder.is_file():
        log_files = [log_folder]
    else:
        log_files = list(log_folder.glob(args.pattern))

    if not log_files:
        print(f"No log files matching '{args.pattern}' found in {log_folder}")
        sys.exit(1)

    print(f"Found {len(log_files)} log file(s)")

    # Process each log file
    all_data = {}
    all_rows = []
    all_vllm_data = {}
    all_vllm_rows = []
    for log_path in sorted(log_files):
        print(f"Processing: {log_path.name}")
        processed_log = process_log_file(log_path)

        if not processed_log.metrics and not processed_log.vllm_metrics:
            print(f"  Warning: No metrics found in {log_path.name}")
            continue

        if processed_log.metrics:
            print(f"  Detected training metric serialization: {processed_log.serialization}")
            print(f"  Found {len(processed_log.metrics)} training metric blocks")
            all_data[processed_log.name] = processed_log.metrics

            # Add to combined rows
            for m in processed_log.metrics:
                row = {"log_file": processed_log.name}
                row.update(m)
                all_rows.append(row)

        if processed_log.vllm_metrics:
            print(f"  Found {len(processed_log.vllm_metrics)} vLLM stat logger entries")
            aggregated = aggregate_vllm_metrics(processed_log.vllm_metrics)
            summary = generate_vllm_summary(processed_log.vllm_metrics, aggregated)

            all_vllm_data[processed_log.name] = {
                "raw": processed_log.vllm_metrics,
                "aggregated": aggregated,
                "summary": summary,
            }

            # Add aggregated to combined rows
            for a in aggregated:
                row = {"log_file": processed_log.name}
                row.update(a)
                all_vllm_rows.append(row)

    if not all_rows and not all_vllm_rows:
        print("Error: No metrics found in any log files")
        sys.exit(1)

    trace_jobs_dir = resolve_trace_jobs_dir(log_folder, args.trace_jobs_dir)

    # Timestamp prefix for all output files
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create DataFrame for training metrics
    df = pd.DataFrame(all_rows) if all_rows else pd.DataFrame()

    # Rename trainer/global_step for easier access
    if not df.empty and "trainer/global_step" in df.columns:
        df["global_step"] = df["trainer/global_step"]

    # Save training metrics CSV
    if not df.empty:
        csv_path = output_folder / f"{ts}_metrics_table.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nSaved training metrics table to: {csv_path}")
        canonical_csv = output_folder / "metrics.csv"
        df.to_csv(canonical_csv, index=False)
        print(f"Saved canonical metrics.csv to: {canonical_csv}")

        # Save per-log CSVs
        for log_name, metrics in all_data.items():
            if metrics:
                log_df = pd.DataFrame(metrics)
                log_csv_path = output_folder / f"{ts}_metrics_{log_name}.csv"
                log_df.to_csv(log_csv_path, index=False)
                print(f"Saved per-log training metrics to: {log_csv_path}")

    # Create DataFrame for vLLM metrics
    vllm_df = pd.DataFrame(all_vllm_rows) if all_vllm_rows else pd.DataFrame()

    # Save vLLM metrics CSV
    if not vllm_df.empty:
        vllm_csv_path = output_folder / f"{ts}_vllm_metrics_table.csv"
        vllm_df.to_csv(vllm_csv_path, index=False)
        print(f"\nSaved vLLM metrics table to: {vllm_csv_path}")
        canonical_vllm_csv = output_folder / "vllm_metrics.csv"
        vllm_df.to_csv(canonical_vllm_csv, index=False)
        print(f"Saved canonical vllm_metrics.csv to: {canonical_vllm_csv}")

        # Save per-log vLLM CSVs
        for log_name, data in all_vllm_data.items():
            aggregated = data.get("aggregated", [])
            if aggregated:
                log_vllm_df = pd.DataFrame(aggregated)
                log_vllm_csv_path = output_folder / f"{ts}_vllm_metrics_{log_name}.csv"
                log_vllm_df.to_csv(log_vllm_csv_path, index=False)
                print(f"Saved per-log vLLM metrics to: {log_vllm_csv_path}")

    # Parse per-trial result.json files. Trace availability describes the
    # harness; it is independent of the training metric serialization.
    trial_data = []
    trial_stats_result = None

    if trace_jobs_dir:
        print(f"\nParsing result.json files from: {trace_jobs_dir}")
        trial_data = parse_result_files(trace_jobs_dir)
        if trial_data:
            print(f"  Parsed {len(trial_data)} trial results")
            trial_stats_result = compute_trial_stats(trial_data)

            trial_df = pd.DataFrame(trial_data)
            trial_csv_path = output_folder / f"{ts}_trial_results.csv"
            trial_df.to_csv(trial_csv_path, index=False)
            print(f"Saved trial results to: {trial_csv_path}")
        else:
            print("  No trial results found")
    else:
        print("\nNo trace_jobs directory found; skipping per-trial analysis")

    run_dir = Path(args.run_dir) if args.run_dir else None
    selection = select_best_checkpoint(all_rows, run_dir=run_dir, save_every=args.save_every)
    print_best_checkpoint(selection, args.save_every)

    md_path = output_folder / "report.md"
    generate_markdown_report(
        all_data,
        md_path,
        df,
        vllm_data=all_vllm_data if all_vllm_data else None,
        trial_stats=trial_stats_result,
        selection=selection,
    )
    print(f"Saved markdown report to: {md_path}")

    # Generate reward vs steps plot
    if all_data:
        plot_path = output_folder / "reward_plot.png"
        generate_reward_plot(all_data, plot_path)

    # Generate turn count plot
    if trial_data:
        turn_plot_path = output_folder / f"{ts}_turn_count_distribution.png"
        generate_turn_count_plot(trial_data, turn_plot_path)

    # Print quick summary
    print("\n" + "=" * 60)
    print("QUICK SUMMARY")
    print("=" * 60)

    for log_name, metrics in all_data.items():
        if not metrics:
            continue

        steps = len(metrics)
        global_steps = [m.get("trainer/global_step", 0) for m in metrics]
        total_steps = max(global_steps) if global_steps else 0
        rewards = [m.get("reward/avg_raw_reward", 0) for m in metrics]
        final_reward = rewards[-1] if rewards else 0
        max_reward = max(rewards) if rewards else 0
        avg_step_time = sum(m.get("timing/step", 0) for m in metrics) / steps if steps else 0

        print(f"\n{log_name}:")
        print(f"  Total Steps: {total_steps}  ({steps} metric blocks)")
        print(f"  Final Reward: {final_reward:.4f}")
        print(f"  Max Reward: {max_reward:.4f}")
        print(f"  Avg Step Time: {avg_step_time:.1f}s")

        # Add vLLM summary if available
        if log_name in all_vllm_data:
            summary = all_vllm_data[log_name].get("summary", {})
            if summary:
                print("  vLLM (per-engine):")
                print(f"    Avg Running Reqs: {summary.get('avg_running_per_engine', 0):.1f}")
                print(f"    Avg Waiting Reqs: {summary.get('avg_total_waiting_requests', 0):.1f}")
                print(f"    Avg Gen Throughput: {summary.get('avg_generation_throughput_per_engine', 0):.1f} tok/s")
                print(f"    Avg Prefix Cache Hit: {summary.get('avg_prefix_cache_hit_rate_pct', 0):.1f}%")

    # Print vLLM-only summaries for logs that only have vLLM metrics
    for log_name, data in all_vllm_data.items():
        if log_name in all_data:
            continue  # Already printed above

        summary = data.get("summary", {})
        if summary:
            print(f"\n{log_name} (vLLM metrics only):")
            print("  vLLM (per-engine):")
            print(f"    Avg Running Reqs: {summary.get('avg_running_per_engine', 0):.1f}")
            print(f"    Avg Waiting Reqs: {summary.get('avg_total_waiting_requests', 0):.1f}")
            print(f"    Avg Gen Throughput: {summary.get('avg_generation_throughput_per_engine', 0):.1f} tok/s")
            print(f"    Avg Prefix Cache Hit: {summary.get('avg_prefix_cache_hit_rate_pct', 0):.1f}%")

    # Print trial stats summary
    if trial_stats_result:
        print("\n" + "-" * 40)
        print("TRIAL-LEVEL STATS (from result.json)")
        print("-" * 40)
        print(f"  Total trials: {trial_stats_result.get('total_trials', 0)}")

        tc = trial_stats_result.get("turn_count")
        if tc:
            print(f"  Turn count: mean={tc['mean']:.1f}, median={tc['median']:.1f}, min={tc['min']}, max={tc['max']}")

        reward_info = trial_stats_result.get("reward")
        if reward_info:
            print(f"  Reward: mean={reward_info['mean']:.4f}, success_rate={reward_info['success_rate']:.1%}")

        exc_dist = trial_stats_result.get("exception_distribution", {})
        if exc_dist:
            top3 = sorted(exc_dist.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"  Top exceptions: {', '.join(f'{k}: {v}' for k, v in top3)}")

        turns_by_outcome = trial_stats_result.get("turns_by_outcome", {})
        if turns_by_outcome:
            for outcome, stats in turns_by_outcome.items():
                print(
                    f"  Turns ({outcome}): mean={stats['mean']:.1f}, median={stats['median']:.1f}, n={stats['count']}"
                )


if __name__ == "__main__":
    main()
