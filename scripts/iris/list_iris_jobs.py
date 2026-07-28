#!/usr/bin/env python3
"""List one Iris user's recent jobs across selected clusters (oldest first)."""

from __future__ import annotations

import argparse
import csv
import getpass
import io
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from scripts.iris.iris_ops import (
    STATE_NAMES,
    box_table,
    filter_records,
    format_duration,
    parse_regex_filters,
    run_iris_command,
)

USER_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
STATE_LABELS = {"killed": "terminated", "worker_failed": "worker failed"}
COREWEAVE_KUBECONFIGS = {
    "cw-rno2a": Path("/Users/benjaminfeuer/.kube/coreweave-iris"),
    "cw-us-east-02a": Path("/Users/benjaminfeuer/.kube/coreweave-iris"),
}
DEFAULT_CLUSTERS = ("cw-rno2a", "cw-us-east-02a", "marin")


def command_environment(cluster: str) -> dict[str, str] | None:
    """Return the local kubeconfig override required for a CoreWeave cluster."""
    kubeconfig = COREWEAVE_KUBECONFIGS.get(cluster)
    if kubeconfig is None:
        return None
    environment = os.environ.copy()
    environment["KUBECONFIG"] = str(kubeconfig)
    return environment


def classify_job_type(job_id: str) -> str:
    """Return a conservative workload hint based only on the submitted job name."""
    name = job_id.lower()
    if any(hint in name for hint in ("mirror", "gcs2s3", "hf2s3")):
        return "mirror"
    if any(hint in name for hint in ("skyrl", "grpo", "rl-", "-rl", "terminus")):
        return "RL"
    if any(hint in name for hint in ("eval", "tb2", "terminal-bench", "swebench")):
        return "eval"
    if any(hint in name for hint in ("tracegen", "datagen", "harbor", "pilot")):
        return "datagen"
    if any(hint in name for hint in ("sft", "finetune", "finetuning")):
        return "SFT"
    if any(hint in name for hint in ("serve", "vllm", "endpoint")):
        return "serve"
    return "other"


def state_label(value: str | int | None) -> str:
    """Normalize a numeric controller state into a readable lifecycle state."""
    try:
        state = STATE_NAMES.get(int(value), f"state {value}")
    except (TypeError, ValueError):
        state = str(value or "unknown").lower()
    return STATE_LABELS.get(state, state.replace("_", " "))


def parse_jobs_csv(output: str) -> list[dict[str, str]]:
    """Parse Iris CSV while discarding the CLI's informational preamble."""
    lines = output.splitlines()
    header_index = next((i for i, line in enumerate(lines) if line.startswith("job_id,")), None)
    if header_index is None:
        raise ValueError("Iris query returned no CSV job_id header")
    return list(csv.DictReader(io.StringIO("\n".join(lines[header_index:]))))


def query_jobs(*, user: str, hours: float, cluster: str, now_ms: int | None = None) -> list[dict[str, str]]:
    """Return one user's submitted jobs since the bounded epoch cutoff."""
    if not USER_RE.fullmatch(user):
        raise ValueError(f"Invalid Iris user {user!r}")
    if hours < 0:
        raise ValueError("--hours must be non-negative")
    cutoff_clause = ""
    if hours:
        cutoff_ms = (now_ms if now_ms is not None else int(time.time() * 1000)) - int(hours * 3_600_000)
        cutoff_clause = f" AND submitted_at_ms >= {cutoff_ms}"
    sql = (
        "SELECT job_id,state,submitted_at_ms,started_at_ms,finished_at_ms,error,exit_code "
        "FROM jobs "
        f"WHERE job_id LIKE '/{user}/%' {cutoff_clause} "
        "ORDER BY submitted_at_ms ASC"
    )
    result = run_iris_command(
        ["query", sql, "-f", "csv"],
        cluster=cluster,
        environment=command_environment(cluster),
    )
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"Iris jobs query failed: {detail[-600:]}")
    return parse_jobs_csv(result.stdout)


def _timestamp(ms: str | None) -> str:
    try:
        return datetime.fromtimestamp(int(ms or "") / 1000, tz=timezone.utc).strftime("%m-%d %H:%M")
    except (TypeError, ValueError, OSError):
        return "—"


def _milliseconds(value: str | None) -> int | None:
    try:
        return int(value or "")
    except (TypeError, ValueError):
        return None


def job_filter_values(row: dict[str, str], *, now_ms: int | None = None) -> dict[str, str]:
    """Return the visible fields available to ``--filter`` for one job."""
    started_at_ms = _milliseconds(row.get("started_at_ms"))
    submitted_at_ms = _milliseconds(row.get("submitted_at_ms"))
    finished_at_ms = _milliseconds(row.get("finished_at_ms"))
    job_id = row.get("job_id", "")
    return {
        "cluster": row.get("cluster", ""),
        "submitted": _timestamp(row.get("submitted_at_ms")),
        "job": job_id,
        "name": job_id.rsplit("/", 1)[-1],
        "type": classify_job_type(job_id),
        "state": state_label(row.get("state")),
        "duration": format_duration(
            started_at_ms if started_at_ms is not None else submitted_at_ms, finished_at_ms, now_ms=now_ms
        ),
        "exit": row.get("exit_code") or "—",
        "error": (row.get("error") or "").replace("\n", " "),
    }


def _short(value: str, width: int) -> str:
    return value if len(value) <= width else f"{value[: width - 1]}…"


def render_table(rows: list[dict[str, str]], *, now_ms: int | None = None) -> str:
    """Render controller rows as a stable oldest-first Unicode table."""
    headers = ("Cluster", "Submitted UTC", "Job", "Type (hint)", "State", "Duration", "Exit", "Error")
    values = [
        (
            fields["cluster"] or "—",
            fields["submitted"],
            _short(fields["job"], 54),
            fields["type"],
            fields["state"],
            fields["duration"],
            fields["exit"],
            _short(fields["error"], 54) or "—",
        )
        for row in rows
        for fields in [job_filter_values(row, now_ms=now_ms)]
    ]
    return box_table(headers, values)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", default=getpass.getuser(), help="Iris username (default: current OS user).")
    parser.add_argument(
        "--hours", type=float, default=24.0, help="Submitted-job window in hours; 0 means all history (default: 24)."
    )
    parser.add_argument(
        "--cluster",
        action="append",
        help=f"Iris cluster to query; repeat to select several (default: {', '.join(DEFAULT_CLUSTERS)}).",
    )
    parser.add_argument(
        "--filter",
        action="append",
        default=[],
        metavar="KEY=REGEX",
        help=(
            "Keep jobs matching every case-insensitive regex filter. Available keys: "
            "cluster, submitted, job, name, type, state, duration, exit, error. "
            "Repeat the flag (for example: --filter 'state=running' --filter 'name=glm52')."
        ),
    )
    args = parser.parse_args(argv)
    clusters = args.cluster or DEFAULT_CLUSTERS
    try:
        filters = parse_regex_filters(
            args.filter,
            {"cluster", "submitted", "job", "name", "type", "state", "duration", "exit", "error"},
        )
    except ValueError as error:
        parser.error(str(error))
    rows: list[dict[str, str]] = []
    errors: list[str] = []
    for cluster in clusters:
        try:
            rows.extend(
                {**row, "cluster": cluster} for row in query_jobs(user=args.user, hours=args.hours, cluster=cluster)
            )
        except (RuntimeError, ValueError) as error:
            errors.append(f"{cluster}: {error}")
    if errors and not rows:
        parser.error("; ".join(errors))

    rows.sort(key=lambda row: _milliseconds(row.get("submitted_at_ms")) or 0)
    now_ms = int(time.time() * 1000)
    rows = filter_records(rows, filters, lambda row: job_filter_values(row, now_ms=now_ms))
    filter_suffix = f" filters={','.join(args.filter)}" if args.filter else ""
    window = "all" if args.hours == 0 else f"{args.hours:g}h"
    print(f"# Iris jobs — user={args.user} clusters={','.join(clusters)} last={window}{filter_suffix}; oldest first")
    print(render_table(rows, now_ms=now_ms))
    if errors:
        print("\n## Cluster query errors")
        for error in errors:
            print(f"- {error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
