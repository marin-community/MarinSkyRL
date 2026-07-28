#!/usr/bin/env python3
"""Analyze a CoreWeave Iris RL job from its canonical local evidence bundle.

The analyzer reads the same bundle produced by ``watch_coreweave_rl.py``. By
default it first makes a bounded attempt to refresh the durable finelog; an
unavailable terminal pod or transient controller failure never prevents it from
reporting the evidence already present locally. Use ``--local-only`` to make no
remote calls.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.iris.iris_ops import (  # noqa: E402
    DEFAULT_BUNDLE_ROOT,
    IRIS_BIN,
    job_bundle,
    load_bundle_manifest,
    write_bundle_manifest,
)
from scripts.iris.coreweave_ops import CLUSTERS as COREWEAVE_CLUSTERS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_id", help="Iris job id, e.g. /benjaminfeuer/my-rl-run")
    parser.add_argument("--cluster", default="cw-us-east-02a")
    parser.add_argument("--bundle-root", type=Path, default=DEFAULT_BUNDLE_ROOT)
    parser.add_argument("--output", type=Path, help="Markdown report path (defaults inside the bundle).")
    parser.add_argument("--local-only", action="store_true", help="Use the existing bundle without contacting Iris.")
    return parser.parse_args()


def refresh_finelog(bundle_directory: Path, job_id: str, cluster: str) -> str | None:
    """Refresh controller-retained Finelog into the shared bundle, if available."""
    environment = os.environ.copy()
    if cluster in COREWEAVE_CLUSTERS:
        environment["KUBECONFIG"] = str(COREWEAVE_CLUSTERS[cluster].kubeconfig)
    try:
        result = subprocess.run(
            [IRIS_BIN, f"--cluster={cluster}", "job", "logs", job_id, "--max-lines", "10000000", "--no-tail"],
            capture_output=True,
            text=True,
            timeout=900,
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"Finelog refresh unavailable: {error}"
    (bundle_directory / "finelog.refresh.stderr").write_text(result.stderr)
    if result.returncode:
        detail = (result.stderr or result.stdout).strip().replace("\n", " ")
        return f"Finelog refresh failed: {detail[-240:]}"
    (bundle_directory / "finelog.log").write_text(result.stdout)
    return None


def count_trace_results(bundle_directory: Path) -> tuple[int, int]:
    trace_root = bundle_directory / "trace_jobs"
    if not trace_root.exists():
        return 0, 0
    return (
        sum(1 for path in trace_root.rglob("result.json") if path.is_file()),
        sum(1 for path in trace_root.rglob("exception.txt") if path.is_file()),
    )


def finelog_tail(bundle_directory: Path) -> str:
    path = bundle_directory / "finelog.log"
    if not path.exists():
        return "No local Finelog was available."
    lines = path.read_text(errors="replace").splitlines()
    return "\n".join(lines[-30:]) or "Finelog is empty."


def render_report(manifest: dict[str, Any], bundle_directory: Path, refresh_error: str | None) -> str:
    trace_results, exceptions = count_trace_results(bundle_directory)
    artifacts = manifest.get("artifacts", {})
    lines = [
        f"# CoreWeave RL job analysis — {manifest.get('job_id', 'unknown')}",
        "",
        f"- Cluster: `{manifest.get('cluster', 'unknown')}`",
        f"- Bundle: `{bundle_directory}`",
        f"- Last bundle sync: {manifest.get('synced_at', 'unknown')}",
        f"- Trace result files: {trace_results}",
        f"- Trace exception files: {exceptions}",
        f"- Finelog: {artifacts.get('finelog', 'not recorded')}",
        f"- Pod logs: {artifacts.get('pod_logs', 'not recorded')}",
        f"- Ray/vLLM logs: {artifacts.get('ray_logs', 'not recorded')}",
    ]
    if refresh_error:
        lines.extend(["", "## Refresh note", "", refresh_error])
    lines.extend(["", "## Finelog tail", "", "```text", finelog_tail(bundle_directory), "```"])
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    bundle = job_bundle(args.bundle_root, args.cluster, args.job_id)
    bundle.directory.mkdir(parents=True, exist_ok=True)
    refresh_error = None if args.local_only else refresh_finelog(bundle.directory, args.job_id, args.cluster)
    write_bundle_manifest(
        bundle,
        {
            "kind": "rl",
            "last_analysis_refresh_at": datetime.now(UTC).isoformat(),
            "last_analysis_refresh_error": refresh_error,
        },
    )
    manifest = load_bundle_manifest(bundle)
    output = args.output or bundle.directory / "reports" / "analysis.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_report(manifest, bundle.directory, refresh_error))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
