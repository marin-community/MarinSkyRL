"""Command-line entrypoint for local SkyRL behavior analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import analyze_local_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze local SkyRL rollout and evaluation artifacts.")
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--post-dir", type=Path)
    parser.add_argument("--training-log-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bin-hours", type=float, default=4.0)
    args = parser.parse_args()
    analyze_local_run(
        rollout_dir=args.rollout_dir,
        baseline_dir=args.baseline_dir,
        post_dir=args.post_dir,
        training_log_dir=args.training_log_dir,
        output_dir=args.output_dir,
        bin_hours=args.bin_hours,
    )


if __name__ == "__main__":
    main()
