"""Gate the one-off Nemotron Ultra Iris run from its mirrored trainer metrics."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

from ci.marin_nightly.gate import ANSI_ESCAPE, TRAIN, parse_metrics
from infra.rl_data.sources import NEMOTRON_ULTRA_RLVR1_AGENTS, NEMOTRON_ULTRA_RLVR2_AGENTS

SAMPLE_LINE = re.compile(r"NEMOTRON_ULTRA_SAMPLE (?P<manifest>\{.*\})\s*$")


def expected_coverage() -> set[str]:
    return {
        f"nemotron_ultra/coverage/{blend}/{agent}"
        for blend, agents in (
            ("rlvr1", NEMOTRON_ULTRA_RLVR1_AGENTS),
            ("rlvr2", NEMOTRON_ULTRA_RLVR2_AGENTS),
        )
        for agent in agents
    }


def check_log(log_text: str) -> list[str]:
    failures: list[str] = []
    clean_log = ANSI_ESCAPE.sub("", log_text)
    manifests = [json.loads(match["manifest"]) for line in clean_log.splitlines() if (match := SAMPLE_LINE.search(line))]
    if len(manifests) != 1:
        failures.append(f"found {len(manifests)} sample manifests, expected exactly one")
    elif manifests[0].get("rows") != len(expected_coverage()):
        failures.append(f"sample manifest reports {manifests[0].get('rows')!r} rows, expected {len(expected_coverage())}")

    train_steps = [step for step in parse_metrics(log_text) if step.kind == TRAIN]
    if len(train_steps) != 1:
        failures.append(f"logged {len(train_steps)} training steps, expected exactly one")
    if not train_steps:
        return failures
    metrics = train_steps[-1].values
    for name in ("policy/policy_loss", "policy/final_loss"):
        value = metrics.get(name)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            failures.append(f"training step did not log finite {name}")
    for key in sorted(expected_coverage()):
        if metrics.get(key) != 2:
            failures.append(f"{key}={metrics.get(key)!r}, expected 2 completed rollouts")
    failed = metrics.get("generate/num_failed_trajectories")
    if failed != 0:
        failures.append(f"generate/num_failed_trajectories={failed!r}, expected 0")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    failures = check_log(args.log.read_text())
    if failures:
        print("Nemotron Ultra acceptance failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(f"Nemotron Ultra acceptance passed all {len(expected_coverage())} phase/generator pairs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
