"""Decide whether a nightly SkyRL training run is healthy, from its log alone.

The trainer mirrors every tracker payload to stdout as a ``WANDB_MIRROR`` line (see
``RayPPOTrainer._log_metrics_stdout``), so a run's metrics survive in its log without
wandb, a checkpoint, or cluster access. This reads that log and checks the run against a
spec: enough training steps completed, the metrics that must exist are there and finite,
the ones with a meaningful range are inside it, and the run finished inside its
wall-clock budget.

The gate is deliberately coarse. It answers "does the user-facing training path still
work end to end", not "is the model any good" -- a two-step run of a 0.6B policy has no
signal about quality, and a reward floor above zero would just be flaky.

    python -m ci.marin_nightly.gate --log run.log \
        --spec ci/marin_nightly/specs/gsm8k-qwen3-0.6b.json --wall-clock-seconds 900
"""

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# `WANDB_MIRROR kind=train step=2 metrics={"policy/policy_loss": 0.1, ...}`, embedded in a
# loguru line, so the prefix is matched loosely and the JSON object runs to end of line.
METRIC_LINE = re.compile(r"WANDB_MIRROR kind=(?P<kind>\w+) step=(?P<step>\d+) metrics=(?P<metrics>\{.*\})\s*$")

TRAIN = "train"


@dataclass(frozen=True)
class StepMetrics:
    """The metrics the trainer logged for one step."""

    kind: str
    step: int
    values: dict[str, object]


@dataclass(frozen=True)
class MetricBound:
    """The closed range a metric must land in."""

    minimum: float
    maximum: float


@dataclass(frozen=True)
class GateSpec:
    """What a healthy run looks like. See the shipped specs for the recorded values."""

    min_train_steps: int
    finite_metrics: tuple[str, ...]
    bounds: dict[str, MetricBound]
    max_wall_clock_seconds: float


def load_spec(path: Path) -> GateSpec:
    raw = json.loads(path.read_text())
    return GateSpec(
        min_train_steps=raw["min_train_steps"],
        finite_metrics=tuple(raw["finite_metrics"]),
        bounds={k: MetricBound(v["minimum"], v["maximum"]) for k, v in raw["bounds"].items()},
        max_wall_clock_seconds=raw["max_wall_clock_seconds"],
    )


def parse_metrics(log_text: str) -> list[StepMetrics]:
    """Pull every WANDB_MIRROR payload out of a run log, in the order they were logged."""
    steps = []
    for line in log_text.splitlines():
        match = METRIC_LINE.search(line)
        if match is None:
            continue
        steps.append(
            StepMetrics(
                kind=match["kind"],
                step=int(match["step"]),
                values=json.loads(match["metrics"]),
            )
        )
    return steps


def _is_finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def check_run(steps: list[StepMetrics], spec: GateSpec, wall_clock_seconds: float) -> list[str]:
    """Check a parsed run against the spec. Returns one message per violation, empty if healthy."""
    failures = []

    if wall_clock_seconds > spec.max_wall_clock_seconds:
        failures.append(f"run took {wall_clock_seconds:.0f}s, over the {spec.max_wall_clock_seconds:.0f}s budget")

    train_steps = [s for s in steps if s.kind == TRAIN]
    if len(train_steps) < spec.min_train_steps:
        failures.append(f"logged {len(train_steps)} training steps, expected at least {spec.min_train_steps}")
    if not train_steps:
        return failures

    # The last step is the one that has to be healthy: an early step can look fine while the
    # run degrades into NaN later.
    final = train_steps[-1]
    for name in spec.finite_metrics:
        if name not in final.values:
            failures.append(f"step {final.step} did not log {name}")
        elif not _is_finite(final.values[name]):
            failures.append(f"step {final.step} logged {name}={final.values[name]!r}, which is not finite")

    for name, bound in spec.bounds.items():
        value = final.values.get(name)
        if not _is_finite(value):
            continue  # already reported by the finiteness check if it was required
        if not bound.minimum <= value <= bound.maximum:
            failures.append(f"step {final.step} logged {name}={value}, outside [{bound.minimum}, {bound.maximum}]")

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True, help="training run log to read")
    parser.add_argument("--spec", type=Path, required=True, help="gate spec to check against")
    parser.add_argument(
        "--wall-clock-seconds",
        type=float,
        required=True,
        help="how long the run took, measured by the caller",
    )
    args = parser.parse_args()

    spec = load_spec(args.spec)
    steps = parse_metrics(args.log.read_text())
    failures = check_run(steps, spec, args.wall_clock_seconds)

    train_steps = [s for s in steps if s.kind == TRAIN]
    print(f"parsed {len(train_steps)} training steps from {args.log} in {args.wall_clock_seconds:.0f}s")
    if train_steps:
        print(f"final step {train_steps[-1].step}: {json.dumps(train_steps[-1].values, sort_keys=True)}")

    if failures:
        print(f"\nFAILED against {args.spec}:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print(f"\nOK against {args.spec}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
