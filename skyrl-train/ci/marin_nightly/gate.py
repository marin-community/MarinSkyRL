"""Check a SkyRL training log against its run spec."""

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# `WANDB_MIRROR kind=train step=2 metrics={"policy/policy_loss": 0.1, ...}`, embedded in a
# loguru line, so the prefix is matched loosely and the JSON object runs to end of line.
METRIC_LINE = re.compile(r"WANDB_MIRROR kind=(?P<kind>\w+) step=(?P<step>\d+) metrics=(?P<metrics>\{.*\})\s*$")

# The trainer's loguru sink colorizes even when piped, so the payload arrives wrapped in SGR
# escapes (`\x1b[32m...WANDB_MIRROR...}\x1b[0m`). The trailing reset defeats the `\}\s*$` anchor,
# which silently parses zero steps out of a perfectly healthy run -- strip the escapes first.
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")

TRAIN = "train"
EVAL = "eval"


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
class MetricOccurrence:
    """Require at least ``minimum_count`` observations satisfying a threshold."""

    minimum_count: int
    comparison: Literal["above", "below", "at_least"]
    threshold: float

    def __post_init__(self) -> None:
        if self.comparison not in ("above", "below", "at_least") or self.minimum_count < 1:
            raise ValueError("occurrence requires above, below or at_least and a positive minimum_count")


@dataclass(frozen=True)
class SeriesTrend:
    """Compare the first and last windows of one metric series."""

    window: int
    min_improvement: float

    def __post_init__(self) -> None:
        if self.window < 1:
            raise ValueError("trend window must be positive")


@dataclass(frozen=True)
class MetricSeries:
    """Evidence required from one metric in one tracker payload stream."""

    kind: Literal["train", "eval"]
    metric: str
    required: bool
    min_observations: int
    finite_every_step: bool = False
    bounds: MetricBound | None = None
    trend: SeriesTrend | None = None
    occurrence: MetricOccurrence | None = None
    at_step: int | Literal["first", "last"] | None = None
    through_step: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in (TRAIN, EVAL) or self.min_observations < 1:
            raise ValueError("metric series requires train or eval and a positive min_observations")
        if self.at_step is not None and self.at_step not in ("first", "last"):
            if isinstance(self.at_step, bool) or not isinstance(self.at_step, int) or self.at_step < 0:
                raise ValueError("at_step requires a non-negative step, first or last")


@dataclass(frozen=True)
class LogPatternBound:
    """The inclusive occurrence range for one regex in the complete job log."""

    pattern: str
    minimum: int
    maximum: int | None = None


@dataclass(frozen=True)
class GateSpec:
    """What a healthy run looks like. See the shipped specs for the recorded values."""

    min_train_steps: int
    finite_metrics: tuple[str, ...]
    bounds: dict[str, MetricBound]
    max_wall_clock_seconds: float
    required_log_patterns: dict[str, LogPatternBound] | None = None
    metric_series: tuple[MetricSeries, ...] = ()


def load_spec(path: Path) -> GateSpec:
    raw = json.loads(path.read_text())
    return GateSpec(
        min_train_steps=raw["min_train_steps"],
        finite_metrics=tuple(raw.get("finite_metrics", ())),
        bounds={k: MetricBound(v["minimum"], v["maximum"]) for k, v in raw.get("bounds", {}).items()},
        max_wall_clock_seconds=raw["max_wall_clock_seconds"],
        required_log_patterns={
            name: LogPatternBound(value["pattern"], value["minimum"], value.get("maximum"))
            for name, value in raw.get("required_log_patterns", {}).items()
        },
        metric_series=tuple(
            MetricSeries(
                kind=value["kind"],
                metric=value["metric"],
                required=value["required"],
                min_observations=value["min_observations"],
                finite_every_step=value.get("finite_every_step", False),
                bounds=MetricBound(**value["bounds"]) if "bounds" in value else None,
                trend=SeriesTrend(**value["trend"]) if "trend" in value else None,
                occurrence=MetricOccurrence(**value["occurrence"]) if "occurrence" in value else None,
                at_step=value.get("at_step"),
                through_step=value.get("through_step"),
            )
            for value in raw.get("metric_series", ())
        ),
    )


def parse_metrics(log_text: str) -> list[StepMetrics]:
    """Pull every WANDB_MIRROR payload out of a run log, in the order they were logged."""
    steps = []
    for line in ANSI_ESCAPE.sub("", log_text).splitlines():
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


def _distinct_steps(steps: list[StepMetrics]) -> tuple[list[StepMetrics], list[str]]:
    by_key: dict[tuple[str, int], StepMetrics] = {}
    failures = []
    for step in steps:
        if step.kind not in (TRAIN, EVAL):
            continue
        key = (step.kind, step.step)
        prior = by_key.get(key)
        if prior is not None and json.dumps(prior.values, sort_keys=True) != json.dumps(step.values, sort_keys=True):
            failures.append(f"conflicting {step.kind} payloads at step {step.step}")
        by_key[key] = step
    return sorted(by_key.values(), key=lambda item: (item.kind, item.step)), failures


def check_run(steps: list[StepMetrics], spec: GateSpec, wall_clock_seconds: float) -> list[str]:
    """Check a parsed run against the spec. Returns one message per violation, empty if healthy."""
    distinct_steps, failures = _distinct_steps(steps)

    if wall_clock_seconds > spec.max_wall_clock_seconds:
        failures.append(f"run took {wall_clock_seconds:.0f}s, over the {spec.max_wall_clock_seconds:.0f}s budget")

    train_steps = [s for s in distinct_steps if s.kind == TRAIN]
    if len(train_steps) < spec.min_train_steps:
        failures.append(f"logged {len(train_steps)} training steps, expected at least {spec.min_train_steps}")
    if train_steps:
        final = train_steps[-1]
        for name in spec.finite_metrics:
            if name not in final.values:
                failures.append(f"step {final.step} did not log {name}")
            elif not _is_finite(final.values[name]):
                failures.append(f"step {final.step} logged {name}={final.values[name]!r}, which is not finite")

        for name, bound in spec.bounds.items():
            value = final.values.get(name)
            if not _is_finite(value):
                continue
            if not bound.minimum <= value <= bound.maximum:
                failures.append(f"step {final.step} logged {name}={value}, outside [{bound.minimum}, {bound.maximum}]")

    for requirement in spec.metric_series:
        failures.extend(_metric_series_failures(distinct_steps, requirement))

    return failures


def check_log_patterns(log_text: str, spec: GateSpec) -> list[str]:
    """Check named, production-observable events that do not belong to trainer metrics."""
    failures = []
    clean_log = ANSI_ESCAPE.sub("", log_text)
    for name, bound in (spec.required_log_patterns or {}).items():
        count = len(re.findall(bound.pattern, clean_log))
        if count < bound.minimum:
            failures.append(f"log pattern {name!r} occurred {count} times, expected at least {bound.minimum}")
        if bound.maximum is not None and count > bound.maximum:
            failures.append(f"log pattern {name!r} occurred {count} times, expected at most {bound.maximum}")
    return failures


def _metric_series_failures(steps: list[StepMetrics], requirement: MetricSeries) -> list[str]:
    kind_steps = [step for step in steps if step.kind == requirement.kind]
    if isinstance(requirement.at_step, int):
        kind_steps = [step for step in kind_steps if step.step == requirement.at_step]
    elif requirement.at_step == "first":
        kind_steps = kind_steps[:1]
    elif requirement.at_step == "last":
        kind_steps = kind_steps[-1:]
    if requirement.through_step is not None:
        kind_steps = [step for step in kind_steps if step.step <= requirement.through_step]
    observed = [step for step in kind_steps if requirement.metric in step.values]
    if not observed and not requirement.required:
        return []

    failures = []
    values: list[float] = []
    for step in kind_steps:
        value = step.values.get(requirement.metric)
        if requirement.metric not in step.values:
            if requirement.finite_every_step:
                failures.append(f"{requirement.kind} step {step.step} did not log {requirement.metric}")
            continue
        if not _is_finite(value):
            failures.append(f"{requirement.kind} step {step.step} logged nonfinite {requirement.metric}={value!r}")
            continue
        values.append(value)
        if requirement.bounds is not None and not requirement.bounds.minimum <= value <= requirement.bounds.maximum:
            failures.append(
                f"{requirement.kind} step {step.step} logged {requirement.metric}={value}, "
                f"outside [{requirement.bounds.minimum}, {requirement.bounds.maximum}]"
            )

    if len(values) < requirement.min_observations:
        failures.append(
            f"{requirement.kind} {requirement.metric} has {len(values)} finite observations, "
            f"expected at least {requirement.min_observations}"
        )

    if requirement.trend is not None:
        trend = requirement.trend
        if len(values) < 2 * trend.window:
            failures.append(
                f"{requirement.kind} {requirement.metric} has {len(values)} finite observations, "
                f"expected at least {2 * trend.window} for its trend"
            )
        else:
            early = math.fsum(value / trend.window for value in values[: trend.window])
            late = math.fsum(value / trend.window for value in values[-trend.window :])
            improvement = late - early
            if not math.isfinite(early) or not math.isfinite(late) or improvement < trend.min_improvement:
                failures.append(
                    f"{requirement.kind} {requirement.metric} rose by {improvement:+.4f}, "
                    f"expected at least {trend.min_improvement:+.4f}"
                )

    if requirement.occurrence is not None:
        occurrence = requirement.occurrence
        count = sum(
            (
                value >= occurrence.threshold
                if occurrence.comparison == "at_least"
                else value > occurrence.threshold
                if occurrence.comparison == "above"
                else value < occurrence.threshold
            )
            for value in values
        )
        if count < occurrence.minimum_count:
            failures.append(
                f"{requirement.kind} {requirement.metric} has {count} observations "
                f"{occurrence.comparison} {occurrence.threshold}, "
                f"expected at least {occurrence.minimum_count}"
            )

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
    log_text = args.log.read_text()
    steps = parse_metrics(log_text)
    failures = check_run(steps, spec, args.wall_clock_seconds)
    failures.extend(check_log_patterns(log_text, spec))

    distinct_steps, _ = _distinct_steps(steps)
    train_steps = [s for s in distinct_steps if s.kind == TRAIN]
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
