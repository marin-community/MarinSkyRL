"""Tests for the nightly end-to-end gate (ci/marin_nightly/gate.py).

Run with: uv run --isolated --group dev --extra cpu pytest tests/cpu/test_marin_nightly_gate.py
"""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from ci.marin_nightly.gate import (
    GateSpec,
    LogPatternBound,
    MetricBound,
    MetricSeries,
    SeriesTrend,
    StepMetrics,
    check_log_patterns,
    check_run,
    load_spec,
    parse_metrics,
)

SHIPPED_SPEC = Path(__file__).parents[2] / "ci" / "marin_nightly" / "specs" / "gsm8k-qwen3-0.6b-megatron.json"
FSDP_SPEC = Path(__file__).parents[2] / "ci" / "marin_nightly" / "specs" / "gsm8k-qwen3-0.6b-fsdp2.json"
OPENCODE_SPEC = Path(__file__).parents[2] / "ci" / "marin_nightly" / "specs" / "opencode-qwen3-8b.json"

# What the trainer actually writes: loguru decorates the line, so the payload is embedded
# rather than anchored at the start. Keep this in the shape the trainer emits it.
LOGURU_PREFIX = "2026-07-14 09:00:00.000 | INFO     | skyrl_train.trainer:_log_metrics_stdout:1951 - "


def mirror_line(step: int, kind: str = "train", drop: tuple[str, ...] = (), **metrics) -> str:
    payload = {
        "policy/policy_loss": 0.42,
        "policy/final_loss": 0.42,
        "policy/policy_entropy": 1.1,
        "reward/avg_raw_reward": 0.25,
        **metrics,
    }
    for name in drop:
        del payload[name]
    return f"{LOGURU_PREFIX}WANDB_MIRROR kind={kind} step={step} metrics={json.dumps(payload, sort_keys=True)}"


def healthy_log(steps: int = 2) -> str:
    """A log for a run that trained cleanly. Reward climbs step over step the way a healthy
    GRPO run's does, so it also satisfies the reward-trend gate, not just the structural checks."""
    lines = ["Ray runtime started.", "::: training"]
    for step in range(1, steps + 1):
        reward = round(0.05 + 0.005 * (step - 1), 4)
        lines.append(mirror_line(step, **{"reward/avg_raw_reward": reward}))
    lines.append("Training complete.")
    return "\n".join(lines)


@pytest.fixture
def spec() -> GateSpec:
    return GateSpec(
        min_train_steps=2,
        finite_metrics=("policy/policy_loss", "reward/avg_raw_reward"),
        bounds={"reward/avg_raw_reward": MetricBound(0.0, 1.0)},
        max_wall_clock_seconds=900,
    )


def trend_spec(window: int = 2, min_improvement: float = 0.03, min_train_steps: int = 4) -> GateSpec:
    return GateSpec(
        min_train_steps=min_train_steps,
        finite_metrics=(),
        bounds={},
        max_wall_clock_seconds=900,
        metric_series=(
            MetricSeries(
                kind="train",
                metric="reward/avg_raw_reward",
                required=True,
                min_observations=min_train_steps,
                trend=SeriesTrend(window, min_improvement),
            ),
        ),
    )


def reward_log(rewards: list[float]) -> str:
    return "\n".join(mirror_line(i + 1, **{"reward/avg_raw_reward": r}) for i, r in enumerate(rewards))


def test_parse_metrics_reads_payloads_out_of_decorated_log_lines():
    steps = parse_metrics(healthy_log(steps=2))
    assert [(s.kind, s.step) for s in steps] == [("train", 1), ("train", 2)]
    assert steps[-1].values["reward/avg_raw_reward"] == 0.055


def test_parse_metrics_ignores_a_log_with_no_payloads():
    assert parse_metrics("Ray runtime started.\nCUDA out of memory.\n") == []


def test_parse_metrics_strips_ansi_colour_codes():
    """loguru colorizes even when piped, wrapping the payload in SGR escapes whose trailing reset
    (`}\x1b[0m`) would otherwise defeat the end-of-line anchor and parse zero steps out of a
    perfectly healthy run."""
    coloured = f"\x1b[32m{mirror_line(1)}\x1b[0m"
    steps = parse_metrics(coloured)
    assert [(s.kind, s.step) for s in steps] == [("train", 1)]


def test_healthy_run_passes(spec):
    assert check_run(parse_metrics(healthy_log()), spec, wall_clock_seconds=300) == []


def test_run_that_diverged_to_nan_fails(spec):
    # The trainer serialises a non-finite loss as the JSON literal NaN, which json.loads
    # reads back as float("nan").
    log = "\n".join([mirror_line(1), mirror_line(2, **{"policy/policy_loss": float("nan")})])
    failures = check_run(parse_metrics(log), spec, wall_clock_seconds=300)
    assert len(failures) == 1
    assert "policy/policy_loss" in failures[0]


def test_run_healthy_early_but_broken_at_the_final_step_fails(spec):
    """A run can look fine for a step and then degrade; the last step is what is gated."""
    log = "\n".join([mirror_line(1), mirror_line(2, **{"policy/policy_loss": float("inf")})])
    assert check_run(parse_metrics(log), spec, wall_clock_seconds=300) != []


def test_run_that_stopped_early_fails(spec):
    failures = check_run(parse_metrics(healthy_log(steps=1)), spec, wall_clock_seconds=300)
    assert len(failures) == 1
    assert "expected at least 2" in failures[0]


def test_run_that_logged_nothing_fails(spec):
    assert check_run([], spec, wall_clock_seconds=300) != []


def test_missing_required_metric_fails(spec):
    log = "\n".join([mirror_line(1), mirror_line(2, drop=("reward/avg_raw_reward",))])
    failures = check_run(parse_metrics(log), spec, wall_clock_seconds=300)
    assert len(failures) == 1
    assert "did not log reward/avg_raw_reward" in failures[0]


@pytest.mark.parametrize("reward", [-0.1, 1.5])
def test_reward_outside_the_environments_range_fails(spec, reward):
    """gsm8k scores each rollout 0 or 1, so a mean outside [0, 1] means the reward path broke."""
    log = "\n".join([mirror_line(1), mirror_line(2, **{"reward/avg_raw_reward": reward})])
    failures = check_run(parse_metrics(log), spec, wall_clock_seconds=300)
    assert len(failures) == 1
    assert "outside [0.0, 1.0]" in failures[0]


def test_run_over_the_wall_clock_budget_fails(spec):
    failures = check_run(parse_metrics(healthy_log()), spec, wall_clock_seconds=901)
    assert len(failures) == 1
    assert "budget" in failures[0]


def test_required_log_patterns_have_named_inclusive_bounds(spec):
    spec = replace(
        spec,
        required_log_patterns={
            "compaction": LogPatternBound(r"history (?:did not grow|was rewritten)", 1, 2),
            "timeout": LogPatternBound(r"AgentTimeoutError", 1, 1),
        },
    )
    log = "history did not grow\nAgentTimeoutError\nhistory was rewritten\n"

    assert check_log_patterns(log, spec) == []
    failures = check_log_patterns(log + "AgentTimeoutError\n", spec)
    assert failures == ["log pattern 'timeout' occurred 2 times, expected at most 1"]


def test_eval_payloads_do_not_count_as_training_steps(spec):
    log = "\n".join([mirror_line(1), mirror_line(1, kind="eval"), mirror_line(2, kind="eval")])
    failures = check_run(parse_metrics(log), spec, wall_clock_seconds=300)
    assert "expected at least 2" in failures[0]

    eval_spec = replace(
        spec,
        min_train_steps=0,
        finite_metrics=(),
        bounds={},
        metric_series=(MetricSeries("eval", "eval/exact", True, 1),),
    )
    assert check_run([StepMetrics("eval", 0, {"eval/exact": 1.0})], eval_spec, 300) == []
    assert check_run([], eval_spec, 300) != []
    assert check_run([StepMetrics("eval", 0, {"eval/exact": float("nan")})], eval_spec, 300) != []


@pytest.mark.parametrize("rewards,window", [([0.25] * 4, 2), ([1e308] * 9 + [0.5], 5)])
def test_insufficient_reward_improvement_fails_the_trend_gate(rewards, window):
    failures = check_run(parse_metrics(reward_log(rewards)), trend_spec(window=window), wall_clock_seconds=300)
    assert len(failures) == 1
    assert "expected at least +0.0300" in failures[0]


@pytest.mark.parametrize("rewards", [[0.05, 0.06, 0.20, 0.22], [-1e308, -1e308, 1e308, 1e308]])
def test_rising_reward_passes_the_trend_gate(rewards):
    failures = check_run(parse_metrics(reward_log(rewards)), trend_spec(), wall_clock_seconds=300)
    assert failures == []


def test_trend_gate_fails_when_there_are_too_few_steps_to_judge():
    spec = trend_spec(window=5, min_improvement=0.03, min_train_steps=1)
    failures = check_run(parse_metrics(reward_log([0.25, 0.25, 0.25])), spec, wall_clock_seconds=300)
    assert any("expected at least 10 for its trend" in failure for failure in failures)


def test_duplicate_payloads_do_not_count_as_completed_steps(spec):
    first = parse_metrics(mirror_line(1))[0]
    failures = check_run([first, first], spec, wall_clock_seconds=300)
    assert any("logged 1 training steps" in failure for failure in failures)

    changed = replace(first, values={**first.values, "policy/policy_loss": 0.1})
    failures = check_run([first, changed], spec, wall_clock_seconds=300)
    assert any("conflicting train payloads at step 1" in failure for failure in failures)

    startup = [
        StepMetrics("startup", 0, {"startup/model_loading": 2.0}),
        StepMetrics("startup", 0, {"startup/eval_before_train": 1.0}),
    ]
    assert check_run([*startup, *parse_metrics(healthy_log())], spec, 300) == []

    boolean_value = replace(first, values={**first.values, "x": True})
    numeric_value = replace(first, values={**first.values, "x": 1})
    assert any(
        "conflicting train payloads" in failure for failure in check_run([boolean_value, numeric_value], spec, 300)
    )

    nan_copies = parse_metrics("\n".join([mirror_line(1, **{"policy/policy_loss": float("nan")})] * 2))
    failures = check_run(nan_copies, replace(spec, min_train_steps=1), 300)
    assert len(failures) == 1
    assert "not finite" in failures[0]


def test_cat_count_series_requires_finite_learning_and_enough_train_and_eval_evidence(tmp_path):
    spec_path = tmp_path / "cat-count-spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "min_train_steps": 6,
                "finite_metrics": [],
                "bounds": {},
                "max_wall_clock_seconds": 600,
                "metric_series": [
                    {
                        "kind": "train",
                        "metric": "policy/policy_loss",
                        "required": True,
                        "min_observations": 6,
                        "finite_every_step": True,
                    },
                    {
                        "kind": "train",
                        "metric": "environment/exact_n10",
                        "required": True,
                        "min_observations": 5,
                        "trend": {"window": 2, "min_improvement": 0.2},
                    },
                    {
                        "kind": "train",
                        "metric": "reward/zero_std_group_fraction",
                        "required": True,
                        "min_observations": 6,
                        "occurrence": {"minimum_count": 4, "comparison": "above", "threshold": 0.0},
                    },
                    {
                        "kind": "train",
                        "metric": "policy/ppo_ratio_exact_unit_fraction",
                        "required": True,
                        "min_observations": 6,
                        "occurrence": {"minimum_count": 1, "comparison": "below", "threshold": 1.0},
                    },
                    {
                        "kind": "eval",
                        "metric": "eval/cat_count_n10/avg_score",
                        "required": True,
                        "min_observations": 3,
                        "trend": {"window": 1, "min_improvement": 0.4},
                    },
                    {
                        "kind": "train",
                        "metric": "policy/ppo_clip_ratio",
                        "required": False,
                        "min_observations": 1,
                    },
                ],
            }
        )
    )
    spec = load_spec(spec_path)
    train = [
        StepMetrics(
            "train",
            step,
            {
                "policy/policy_loss": 0.4,
                "reward/zero_std_group_fraction": 0.0 if step == 1 else 0.5,
                "policy/ppo_ratio_exact_unit_fraction": 0.9 if step == 6 else 1.0,
                **({"environment/exact_n10": exact} if exact is not None else {}),
            },
        )
        for step, exact in enumerate((0.0, None, 0.1, 0.2, 0.4, 0.6), start=1)
    ]
    evaluation = [
        StepMetrics("eval", step, {"eval/cat_count_n10/avg_score": score})
        for step, score in ((0, 0.0), (3, 0.3), (6, 0.8))
    ]
    steps = [*train, *evaluation]
    assert check_run(steps, spec, wall_clock_seconds=300) == []

    nan_loss = [
        replace(step, values={**step.values, "policy/policy_loss": float("nan")})
        if step.step == 3 and step.kind == "train"
        else step
        for step in steps
    ]
    assert any("nonfinite policy/policy_loss" in failure for failure in check_run(nan_loss, spec, 300))

    missing_loss = [
        replace(step, values={key: value for key, value in step.values.items() if key != "policy/policy_loss"})
        if step.step == 3 and step.kind == "train"
        else step
        for step in steps
    ]
    assert any("step 3 did not log policy/policy_loss" in failure for failure in check_run(missing_loss, spec, 300))

    flat = [
        replace(step, values={**step.values, "environment/exact_n10": 0.1})
        if "environment/exact_n10" in step.values
        else step
        for step in steps
    ]
    assert any("environment/exact_n10 rose by" in failure for failure in check_run(flat, spec, 300))

    missing = [
        replace(step, values={key: value for key, value in step.values.items() if key != "environment/exact_n10"})
        if step.kind == "train" and step.step == 3
        else step
        for step in steps
    ]
    assert any(
        "environment/exact_n10 has 4 finite observations" in failure for failure in check_run(missing, spec, 300)
    )

    flat_eval = [
        replace(step, values={"eval/cat_count_n10/avg_score": 0.1}) if step.kind == "eval" else step for step in steps
    ]
    assert any("eval eval/cat_count_n10/avg_score rose by" in failure for failure in check_run(flat_eval, spec, 300))

    no_eval = [step for step in steps if step.kind == "train"]
    assert any(
        "eval eval/cat_count_n10/avg_score has 0 finite observations" in failure
        for failure in check_run(no_eval, spec, 300)
    )

    flat_groups = [
        replace(step, values={**step.values, "reward/zero_std_group_fraction": 0.0}) if step.kind == "train" else step
        for step in steps
    ]
    assert any(
        "reward/zero_std_group_fraction has 0 observations above 0.0" in failure
        for failure in check_run(flat_groups, spec, 300)
    )

    no_clip = [
        replace(step, values={**step.values, "policy/ppo_ratio_exact_unit_fraction": 1.0})
        if step.kind == "train"
        else step
        for step in steps
    ]
    assert any(
        "policy/ppo_ratio_exact_unit_fraction has 0 observations below 1.0" in failure
        for failure in check_run(no_clip, spec, 300)
    )

    bad_optional = [
        replace(step, values={**step.values, "policy/ppo_clip_ratio": float("nan")})
        if step.step == 4 and step.kind == "train"
        else step
        for step in steps
    ]
    assert any("nonfinite policy/ppo_clip_ratio" in failure for failure in check_run(bad_optional, spec, 300))


def test_shipped_spec_gates_a_healthy_run():
    """The checked-in spec has to stay loadable by the gate and pass a plausible run."""
    for path in (SHIPPED_SPEC, FSDP_SPEC):
        spec = load_spec(path)
        assert check_run(parse_metrics(healthy_log(steps=spec.min_train_steps)), spec, wall_clock_seconds=600) == []


def test_opencode_spec_requires_exact_concurrent_literal_coverage():
    spec = load_spec(OPENCODE_SPEC)
    exact_metrics = {
        "generate/failed_trajectory_fraction": 0.0,
        "generate/literal_bridge/correlated_trials": 8.0,
        "generate/literal_bridge/correlated_turns": 24.0,
        "generate/tis/exact_match_fraction": 1.0,
        "generate/tis/lcs_fallback_fraction": 0.0,
        "generate/tis/unaligned_fraction": 0.0,
        "generate/tis/tito_full/success_fraction": 1.0,
        "generate/tis/tito_full/decline_count": 0.0,
        "tis/skipped_fraction": 0.0,
    }
    healthy = parse_metrics(mirror_line(1, **exact_metrics))
    assert check_run(healthy, spec, wall_clock_seconds=900) == []

    approximate = parse_metrics(
        mirror_line(
            1,
            **{
                **exact_metrics,
                "generate/tis/exact_match_fraction": 0.99,
                "generate/tis/lcs_fallback_fraction": 0.01,
            },
        )
    )
    failures = check_run(approximate, spec, wall_clock_seconds=900)
    assert any("exact_match_fraction" in failure for failure in failures)
    assert any("lcs_fallback_fraction" in failure for failure in failures)
