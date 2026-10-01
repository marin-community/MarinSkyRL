"""Tests for the nightly end-to-end gate (ci/marin_nightly/gate.py).

Run with: uv run --isolated --group dev --extra cpu pytest tests/cpu/test_marin_nightly_gate.py
"""

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from skyrl_train.objective.teacher import teacher_advantages

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
OPD_SPEC = Path(__file__).parents[2] / "ci" / "marin_nightly" / "specs" / "opd-qwen3-sync.json"

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


@pytest.mark.parametrize(
    "rewards,window,expected_pass,expected_message",
    [
        ([0.25] * 4, 2, False, "rose by"),
        ([0.05, 0.06, 0.20, 0.22], 2, True, ""),
        ([0.25] * 3, 5, False, "expected at least 10 for its trend"),
    ],
)
def test_reward_trend_gate(rewards, window, expected_pass, expected_message):
    failures = check_run(
        parse_metrics(reward_log(rewards)), trend_spec(window=window, min_train_steps=1), wall_clock_seconds=300
    )
    assert (failures == []) is expected_pass
    if not expected_pass:
        assert len(failures) == 1
        assert expected_message in failures[0]


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

    nan_copies = parse_metrics("\n".join([mirror_line(1, **{"policy/policy_loss": float("nan")})] * 2))
    failures = check_run(nan_copies, replace(spec, min_train_steps=1), 300)
    assert len(failures) == 1
    assert "not finite" in failures[0]


@pytest.mark.parametrize(
    "mutation,expected",
    [
        ("healthy", ""),
        ("nan_loss", "nonfinite policy/policy_loss"),
        ("missing_loss", "step 3 did not log policy/policy_loss"),
        ("flat_train", "environment/exact_n10 rose by"),
        ("sparse_train", "environment/exact_n10 has 4 finite observations"),
        ("flat_eval", "eval eval/cat_count_n10/avg_score rose by"),
        ("no_eval", "eval eval/cat_count_n10/avg_score has 0 finite observations"),
        ("no_zero_variance", "reward/zero_std_group_fraction has 0 observations above 0.0"),
        ("no_ratio_change", "policy/ppo_ratio_exact_unit_fraction has 0 observations below 1.0"),
        ("nan_optional", "nonfinite policy/ppo_clip_ratio"),
    ],
)
def test_cat_count_series_requires_finite_learning_and_enough_evidence(tmp_path, mutation, expected):
    path = tmp_path / "spec.json"
    path.write_text(
        json.dumps(
            {
                "min_train_steps": 6,
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
                    {"kind": "train", "metric": "policy/ppo_clip_ratio", "required": False, "min_observations": 1},
                ],
            }
        )
    )
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
    for index, row in enumerate(steps):
        values = dict(row.values)
        if row.kind == "train":
            if mutation == "nan_loss" and row.step == 3:
                values["policy/policy_loss"] = float("nan")
            if mutation == "missing_loss" and row.step == 3:
                values.pop("policy/policy_loss")
            if mutation == "flat_train" and "environment/exact_n10" in values:
                values["environment/exact_n10"] = 0.1
            if mutation == "sparse_train" and row.step == 3:
                values.pop("environment/exact_n10")
            if mutation == "no_zero_variance":
                values["reward/zero_std_group_fraction"] = 0.0
            if mutation == "no_ratio_change":
                values["policy/ppo_ratio_exact_unit_fraction"] = 1.0
            if mutation == "nan_optional" and row.step == 4:
                values["policy/ppo_clip_ratio"] = float("nan")
        elif mutation == "flat_eval":
            values["eval/cat_count_n10/avg_score"] = 0.1
        steps[index] = replace(row, values=values)
    if mutation == "no_eval":
        steps = train
    failures = check_run(steps, load_spec(path), 300)
    if expected:
        assert any(expected in failure for failure in failures)
    else:
        assert failures == []


@pytest.mark.parametrize(
    "mutation",
    ["healthy", "flat_eval", "late_crossing", "dp_divergence", "missing_step_zero", "missing_initial_metric"],
)
def test_cat_count_shipped_specs_require_learning_from_step_zero(mutation):
    path = SHIPPED_SPEC.parent / "cat-count-canary-qwen2.5-0.5b-async.json"
    spec = load_spec(path)
    metrics = {
        "policy/policy_loss": 0.1,
        "policy/final_loss": 0.1,
        "policy/policy_entropy": 1.0,
        "reward/zero_std_group_fraction": 0.2,
        "policy/ppo_clip_ratio": 0.01,
        "policy/ppo_ratio_exact_unit_fraction": 0.9,
        "policy/mismatch/pooled/log_ratio_abs_mean": 0.01,
        "async/staleness_mean": 1.0,
        "policy/dp_weight_checksum_mismatch": 0.0,
        "environment/exact_n10": 0.5,
        "environment/exact_n20": 0.25,
        "policy/rollout_train_prob_diff_mean": 1.003,
    }
    steps = [StepMetrics("train", step, metrics) for step in range(1, max(10, spec.min_train_steps) + 1)]
    evaluations = [
        StepMetrics("eval", step, {"eval/sampled/train/avg_score": score})
        for step, score in ((0, 0.25), (5, 0.30), (10, 0.65))
    ]
    if mutation == "flat_eval":
        evaluations[-1] = replace(evaluations[-1], values={"eval/sampled/train/avg_score": 0.25})
    if mutation == "late_crossing":
        evaluations[-1] = replace(evaluations[-1], step=35)
    if mutation == "dp_divergence":
        steps[1] = replace(steps[1], values={**metrics, "policy/dp_weight_checksum_mismatch": 1.0})
    if mutation == "missing_step_zero":
        evaluations = evaluations[1:]
    if mutation == "missing_initial_metric":
        evaluations[0] = replace(evaluations[0], values={})
    failures = check_run([*steps, *evaluations], spec, 300)
    if mutation == "healthy":
        assert failures == []
    else:
        assert failures
    assert check_log_patterns("Training done!\n[telemetry] enabled run_id=test\n", spec) == []


@pytest.mark.parametrize(
    "selector,bound,selected_step,bad_score",
    [
        (0, MetricBound(0.0, 0.1), 0, 0.3),
        ("first", MetricBound(0.0, 0.1), 0, 0.3),
        ("last", MetricBound(0.7, 1.0), 6, 0.6),
    ],
)
def test_evaluation_step_selection_requires_the_selected_value(spec, selector, bound, selected_step, bad_score):
    spec = replace(
        spec,
        min_train_steps=0,
        finite_metrics=(),
        bounds={},
        metric_series=(MetricSeries("eval", "eval/score", True, 1, bounds=bound, at_step=selector),),
    )
    rows = [StepMetrics("eval", step, {"eval/score": score}) for step, score in ((0, 0.0), (3, 0.3), (6, 0.8))]
    assert check_run(list(reversed(rows)), spec, 300) == []
    broken = [replace(row, values={"eval/score": bad_score}) if row.step == selected_step else row for row in rows]
    assert check_run(broken, spec, 300)
    missing = [row for row in rows if row.step != selected_step]
    assert check_run(missing, spec, 300)


def test_shipped_spec_gates_a_healthy_run():
    """The checked-in spec has to stay loadable by the gate and pass a plausible run."""
    spec = load_spec(SHIPPED_SPEC)
    assert check_run(parse_metrics(healthy_log(steps=spec.min_train_steps)), spec, wall_clock_seconds=600) == []
    early_nan = parse_metrics(healthy_log(steps=spec.min_train_steps))
    early_nan[0] = replace(early_nan[0], values={**early_nan[0].values, "policy/policy_loss": float("nan")})
    assert check_run(early_nan, spec, wall_clock_seconds=600)


def test_opd_gate_requires_teacher_credit_on_valid_training_tokens():
    spec = load_spec(OPD_SPEC)
    for eligible in (True, False):
        _, metrics = teacher_advantages(
            torch.tensor([[-0.4, -1.2]]),
            torch.tensor([[-1.0, -1.0]]),
            torch.full((1, 2), eligible),
            torch.tensor([[0.3, 0.7]]),
            clip=None,
        )
        metrics.update({"policy/raw_grad_norm": 0.5, "distillation/scored_tokens": 2, "distillation/teacher_count": 1})
        failures = check_run(parse_metrics(mirror_line(1, **metrics)), spec, wall_clock_seconds=300)
        assert (failures == []) == eligible
