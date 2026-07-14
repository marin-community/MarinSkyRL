"""Tests for the nightly end-to-end gate (ci/marin_nightly/gate.py).

Run with: uv run --isolated --extra dev pytest tests/cpu/test_marin_nightly_gate.py
"""

import json
from pathlib import Path

import pytest

from ci.marin_nightly.gate import GateSpec, MetricBound, check_run, load_spec, parse_metrics

SHIPPED_SPEC = Path(__file__).parents[2] / "ci" / "marin_nightly" / "specs" / "gsm8k-qwen3-0.6b.json"

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
    lines = ["Ray runtime started.", "::: training"]
    for step in range(1, steps + 1):
        lines.append(mirror_line(step))
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


def test_parse_metrics_reads_payloads_out_of_decorated_log_lines():
    steps = parse_metrics(healthy_log(steps=2))
    assert [(s.kind, s.step) for s in steps] == [("train", 1), ("train", 2)]
    assert steps[-1].values["reward/avg_raw_reward"] == 0.25


def test_parse_metrics_ignores_a_log_with_no_payloads():
    assert parse_metrics("Ray runtime started.\nCUDA out of memory.\n") == []


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


def test_eval_payloads_do_not_count_as_training_steps(spec):
    log = "\n".join([mirror_line(1), mirror_line(1, kind="eval"), mirror_line(2, kind="eval")])
    failures = check_run(parse_metrics(log), spec, wall_clock_seconds=300)
    assert "expected at least 2" in failures[0]


def test_shipped_spec_gates_a_healthy_run():
    """The checked-in spec has to stay loadable by the gate and pass a plausible run."""
    spec = load_spec(SHIPPED_SPEC)
    assert check_run(parse_metrics(healthy_log(steps=spec.min_train_steps)), spec, wall_clock_seconds=600) == []
