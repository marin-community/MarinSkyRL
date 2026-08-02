from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from rl_cleanup import parse_skyrl_metrics


def write_trial_result(trace_jobs_dir: Path, result: dict[str, Any]) -> None:
    trial_dir = trace_jobs_dir / result["trial_name"]
    trial_dir.mkdir(parents=True)
    (trial_dir / "result.json").write_text(json.dumps(result))


def timing_frame(**seconds: float) -> pd.DataFrame:
    return pd.DataFrame([{f"timing/{name}": value for name, value in seconds.items()}])


def span_named(spans: list[parse_skyrl_metrics.TimingSpan], name: str) -> parse_skyrl_metrics.TimingSpan:
    (span,) = [span for span in spans if span.name == name]
    return span


def test_default_parser_reads_agentic_wandb_json(tmp_path):
    log_path = tmp_path / "terminus-trainer.out"
    log_path.write_text(
        'WANDB_MIRROR kind=train step=7 metrics={"async/staleness_max": 0, '
        '"reward/avg_raw_reward": 0.75, "trainer/global_step": 7}\n'
    )

    result = parse_skyrl_metrics.process_log_file(log_path)

    assert result.metrics == [
        {
            "async/staleness_max": 0,
            "reward/avg_raw_reward": 0.75,
            "trainer/global_step": 7,
        }
    ]
    assert result.serialization is parse_skyrl_metrics.MetricSerialization.WANDB_JSON


def test_legacy_python_dict_log_is_auto_detected(tmp_path, monkeypatch):
    log_path = tmp_path / "trainer.out"
    log_path.write_text(
        "\n".join(
            [
                "{'async/staleness_max': 0, 'reward/avg_raw_reward': 0.2, 'trainer/global_step': 20}",
                "{'async/staleness_max': 0, 'reward/avg_raw_reward': 0.8, 'trainer/global_step': 40}",
                "{'async/staleness_max': 0, 'reward/avg_raw_reward': 0.5, 'trainer/global_step': 60}",
            ]
        )
    )
    trace_jobs_dir = tmp_path / "trace_jobs"
    trace_jobs_dir.mkdir()
    output_path = tmp_path / "results"
    monkeypatch.setattr(parse_skyrl_metrics, "generate_reward_plot", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "parse_skyrl_metrics.py",
            str(log_path),
            str(output_path),
            "--trace_jobs_dir",
            str(trace_jobs_dir),
            "--save_every",
            "20",
        ],
    )

    parse_skyrl_metrics.main()

    report = (output_path / "report.md").read_text()
    assert "## Best Checkpoint (trailing-5 EMA of reward/avg_raw_reward)" in report
    assert "| 40 | 0.8000 | 0.4000 | yes |" in report


def test_json_log_with_trace_directory_uses_trace_analysis(tmp_path, monkeypatch):
    log_path = tmp_path / "terminus-trainer.out"
    log_path.write_text(
        'WANDB_MIRROR kind=train step=7 metrics={"async/staleness_max": 0, '
        '"reward/avg_raw_reward": 0.75, "trainer/global_step": 7}\n'
    )
    trace_jobs_dir = tmp_path / "trace_jobs"
    trace_jobs_dir.mkdir()
    trial_dir = trace_jobs_dir / "trial-1"
    trial_dir.mkdir()
    (trial_dir / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task-1",
                "trial_name": "trial-1",
                "agent_result": {"metadata": {"n_episodes": 3}},
                "verifier_result": {"rewards": {"reward": 1.0}},
            }
        )
    )
    output_path = tmp_path / "results"
    monkeypatch.setattr(parse_skyrl_metrics, "generate_reward_plot", lambda *args, **kwargs: None)
    monkeypatch.setattr(parse_skyrl_metrics, "generate_turn_count_plot", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "parse_skyrl_metrics.py",
            str(log_path),
            str(output_path),
            "--trace_jobs_dir",
            str(trace_jobs_dir),
        ],
    )

    parse_skyrl_metrics.main()

    report = (output_path / "report.md").read_text()
    assert "Total trials parsed: 1" in report
    assert "Success rate: 100.0%" in report


def test_missing_explicit_trace_directory_fails_loudly(tmp_path, monkeypatch):
    log_path = tmp_path / "terminus-trainer.out"
    log_path.write_text(
        'WANDB_MIRROR kind=train step=7 metrics={"reward/avg_raw_reward": 0.75, '
        '"trainer/global_step": 7}\n'
    )
    missing_trace_dir = tmp_path / "missing-trace-jobs"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "parse_skyrl_metrics.py",
            str(log_path),
            str(tmp_path / "results"),
            "--trace_jobs_dir",
            str(missing_trace_dir),
        ],
    )

    with pytest.raises(ValueError, match=str(missing_trace_dir)):
        parse_skyrl_metrics.main()


def test_parsed_trial_reports_a_duration_for_every_harbor_phase(tmp_path):
    trace_jobs_dir = tmp_path / "trace_jobs"
    write_trial_result(
        trace_jobs_dir,
        {
            "task_name": "task-1",
            "trial_name": "trial-1",
            "environment_setup": {
                "started_at": "2026-01-01T00:00:00Z",
                "finished_at": "2026-01-01T00:01:18Z",
            },
            "agent_setup": {
                "started_at": "2026-01-01T00:01:18Z",
                "finished_at": "2026-01-01T00:01:39Z",
            },
            "agent_execution": {
                "started_at": "2026-01-01T00:01:39Z",
                "finished_at": "2026-01-01T00:02:12Z",
            },
            "verifier": {
                "started_at": "2026-01-01T00:02:12Z",
                "finished_at": "2026-01-01T00:02:13Z",
            },
        },
    )

    (trial,) = parse_skyrl_metrics.parse_result_files(trace_jobs_dir)

    assert trial["environment_setup_duration"] == 78.0
    assert trial["agent_setup_duration"] == 21.0
    assert trial["agent_execution_duration"] == 33.0
    assert trial["verifier_duration"] == 1.0


@pytest.mark.parametrize(
    "environment_setup_field",
    [
        pytest.param({}, id="phase-absent"),
        pytest.param({"environment_setup": None}, id="phase-null"),
        pytest.param(
            {"environment_setup": {"started_at": "2026-01-01T00:00:00Z"}},
            id="phase-never-finished",
        ),
        pytest.param(
            {"environment_setup": {"started_at": "just after lunch", "finished_at": "2026-01-01T00:01:18Z"}},
            id="phase-unparseable-timestamp",
        ),
    ],
)
def test_unusable_phase_timing_leaves_the_duration_empty(tmp_path, environment_setup_field):
    trace_jobs_dir = tmp_path / "trace_jobs"
    write_trial_result(
        trace_jobs_dir,
        {
            "task_name": "task-1",
            "trial_name": "trial-1",
            "agent_execution": {
                "started_at": "2026-01-01T00:01:39Z",
                "finished_at": "2026-01-01T00:02:12Z",
            },
            **environment_setup_field,
        },
    )

    (trial,) = parse_skyrl_metrics.parse_result_files(trace_jobs_dir)

    assert trial["environment_setup_duration"] is None
    # A phase that cannot be timed must not cost the trial its other phases.
    assert trial["agent_execution_duration"] == 33.0


def test_nested_optimizer_span_is_measured_against_its_container_not_the_step():
    spans = parse_skyrl_metrics.summarize_timing_spans(
        timing_frame(step=100.0, generate=60.0, train_critic_and_policy=30.0, policy_train=28.0)
    )

    policy_train = span_named(spans, "policy_train")
    assert policy_train.within == "train_critic_and_policy"
    assert policy_train.share_of_within == pytest.approx(28.0 / 30.0)
    # Charging both to the step would spend 58% of it twice over.
    assert span_named(spans, "train_critic_and_policy").share_of_within == pytest.approx(0.3)


def test_synchronous_tree_attributes_optimizer_work_straight_to_the_step():
    spans = parse_skyrl_metrics.summarize_timing_spans(
        timing_frame(step=100.0, generate=60.0, fwd_logprobs_values_reward=10.0, train_critic_and_policy=30.0)
    )

    assert span_named(spans, "fwd_logprobs_values_reward").within == "step"
    assert span_named(spans, "train_critic_and_policy").within == "step"


def test_asynchronous_tree_attributes_optimizer_work_to_the_training_span():
    spans = parse_skyrl_metrics.summarize_timing_spans(
        timing_frame(
            step=100.0,
            wait_for_generation_buffer=55.0,
            run_training=40.0,
            fwd_logprobs_values_reward=10.0,
            train_critic_and_policy=30.0,
        )
    )

    assert span_named(spans, "fwd_logprobs_values_reward").within == "run_training"
    assert span_named(spans, "train_critic_and_policy").within == "run_training"
    assert span_named(spans, "run_training").within == "step"


def test_work_between_steps_reports_no_share_of_the_step():
    spans = parse_skyrl_metrics.summarize_timing_spans(
        timing_frame(step=100.0, generate=90.0, save_checkpoints=45.0, eval=20.0)
    )

    for name in ("save_checkpoints", "eval"):
        span = span_named(spans, name)
        assert span.within is None
        assert span.share_of_within is None


def test_undeclared_span_is_reported_without_a_share():
    spans = parse_skyrl_metrics.summarize_timing_spans(timing_frame(step=100.0, some_new_trainer_phase=25.0))

    span = span_named(spans, "some_new_trainer_phase")
    assert span.within is None
    assert span.share_of_within is None
    assert span.mean_seconds == pytest.approx(25.0)


def test_unattributed_span_covers_the_step_time_no_child_claims():
    spans = parse_skyrl_metrics.summarize_timing_spans(
        timing_frame(step=100.0, generate=60.0, sync_weights=10.0, save_checkpoints=45.0)
    )

    unattributed = span_named(spans, "unattributed")
    assert unattributed.within == "step"
    # The 45 s checkpoint save is outside the step and must not shrink the remainder.
    assert unattributed.mean_seconds == pytest.approx(30.0)
    assert unattributed.share_of_within == pytest.approx(0.3)


def test_trial_phase_shares_are_taken_against_trial_wall_clock():
    summary = parse_skyrl_metrics.summarize_trial_phases(
        pd.DataFrame(
            [
                {
                    "trial_duration": 200.0,
                    "environment_setup_duration": 100.0,
                    "agent_setup_duration": 20.0,
                    "agent_execution_duration": 60.0,
                    "verifier_duration": 10.0,
                }
            ]
        )
    )

    assert summary["trial_count"] == 1
    assert summary["phases"]["environment_setup"]["share"] == pytest.approx(0.5)
    assert summary["phases"]["verifier"]["share"] == pytest.approx(0.05)
    assert summary["unmeasured_share"] == pytest.approx(0.05)


def test_multi_step_trial_still_reports_the_phases_it_records():
    summary = parse_skyrl_metrics.summarize_trial_phases(
        pd.DataFrame(
            [
                {
                    "trial_duration": 300.0,
                    "environment_setup_duration": 90.0,
                    "agent_setup_duration": 30.0,
                    # Harbor records these per step on a multi-step trial, not on the trial.
                    "agent_execution_duration": None,
                    "verifier_duration": None,
                }
            ]
        )
    )

    assert set(summary["phases"]) == {"environment_setup", "agent_setup"}
    assert summary["phases"]["environment_setup"]["count"] == 1
    assert summary["unmeasured_share"] == pytest.approx(0.6)


def test_phases_are_still_reported_when_the_trial_wall_clock_is_missing():
    summary = parse_skyrl_metrics.summarize_trial_phases(
        pd.DataFrame(
            [
                {
                    "trial_duration": None,
                    "environment_setup_duration": 90.0,
                    "agent_setup_duration": 30.0,
                    "agent_execution_duration": None,
                    "verifier_duration": None,
                }
            ]
        )
    )

    assert summary["phases"]["environment_setup"]["total"] == pytest.approx(90.0)
    # Without a denominator the report must withhold shares rather than invent one.
    assert "share" not in summary["phases"]["environment_setup"]
    assert "unmeasured_share" not in summary


def test_report_publishes_the_trial_phase_breakdown(tmp_path, monkeypatch):
    log_path = tmp_path / "terminus-trainer.out"
    log_path.write_text(
        'WANDB_MIRROR kind=train step=1 metrics={"reward/avg_raw_reward": 0.5, "trainer/global_step": 1, '
        '"timing/step": 100.0, "timing/generate": 60.0, "timing/train_critic_and_policy": 30.0, '
        '"timing/policy_train": 28.0, "timing/save_checkpoints": 45.0}\n'
    )
    trace_jobs_dir = tmp_path / "trace_jobs"
    write_trial_result(
        trace_jobs_dir,
        {
            "task_name": "task-1",
            "trial_name": "trial-1",
            "started_at": "2026-01-01T00:00:00Z",
            "finished_at": "2026-01-01T00:03:20Z",
            "environment_setup": {"started_at": "2026-01-01T00:00:00Z", "finished_at": "2026-01-01T00:01:40Z"},
            "agent_setup": {"started_at": "2026-01-01T00:01:40Z", "finished_at": "2026-01-01T00:02:00Z"},
            "agent_execution": {"started_at": "2026-01-01T00:02:00Z", "finished_at": "2026-01-01T00:03:00Z"},
            "verifier": {"started_at": "2026-01-01T00:03:00Z", "finished_at": "2026-01-01T00:03:10Z"},
        },
    )
    output_path = tmp_path / "results"
    monkeypatch.setattr(parse_skyrl_metrics, "generate_reward_plot", lambda *args, **kwargs: None)
    monkeypatch.setattr(parse_skyrl_metrics, "generate_turn_count_plot", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        sys,
        "argv",
        ["parse_skyrl_metrics.py", str(log_path), str(output_path), "--trace_jobs_dir", str(trace_jobs_dir)],
    )

    parse_skyrl_metrics.main()

    report = (output_path / "report.md").read_text()
    assert "| environment_setup | 100.0 | 100.0 | 100.0 | 50.0% | 1 |" in report
    assert "| unmeasured | — | — | 10.0 | 5.0% | — |" in report
    assert "| policy_train | `train_critic_and_policy` | 28.0 | 93.3% | 1 |" in report
    assert "| save_checkpoints | outside `step` | 45.0 | — | 1 |" in report


def test_checkpoint_selection_calculates_trailing_five_ema():
    selection = parse_skyrl_metrics.select_best_checkpoint(
        [
            {"trainer/global_step": 20, "reward/avg_raw_reward": 0.2},
            {"trainer/global_step": 40, "reward/avg_raw_reward": 0.8},
        ],
        save_every=20,
    )
    assert selection.ema[40] == pytest.approx(0.4)


def test_checkpoint_selection_rejects_an_invalid_cap(tmp_path):
    (tmp_path / "latest_ckpt_global_step.txt").write_text("not-a-step")

    with pytest.raises(ValueError, match="latest_ckpt_global_step.txt"):
        parse_skyrl_metrics.select_best_checkpoint(
            [{"trainer/global_step": 40, "reward/avg_raw_reward": 0.8}],
            run_dir=tmp_path,
            save_every=20,
        )


def test_checkpoint_selection_rejects_an_invalid_reward():
    with pytest.raises(ValueError, match="reward='not-a-reward'"):
        parse_skyrl_metrics.select_best_checkpoint(
            [{"trainer/global_step": 40, "reward/avg_raw_reward": "not-a-reward"}],
            save_every=20,
        )
