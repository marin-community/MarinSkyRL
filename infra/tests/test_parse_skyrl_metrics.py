import json
import sys
from pathlib import Path
from typing import Any

import pytest

from rl_cleanup import parse_skyrl_metrics


def write_trial_result(trace_jobs_dir: Path, result: dict[str, Any]) -> None:
    trial_dir = trace_jobs_dir / result["trial_name"]
    trial_dir.mkdir(parents=True)
    (trial_dir / "result.json").write_text(json.dumps(result))


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
