import json
from pathlib import Path

from infra.rl_analysis.pipeline import analyze_local_run
from infra.rl_analysis.traces import load_trace_records


def _write_trace(root: Path, task_id: str, reward: float, timestamp: str, turns: int) -> None:
    task_dir = root / task_id
    task_dir.mkdir(parents=True)
    (task_dir / "result.json").write_text(
        json.dumps(
            {
                "task": task_id,
                "result": {"reward": reward},
                "started_at": timestamp,
                "steps": [{"type": "assistant"}] * turns,
            }
        ),
        encoding="utf-8",
    )


def test_load_trace_records_extracts_local_result_contract(tmp_path: Path) -> None:
    traces_dir = tmp_path / "rollouts"
    _write_trace(traces_dir, "task-1", 1.0, "2026-07-30T00:00:00Z", 3)

    records = load_trace_records(traces_dir)

    assert [(record.task_id, record.reward, record.turns) for record in records] == [("task-1", 1.0, 3)]


def test_analyze_local_run_marks_unmatched_evaluations_invalid(tmp_path: Path) -> None:
    rollout_dir = tmp_path / "rollouts"
    baseline_dir = tmp_path / "baseline"
    post_dir = tmp_path / "post"
    output_dir = tmp_path / "analysis"
    _write_trace(rollout_dir, "training-task", 0.5, "2026-07-30T00:00:00Z", 2)
    _write_trace(rollout_dir, "training-task-2", 1.0, "2026-07-30T04:00:00Z", 4)
    _write_trace(baseline_dir, "baseline-task", 1.0, "2026-07-29T00:00:00Z", 1)
    _write_trace(post_dir, "post-task", 0.0, "2026-07-30T08:00:00Z", 3)

    index_path = analyze_local_run(
        rollout_dir=rollout_dir,
        baseline_dir=baseline_dir,
        post_dir=post_dir,
        training_log_dir=None,
        output_dir=output_dir,
        bin_hours=4,
    )

    index = index_path.read_text(encoding="utf-8")
    validity = json.loads((output_dir / "Q1_behavioral_delta" / "validity.json").read_text())
    temporal = json.loads((output_dir / "Q2_temporal" / "temporal_summary.json").read_text())
    overlay = json.loads((output_dir / "Q3_temporal_overlay" / "overlay.json").read_text())
    evaluation_context = json.loads(
        (output_dir / "Q4_solve_rate_by_context" / "evaluation_context_summary.json").read_text()
    )
    assert validity == {
        "common_task_count": 0,
        "invalid_for_comparison": True,
        "mean_reward_delta": None,
    }
    assert temporal["bins"]["2026-07-30T00:00:00+00:00"]["mean_reward"] == 0.5
    assert temporal["bins"]["2026-07-30T04:00:00+00:00"]["mean_turns"] == 4.0
    assert overlay["validity"] == {"common_task_count": 0, "invalid_for_comparison": True}
    assert evaluation_context["baseline"]["0+"]["mean_reward"] == 1.0
    assert "invalid for before/after conclusions" in index


def test_analyze_local_run_reports_delta_for_matched_evaluations(tmp_path: Path) -> None:
    rollout_dir = tmp_path / "rollouts"
    baseline_dir = tmp_path / "baseline"
    post_dir = tmp_path / "post"
    output_dir = tmp_path / "analysis"
    _write_trace(rollout_dir, "training-task", 0.5, "2026-07-30T00:00:00Z", 2)
    _write_trace(baseline_dir, "shared-task", 0.25, "2026-07-29T00:00:00Z", 1)
    _write_trace(post_dir, "shared-task", 0.75, "2026-07-30T08:00:00Z", 3)

    analyze_local_run(
        rollout_dir=rollout_dir,
        baseline_dir=baseline_dir,
        post_dir=post_dir,
        training_log_dir=None,
        output_dir=output_dir,
        bin_hours=4,
    )

    validity = json.loads((output_dir / "Q1_behavioral_delta" / "validity.json").read_text())
    assert validity == {
        "common_task_count": 1,
        "invalid_for_comparison": False,
        "mean_reward_delta": 0.5,
    }
