import json
from pathlib import Path

import pytest

from infra.rl_analysis.pipeline import analyze_local_run
from infra.rl_analysis.statistics import MatchedRewardStatistics, matched_reward_statistics
from infra.rl_analysis.traces import TraceRecord, load_trace_records


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


def _write_harbor_trial(
    root: Path,
    trial_name: str,
    *,
    task_name: str | None,
    task_id: dict[str, str],
    reward: float | None,
    prompt_tokens: list[int],
    error_type: str | None = None,
) -> None:
    trial_dir = root / "harbor_jobs" / "job" / trial_name
    trajectory_dir = trial_dir / "agent"
    trajectory_dir.mkdir(parents=True)
    result = {
        "task_name": task_name,
        "task_id": task_id,
        "started_at": "2026-08-12T21:28:45Z",
        "verifier_result": {"rewards": {"reward": reward}},
        "agent_result": {
            "n_input_tokens": sum(prompt_tokens),
            "n_output_tokens": 20,
            "metadata": {"n_episodes": len(prompt_tokens), "summarization_count": 2},
        },
        "exception_info": {"exception_type": error_type} if error_type else None,
    }
    (trial_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
    trajectory = {
        "schema_version": "ATIF-v1.7",
        "steps": [
            {"source": "agent", "metrics": {"prompt_tokens": tokens, "completion_tokens": 10}}
            for tokens in prompt_tokens
        ],
    }
    (trajectory_dir / "trajectory.json").write_text(json.dumps(trajectory), encoding="utf-8")


def test_load_trace_records_joins_current_harbor_trial_artifacts(tmp_path: Path) -> None:
    _write_harbor_trial(
        tmp_path,
        "trial-1",
        task_name="terminal-bench/sanitize-git-repo",
        task_id={"org": "terminal-bench", "name": "sanitize-git-repo", "ref": "sha256:abc"},
        reward=0.0,
        prompt_tokens=[1_024, 65_000, 32_000],
        error_type="AgentTimeoutError",
    )

    [record] = load_trace_records(tmp_path)

    assert record.task_id == "terminal-bench/sanitize-git-repo"
    assert record.reward == 0.0
    assert record.turns == 3
    assert record.peak_prompt_tokens == 65_000
    assert record.cumulative_input_tokens == 98_024
    assert record.summarization_count == 2
    assert record.error_type == "AgentTimeoutError"


def test_load_trace_records_normalizes_structured_harbor_task_id(tmp_path: Path) -> None:
    _write_harbor_trial(
        tmp_path,
        "trial-1",
        task_name=None,
        task_id={"org": "terminal-bench", "name": "task", "ref": "sha256:abc"},
        reward=1.0,
        prompt_tokens=[100],
    )

    [record] = load_trace_records(tmp_path)

    assert record.task_id == "terminal-bench/task@sha256:abc"


def test_load_trace_records_rejects_nonempty_unscored_source(tmp_path: Path) -> None:
    trial_dir = tmp_path / "trial"
    trial_dir.mkdir()
    (trial_dir / "result.json").write_text(json.dumps({"task_name": "task"}), encoding="utf-8")

    with pytest.raises(ValueError, match="No scored trace records"):
        load_trace_records(tmp_path)


def test_matched_reward_statistics_preserves_replicates() -> None:
    before = [
        _record("task-a", 0.0),
        _record("task-a", 1.0),
        _record("task-b", 0.0),
    ]
    after = [
        _record("task-a", 1.0),
        _record("task-a", 1.0),
        _record("task-b", 0.0),
    ]

    assert matched_reward_statistics(before, after) == MatchedRewardStatistics(2, 3, 3, 0.25, 1 / 3)


def _record(task_id: str, reward: float) -> TraceRecord:
    return TraceRecord(task_id, reward, None, 0, None, None)


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
        "baseline_trial_count": 0,
        "common_task_count": 0,
        "invalid_for_comparison": True,
        "mean_reward_delta": None,
        "post_trial_count": 0,
        "task_weighted_mean_reward_delta": None,
        "trial_weighted_mean_reward_delta": None,
    }
    assert temporal["bins"]["2026-07-30T00:00:00+00:00"]["mean_reward"] == 0.5
    assert temporal["bins"]["2026-07-30T04:00:00+00:00"]["mean_turns"] == 4.0
    assert overlay["validity"] == {
        "baseline_trial_count": 0,
        "common_task_count": 0,
        "invalid_for_comparison": True,
        "post_trial_count": 0,
    }
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
        "baseline_trial_count": 1,
        "common_task_count": 1,
        "invalid_for_comparison": False,
        "mean_reward_delta": 0.5,
        "post_trial_count": 1,
        "task_weighted_mean_reward_delta": 0.5,
        "trial_weighted_mean_reward_delta": 0.5,
    }
