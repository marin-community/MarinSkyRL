"""Per-task rewards from evaluation dumps."""

from skyrl_train.entrypoints.taskcompendium_generate import per_task_summary


def _entry(uid: str, score: float | None, passed: bool | None, status: str = "graded") -> dict:
    verification = None if status == "none" else {"status": status, "score": score, "passed": passed}
    return {"uid": uid, "data_source": "heldout", "verification": verification, "stop_reason": "stop"}


def test_per_task_summary_averages_scored_rollouts_and_counts_unscored_ones():
    summary = per_task_summary(
        [
            _entry("task-a", 1.0, True),
            _entry("task-a", 0.5, False),
            _entry("task-a", None, None, status="infra_error"),
            _entry("task-a", None, None, status="none"),
            _entry("task-b", 0.0, False),
        ]
    )

    assert [record["task_id"] for record in summary] == ["task-a", "task-b"]
    task_a = summary[0]
    assert (task_a["rollouts"], task_a["scored"], task_a["unscored"]) == (4, 2, 2)
    assert task_a["mean_reward"] == 0.75
    assert task_a["pass_rate"] == 0.5
    assert summary[1]["mean_reward"] == 0.0 and summary[1]["pass_rate"] == 0.0
