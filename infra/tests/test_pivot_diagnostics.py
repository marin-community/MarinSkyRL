"""Terminal and SWE diagnostic score names can coexist in trainer reductions."""

from types import SimpleNamespace

from marinskyrl.pivot_pilot import diagnostic_metrics


def test_terminal_metrics_reduce_reported_scores_in_eight_sample_groups():
    grades = [
        {"scores": {"exact_commands": int(i == 0), "string_90": int(i < 2), "schema_completion": int(i < 4)}}
        for i in range(8)
    ]
    batch = {
        "verification_results": [SimpleNamespace(diagnostics={"pivot": grade}) for grade in grades],
        "trajectory_ids": [SimpleNamespace(instance_id="terminal-row") for _ in grades],
    }
    result = diagnostic_metrics(batch, prefix="train/pivot")
    assert result["train/pivot/schema_completion/accuracy"] == 0.5
    assert result["train/pivot/string_90/accuracy"] == 0.25
    assert result["train/pivot/exact_commands/accuracy"] == 0.125
    assert result["train/pivot/schema_completion/mixed_group_fraction"] == 1
    assert "train/pivot/tool_name/accuracy" not in result


def test_mixed_domains_use_only_rows_with_each_verifier():
    grades = [
        {"scores": {"tool_name": 1, "nemo": 0, "exact": 0}, "extra_tool_calls": 0},
        {"scores": {"schema_completion": 1, "string_90": 1, "exact_commands": 0}},
    ]
    batch = {
        "verification_results": [SimpleNamespace(diagnostics={"pivot": grade}) for grade in grades],
        "trajectory_ids": [SimpleNamespace(instance_id=str(i)) for i in range(2)],
    }
    result = diagnostic_metrics(batch, prefix="eval/pivot")
    assert result["eval/pivot/tool_name/accuracy"] == 1
    assert result["eval/pivot/schema_completion/accuracy"] == 1
    assert result["eval/pivot/exact_commands/accuracy"] == 0
    assert result["eval/pivot/extra_tool_calls"] == 0
    assert result["eval/pivot/graded_fraction"] == 1
