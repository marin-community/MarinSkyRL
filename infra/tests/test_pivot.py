"""Local dataset-to-Gym and prediction-file checks for the public Pivot releases."""

import json

import pytest
from datasets import Dataset
from omegaconf import OmegaConf
from skyrl_gym.envs.nemotron_ultra.env import NemotronUltraEnv
from skyrl_gym.envs.nemotron_ultra.pivot import PIVOT_PROFILES, pivot_assistant_message, reference_response
from skyrl_gym.verification import RolloutEvidence

from infra.rl_data.pivot import grade_predictions, read_rows, replay_rows
from infra.rl_data.sources import prepare_pivot_row


def _row(dataset):
    row = {
        "trajectory_id": 7,
        "agent_ref": {"name": PIVOT_PROFILES[dataset].source_agent},
        "responses_create_params": {
            "input": [
                {"role": "user", "content": "Find it"},
                {"type": "function_call", "call_id": "call1", "name": "search", "arguments": '{"q":"old"}'},
                {"type": "function_call_output", "call_id": "call1", "output": "not found"},
                {"role": "user", "content": "Try again"},
            ],
            "tools": [{"type": "function", "name": "search", "parameters": {"type": "object"}}],
            "parallel_tool_calls": True,
        },
        "metadata": {"harness": "terminus_2", "source_trajectory_uid": "terminal-source-7"},
    }
    if dataset == "terminal":
        row["expected_answer"] = json.dumps({"analysis": "state", "plan": "plan", "commands": [{"keystrokes": "pwd\n"}]})
    else:
        row["expected_action"] = {"type": "function_call", "name": "search", "arguments": '{"q":"red blue"}'}
    return row


@pytest.mark.parametrize("dataset", PIVOT_PROFILES)
def test_pivot_preparation_roundtrip_grades_references_and_rejects_missing_actions(dataset, tmp_path):
    raw = _row(dataset)
    prepared = prepare_pivot_row(raw, 0, dataset=dataset)
    path = tmp_path / "rows.parquet"
    Dataset.from_list([prepared]).to_parquet(str(path))
    loaded = Dataset.from_parquet(str(path))[0]
    env = NemotronUltraEnv(OmegaConf.create({}), extras={"extra_info": loaded["extra_info"]})
    prompt, metadata = env.init(loaded["prompt"])
    assert prompt[1]["tool_calls"][0]["id"] == prompt[2]["tool_call_id"] == "call1"
    assert metadata["chat_completion_params"]["tools"] == raw["responses_create_params"]["tools"]
    assert metadata["chat_completion_params"]["parallel_tool_calls"] is True
    assert loaded["extra_info"]["trajectory_id"] == ("terminal-source-7" if dataset == "terminal" else "7")
    env.set_rollout_evidence(RolloutEvidence(metadata={"assistant_message": pivot_assistant_message(reference_response(raw))}))
    result = env.step("")
    assert result["reward"] == 1.0
    assert result["done"] is True
    env.set_rollout_evidence(RolloutEvidence(metadata={"assistant_message": {}}))
    assert env.step("")["reward"] == 0.0
    if dataset != "terminal":
        changed = {"tool_calls": [{"function": {"name": "search", "arguments": '{"q":"green yellow"}'}}]}
        env.set_rollout_evidence(RolloutEvidence(metadata={"assistant_message": changed}))
        assert env.step("")["reward"] == (1.0 if dataset == "swe" else 0.0)


def test_local_prediction_files_report_scores_and_prevent_row_misalignment(tmp_path):
    rows = [_row("function_calling"), _row("function_calling")]
    path = tmp_path / "input.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    loaded = read_rows(path, 2)
    assert replay_rows("function_calling", loaded)["passed"] == 2
    output = tmp_path / "grades.jsonl"
    predictions = [{"index": 0, "response": reference_response(rows[0])}, {"index": 1, "response": {}}]
    summary = grade_predictions("function_calling", loaded, predictions, output)
    assert summary["mean_reward"] == 0.5
    assert [row["reward"] for row in read_rows(output, 2)] == [1.0, 0.0]
    with pytest.raises(ValueError, match="consecutive"):
        grade_predictions("function_calling", loaded, predictions[::-1], output)
    with pytest.raises(ValueError, match="one entry"):
        grade_predictions("function_calling", loaded, predictions[:1], output)
