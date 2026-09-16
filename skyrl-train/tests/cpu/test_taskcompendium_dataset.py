import json

from skyrl_train.trajectory_runners.taskcompendium import TaskCompendiumTaskDataset


def _lowering(root, name, agent):
    task = root / name
    task.mkdir()
    (task / "manifest.json").write_text("{}")
    (task / "reference-execution.json").write_text(json.dumps({"agent": agent, "environment": {}, "verifier": {}}))
    return task


def test_taskcompendium_dataset_binds_staged_tasks_to_live_policy(tmp_path):
    _lowering(
        tmp_path,
        "chat",
        {"import_path": "taskcompendium.harbor.agents:DirectChatAgent", "kwargs": {"api_base": "stale"}},
    )
    _lowering(
        tmp_path,
        "tools",
        {"import_path": "taskcompendium.harbor.agents:ShellToolAgent", "kwargs": {"max_turns": 2}},
    )

    dataset = TaskCompendiumTaskDataset(
        [str(tmp_path)],
        api_base="http://policy:8000/v1",
        model_name="snowball",
    )

    assert [row["uid"] for row in dataset] == ["chat", "tools"]
    for row in dataset:
        execution = row["env_extras"]["execution"]
        assert execution["agent"]["kwargs"]["api_base"] == "http://policy:8000/v1"
        assert execution["agent"]["model_name"] == "snowball"
        assert row["env_extras"]["task_dir"] in row["prompt"][0]["content"]
