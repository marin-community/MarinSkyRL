from pathlib import Path
import json

import pytest
from taskcompendium.importers.skyrl import gym_task
from taskcompendium.models import Source

from skyrl_train.dataset.harbor import TerminalBenchTaskDataset, materialize_harbor_tasks
from skyrl_train.dataset.nemotron_ultra import resolve_terminal_task, terminal_task_index


def _write_task(root: Path, name: str) -> Path:
    task = root / name
    task.mkdir()
    (task / "instruction.md").write_text(name)
    return task


def test_terminal_bench_dataset_orders_tasks_by_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    expected = [_write_task(tmp_path, name) for name in ("task-c", "task-a", "task-b")]
    original_iterdir = Path.iterdir

    def reverse_task_listing(path: Path):
        children = list(original_iterdir(path))
        return iter(reversed(children)) if path == tmp_path else iter(children)

    monkeypatch.setattr(Path, "iterdir", reverse_task_listing)

    dataset = TerminalBenchTaskDataset([str(tmp_path)])

    assert list(dataset) == sorted(expected)


@pytest.mark.parametrize("selection", ["missing", "ambiguous"])
def test_terminal_task_selection_fails_before_execution(tmp_path, selection):
    sources = tmp_path / "sources"
    sources.mkdir()
    for name in ("first", "second"):
        task = _write_task(sources, name)
        (task / "tests").mkdir()
        (task / "tests/config.json").write_text(json.dumps({"instance_id": "same-id"}))
        (task / "tests/test.sh").write_text("echo 1 > /logs/verifier/reward.txt\n")
        (task / "task.toml").write_text('[environment]\ndocker_image = "busybox"\n')
    path = materialize_harbor_tasks(
        [str(sources if selection == "ambiguous" else sources / "first")], cache_dir=tmp_path / "cache"
    )
    if selection == "ambiguous":
        with pytest.raises(ValueError, match="Duplicate terminal-bench task ID"):
            terminal_task_index(path)
        return
    task = gym_task(
        [{"role": "user", "content": "Repair the task."}],
        "nemotron_ultra",
        {
            "extra_info": {
                "nemotron_ultra": {
                    "blend": "rlvr1",
                    "agent": "swe",
                    "route": "terminal_bench",
                    "terminal_bench_instance_id": "absent",
                }
            }
        },
        {},
        Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
    )
    with pytest.raises(ValueError, match="absent from the configured task data"):
        resolve_terminal_task(task, terminal_task_index(path))
