from pathlib import Path

import pytest

from skyrl_train.trajectory_runners.harbor.dataset import TerminalBenchTaskDataset


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

    assert dataset.get_task_paths() == sorted(expected)
