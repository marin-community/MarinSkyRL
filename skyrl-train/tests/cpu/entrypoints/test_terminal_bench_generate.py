from pathlib import Path

from omegaconf import OmegaConf

from skyrl_train.entrypoints.terminal_bench_generate import TerminalBenchGenerateExp


def test_terminal_bench_generate_reads_validation_tasks(tmp_path: Path) -> None:
    task_dir = tmp_path / "tasks" / "task-a"
    task_dir.mkdir(parents=True)
    (task_dir / "instruction.md").write_text("Do the task")

    experiment = object.__new__(TerminalBenchGenerateExp)
    experiment.cfg = OmegaConf.create(
        {
            "trainer": {"eval_interval": 1},
            "data": {"train_data": [], "val_data": [str(tmp_path / "tasks")]},
        }
    )

    assert experiment.get_train_dataset() is None
    assert list(experiment.get_eval_dataset()) == [
        {
            "prompt": str(task_dir),
            "env_class": None,
            "env_extras": {"data_source": str(task_dir)},
            "uid": "task-a",
        }
    ]
