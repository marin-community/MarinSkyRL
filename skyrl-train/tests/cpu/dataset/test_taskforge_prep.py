"""Taskforge rows prepared for local machines on the rollout worker's host."""

import json
from pathlib import Path

import polars as pl
import pytest
from rolloutengine.models import TaskRolloutSpec
from shellbox.machine import Backend, PackageSpec
from taskcompendium.models import CommandSemantics, TaskSpec

from skyrl_train.dataset.taskforge import (
    GRADER_BASE_IMAGE,
    SessionSettings,
    prepare_tasks,
    write_preparation,
)
from skyrl_train.dataset.tasks import TASK_ROLLOUT_COLUMN

SETTINGS = SessionSettings()


def _environment(semantics: str | None, *, image: bool, workdir: str | None = None) -> dict:
    return {
        "capabilities": ["shell", "filesystem"] if semantics else [],
        "command_semantics": semantics,
        "docker_image": GRADER_BASE_IMAGE if image else None,
        "working_directory": workdir,
    }


def _task(
    index: int, task_environment: dict, *, worker_path: str = "workspace/data.txt", timeout: float = 120.0
) -> TaskSpec:
    return TaskSpec.model_validate(
        {
            "id": f"task-{index}",
            "context": {"events": [{"type": "message", "role": "user", "content": "Compute it."}]},
            "environment_requirements": task_environment,
            "answer_type": "text",
            "answer_format": {"kind": "plain_text"},
            "grader": {
                "kind": "script",
                "argv": ["python3", "/tests/grade.py"],
                "cwd": "/app",
                "environment": _environment("linux_process", image=True),
                "reward": {
                    "kind": "file",
                    "files": [{"path": "/logs/verifier/reward.json", "format": "json", "key": "reward"}],
                },
                "timeout": timeout,
            },
            "source": {
                "dataset": "open-athena/taskforge",
                "revision": "1",
                "row": str(index),
                "importer_revision": "1",
            },
            "resources": {
                "worker": [{"path": worker_path, "source": {"kind": "inline_file", "content_base64": "MQ=="}}],
                "verifier": [{"path": "grade.py", "source": {"kind": "inline_file", "content_base64": "cGFzcw=="}}],
            },
            "tags": [
                f"taskforge:d01.algebra.cap-{index}:testgen20261013/{index}",
                f"capability:d01.algebra.cap-{index}",
                "subject:d01",
                "band:calibrated",
            ],
        }
    )


def test_python_image_machines_run_locally_with_a_full_solve_pass_threshold():
    task = _task(1, _environment("linux_process", image=True, workdir="/workspace"))

    (row,) = prepare_tasks([task], exclude=set(), labels={}, settings=SETTINGS).rows

    lowered = row.spec
    assert lowered.task.environment_requirements.docker_image is None
    assert lowered.task.environment_requirements.package_spec == PackageSpec(pypi_packages=())
    assert lowered.task.environment_requirements.command_semantics == CommandSemantics.LINUX_PROCESS
    assert lowered.task.grader.environment.package_spec == PackageSpec(pypi_packages=())
    assert lowered.task.grader.reward.pass_above == SETTINGS.pass_above
    assert lowered.task_machine.backend == lowered.verifier_machine.backend == Backend.LOCAL.value
    assert lowered.task_machine.memory_mb is None and lowered.task_machine.cpus is None
    assert lowered.limits.verifier_timeout == 120.0 + SETTINGS.verifier_margin
    assert lowered.limits.max_total_tokens == SETTINGS.max_total_tokens
    assert lowered.task.source.dataset == "d01"
    assert row.item_id == "d01.algebra.cap-1--1"


def test_shellsim_and_machine_free_tasks_keep_their_task_machine_and_grade_locally():
    shellsim = _task(1, _environment("shell_simulator", image=False, workdir="/workspace"))
    text_only = _task(2, _environment(None, image=False), worker_path="data.txt")

    rows = prepare_tasks([shellsim, text_only], exclude=set(), labels={}, settings=SETTINGS).rows

    assert [row.spec.task_machine.backend if row.spec.task_machine else None for row in rows] == ["shellsim", None]
    assert [row.spec.verifier_machine.backend for row in rows] == ["local", "local"]


def test_exclusions_labels_and_unhostable_rows_are_accounted_for():
    tasks = [
        _task(1, _environment("linux_process", image=True)),
        _task(2, _environment("linux_process", image=True)),
        _task(3, _environment("linux_process", image=True), worker_path="opt/tool/config.txt"),
        _task(4, _environment("linux_process", image=True)),
    ]

    preparation = prepare_tasks(
        tasks,
        exclude={"d01.algebra.cap-2--2", "d09.missing--1"},
        labels={"d01.algebra.cap-4--4": "heldout_incap"},
        settings=SETTINGS,
    )

    assert [row.item_id for row in preparation.rows] == ["d01.algebra.cap-1--1", "d01.algebra.cap-4--4"]
    assert [row.spec.task.source.dataset for row in preparation.rows] == ["d01", "heldout_incap"]
    assert preparation.excluded == ["d01.algebra.cap-2--2"]
    assert preparation.exclusions_not_found == ["d09.missing--1"]
    assert list(preparation.dropped) == ["d01.algebra.cap-3--3"]


def test_written_pool_round_trips_and_splits_by_label(tmp_path: Path):
    tasks = [_task(index, _environment("linux_process", image=True)) for index in (1, 2, 3)]
    preparation = prepare_tasks(tasks, exclude=set(), labels={"d01.algebra.cap-3--3": "train"}, settings=SETTINGS)

    manifest = write_preparation(preparation, tmp_path, SETTINGS, ["input.parquet"])

    assert manifest["files"] == {"pool.parquet": 3, "d01.parquet": 2, "train.parquet": 1}
    assert manifest["by_task_machine"] == [{"task_machine": "local", "len": 3}]
    pool = pl.read_parquet(tmp_path / "pool.parquet")
    assert pool.columns == [TASK_ROLLOUT_COLUMN]
    specs = [TaskRolloutSpec.model_validate_json(value) for value in pool[TASK_ROLLOUT_COLUMN]]
    assert [spec.task.id for spec in specs] == ["task-1", "task-2", "task-3"]
    index = pl.read_parquet(tmp_path / "index.parquet")
    assert index.filter(pl.col("label") == "train")["item_id"].to_list() == ["d01.algebra.cap-3--3"]
    assert json.loads((tmp_path / "manifest.json").read_text())["rows"] == 3


def test_labels_for_items_outside_the_input_are_refused():
    with pytest.raises(ValueError, match="not in the input"):
        prepare_tasks(
            [_task(1, _environment("linux_process", image=True))],
            exclude=set(),
            labels={"d01.algebra.cap-9--9": "train"},
            settings=SETTINGS,
        )
