from __future__ import annotations

import gzip
import io
import tarfile
from dataclasses import asdict, replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from marinskyrl.packed_tasks import (
    EmptyTaskSelectionError,
    PackedTaskArchiveError,
    PackedTaskMaterializer,
    select_task_references,
)
from marinskyrl.task_sources import (
    TaskTroveParquetSource,
    TaskTroveSelection,
    TaskTroveSelectionSnapshot,
    TaskTroveTagMatch,
)
from skyrl_train.trajectory_runners.harbor.dataset import TerminalBenchTaskDataset


def _task_binary(name: str, *, solution: bool = False, unsafe_path: bool = False) -> bytes:
    files = {
        "instruction.md": f"Do {name}".encode(),
        "task.toml": b"[environment]\n",
        "environment/Dockerfile": b"FROM python:3.12-slim\n",
        "tests/test.sh": b"#!/bin/sh\nexit 0\n",
    }
    if solution:
        files["solution/solve.sh"] = b"#!/bin/sh\n"
    if unsafe_path:
        files["../escape"] = b"escaped"
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for path, content in files.items():
            info = tarfile.TarInfo(path)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return gzip.compress(buffer.getvalue(), mtime=0)


def _write_dataset(path: Path, *, include_solution: bool = False, include_unsafe_path: bool = False) -> None:
    schema = pa.schema(
        [
            ("source", pa.string()),
            ("tags", pa.list_(pa.string())),
            ("mode", pa.string()),
            ("path", pa.string()),
            ("dockerfile_id", pa.string()),
            ("task_binary", pa.binary()),
        ]
    )
    row_groups = [
        [
            {
                "source": "source-a",
                "tags": ["bash", "terminal"],
                "mode": "script",
                "path": "task-1",
                "dockerfile_id": "env-a",
                "task_binary": _task_binary("one"),
            },
            {
                "source": "source-b",
                "tags": ["judge", "qa"],
                "mode": "judge",
                "path": "task-2",
                "dockerfile_id": "env-b",
                "task_binary": _task_binary("two"),
            },
        ],
        [
            {
                "source": "source-a",
                "tags": ["bash"],
                "mode": "script",
                "path": "task-3",
                "dockerfile_id": "env-a",
                "task_binary": _task_binary("three"),
            },
            {
                "source": "source-c",
                "tags": ["code", "terminal"],
                "mode": "pytest",
                "path": "task-4",
                "dockerfile_id": "env-c",
                "task_binary": _task_binary("four", solution=include_solution, unsafe_path=include_unsafe_path),
            },
        ],
    ]
    with pq.ParquetWriter(path, schema) as writer:
        for rows in row_groups:
            writer.write_table(pa.Table.from_pylist(rows, schema=schema))


def _source(path: Path, selection: TaskTroveSelection) -> TaskTroveParquetSource:
    return TaskTroveParquetSource(
        uri=path.as_uri(),
        identity="tasktrove/clean@fixture:abc123",
        local_path=str(path.parent),
        relative_path=path.name,
        verifier_ref="tasktrove-verify@fixture",
        selection=selection,
    )


def test_tasktrove_selection_combines_source_tags_and_modes(tmp_path: Path) -> None:
    dataset_path = tmp_path / "tasks.parquet"
    _write_dataset(dataset_path)
    source = _source(
        dataset_path,
        TaskTroveSelection(sources=("source-a",), tags=("bash",), modes=("script",)),
    )

    summary = select_task_references(source)

    assert [(reference.row_group, reference.path) for reference in summary.references] == [
        (0, "task-1"),
        (1, "task-3"),
    ]
    assert summary.distinct_environment_count == 1


def test_tasktrove_any_tags_and_limit_have_stable_nested_membership(tmp_path: Path) -> None:
    dataset_path = tmp_path / "tasks.parquet"
    _write_dataset(dataset_path)
    selection = TaskTroveSelection(
        tags=("bash", "terminal"),
        tag_match=TaskTroveTagMatch.ANY,
        limit=2,
        seed=17,
    )

    two = select_task_references(_source(dataset_path, selection))
    three = select_task_references(_source(dataset_path, TaskTroveSelection(**{**asdict(selection), "limit": 3})))

    assert {reference.uid() for reference in two.references} < {reference.uid() for reference in three.references}


def test_tasktrove_selection_rejects_unknown_source(tmp_path: Path) -> None:
    dataset_path = tmp_path / "tasks.parquet"
    _write_dataset(dataset_path)

    with pytest.raises(EmptyTaskSelectionError, match="missing-source"):
        select_task_references(_source(dataset_path, TaskTroveSelection(sources=("missing-source",))))


def test_packed_dataset_defers_extraction_until_materialization(tmp_path: Path) -> None:
    dataset_path = tmp_path / "tasks.parquet"
    _write_dataset(dataset_path)
    source = _source(dataset_path, TaskTroveSelection(sources=("source-a",)))
    dataset = TerminalBenchTaskDataset([asdict(source)])
    cache = tmp_path / "cache"

    first = dataset[0]
    reference = next(iter(select_task_references(source).references))

    assert first["prompt"].startswith("tasktrove://")
    task_path = PackedTaskMaterializer(cache).materialize_batch([reference])[reference]
    assert (task_path / "instruction.md").read_text() == "Do one"
    assert not (task_path / "solution").exists()


def test_packed_dataset_rejects_changed_launch_selection(tmp_path: Path) -> None:
    dataset_path = tmp_path / "tasks.parquet"
    _write_dataset(dataset_path)
    source = _source(dataset_path, TaskTroveSelection(sources=("source-a",)))
    summary = select_task_references(source)
    changed = replace(
        source,
        snapshot=TaskTroveSelectionSnapshot(
            count=len(summary.references),
            digest="changed",
            distinct_environment_count=summary.distinct_environment_count,
        ),
    )

    with pytest.raises(ValueError, match="digest changed"):
        TerminalBenchTaskDataset([asdict(changed)])


def test_packed_materializer_rejects_solution_files(tmp_path: Path) -> None:
    dataset_path = tmp_path / "tasks.parquet"
    _write_dataset(dataset_path, include_solution=True)
    source = _source(dataset_path, TaskTroveSelection(sources=("source-c",)))
    reference = select_task_references(source).references[0]

    with pytest.raises(PackedTaskArchiveError, match="solution"):
        PackedTaskMaterializer(tmp_path / "cache").materialize_batch([reference])


def test_packed_materializer_rejects_parent_traversal(tmp_path: Path) -> None:
    dataset_path = tmp_path / "tasks.parquet"
    _write_dataset(dataset_path, include_unsafe_path=True)
    source = _source(dataset_path, TaskTroveSelection(sources=("source-c",)))
    reference = select_task_references(source).references[0]

    with pytest.raises(PackedTaskArchiveError, match="Unsafe task archive path"):
        PackedTaskMaterializer(tmp_path / "cache").materialize_batch([reference])
