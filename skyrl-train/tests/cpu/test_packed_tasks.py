from __future__ import annotations

import gzip
import io
import shutil
import tarfile
from dataclasses import asdict, replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from omegaconf import OmegaConf

from cloud.iris import rl_data
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import ShellSimBuiltins

from marinskyrl.packed_tasks import (
    EmptyTaskSelectionError,
    PackedTaskArchiveError,
    PackedTaskMaterializer,
    PackedTaskReference,
    select_task_references,
)
from marinskyrl.task_sources import (
    TaskTroveParquetSource,
    TaskTroveSelection,
    TaskTroveSelectionSnapshot,
    TaskTroveTagMatch,
)
from skyrl_train.dataset.harbor import TerminalBenchTaskDataset, materialize_harbor_tasks
from taskcompendium.environment import DockerBuild, EnvironmentKind, ShellVerifierSpec
from taskcompendium.grading import Outcome
from taskcompendium.models import SkippedVerifierSpec, VerifierKind, VerifierSpec
from taskcompendium.parquet import read_tasks
from rolloutengine.contracts import ModelTurn
from rolloutengine.engine import ShellboxRolloutEngine
from taskcompendium.submission import AnswerFormat, SubmissionConvention


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
    return _archive(files)


def _archive(files: dict[str, bytes]) -> bytes:
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


def test_packed_dataset_materializes_reference_from_runtime_yaml(tmp_path: Path) -> None:
    dataset_path = tmp_path / "tasks.parquet"
    _write_dataset(dataset_path)
    source = _source(dataset_path, TaskTroveSelection(sources=("source-a",)))
    resolved = rl_data.resolve_rl_train_data_with_sources([asdict(source)], kind="tasks", verbose=False)
    config_path = tmp_path / "runtime-config.yaml"
    OmegaConf.save(OmegaConf.create({"data": {"train_data": list(resolved.paths)}}), config_path)
    document = OmegaConf.load(config_path)
    dataset = TerminalBenchTaskDataset(OmegaConf.to_container(document, resolve=True)["data"]["train_data"])
    cache = tmp_path / "cache"

    first = dataset[0]
    assert isinstance(first, PackedTaskReference)

    assert first.stable_uri().startswith("tasktrove://")
    task_path = PackedTaskMaterializer(cache).materialize_batch([first])[first]
    assert (task_path / "instruction.md").read_text() == "Do one"
    assert not (task_path / "solution").exists()


def test_packed_selection_becomes_portable_task_parquet(tmp_path: Path) -> None:
    dataset_path = tmp_path / "source.parquet"
    _write_dataset(dataset_path)
    source = _source(dataset_path, TaskTroveSelection(sources=("source-a",)))
    output = materialize_harbor_tasks([asdict(source)], cache_dir=tmp_path / "cache")
    dataset_path.unlink()
    tasks = list(read_tasks(str(output)))
    assert [task.context.events[0].content for task in tasks] == ["Do one", "Do three"]
    assert [task.id for task in tasks] == [
        "tasktrove/clean@fixture:abc123/source-a/task-1",
        "tasktrove/clean@fixture:abc123/source-a/task-3",
    ]
    assert output.stat().st_mode & 0o777 == 0o600
    for task in tasks:
        assert isinstance(task.environment.image, DockerBuild)
        assert {file.path: file.content for file in task.environment.image.files} == {
            "/Dockerfile": b"FROM python:3.12-slim\n"
        }
        verifier = ShellVerifierSpec.model_validate_json(task.verifier.parameters_json)
        assert {file.path: file.content for file in verifier.files} == {"/tests/test.sh": b"#!/bin/sh\nexit 0\n"}


@pytest.mark.asyncio
@pytest.mark.parametrize("staged", [False, True])
@pytest.mark.parametrize("registry_image", [False, True])
@pytest.mark.parametrize("verification", [True, False])
async def test_packed_tasks_execute_after_source_removal(tmp_path, staged, registry_image, verification):
    config = '[environment]\nworkdir = "/workspace"\nallow_internet = false\n'
    files = {"setup_files/input": b"first\n"}
    if registry_image:
        config += 'docker_image = "fixture"\n'
    else:
        files["environment/Dockerfile"] = b"FROM busybox\n"
    grader = b'test "$(cat /workspace/state)" = first && echo 1 > /logs/verifier/reward.txt\n'
    if staged:
        config += '[[steps]]\nname = "first"\nmin_reward = 1\n[[steps]]\nname = "second"\n'
        files.update(
            {
                "steps/first/instruction.md": b"First stage.",
                "steps/first/workdir/setup.sh": b"cp /setup_files/input /workspace/state\n",
                "steps/first/tests/test.sh": grader,
                "steps/second/instruction.md": b"Second stage.",
                "steps/second/workdir/setup.sh": (
                    b'test "$(cat /workspace/state)" = first && echo second > /workspace/state\n'
                ),
                "steps/second/tests/test.sh": grader.replace(b"= first", b"= second"),
            }
        )
    else:
        files["instruction.md"] = b"Single stage."
        files["tests/test.sh"] = grader.replace(b"/workspace/state", b"/setup_files/input")
    files["task.toml"] = config.encode()
    if not verification:
        files = {path: data for path, data in files.items() if "tests" not in Path(path).parts}
    dataset_path = tmp_path / "source.parquet"
    _write_dataset(dataset_path)
    table = pq.read_table(dataset_path)
    payloads = pa.array([_archive(files)] * len(table), type=pa.binary())
    pq.write_table(table.set_column(table.schema.get_field_index("task_binary"), "task_binary", payloads), dataset_path)
    source = _source(dataset_path, TaskTroveSelection(sources=("source-a",), limit=1))
    cache = tmp_path / "cache"
    output = materialize_harbor_tasks(
        [asdict(source)],
        cache_dir=cache,
        verifier_override=None
        if verification
        else VerifierSpec(
            kind=VerifierKind.SKIPPED,
            parameters_json=SkippedVerifierSpec(reason="Verification is disabled").model_dump_json(),
        ),
    )
    dataset_path.unlink()
    shutil.rmtree(cache / "archives")

    class Factory:
        async def create(self, spec):
            return await ShellSimMachineFactory().create(replace(spec, source=ShellSimBuiltins()))

    class Model:
        async def complete(self, request):
            prompt = (*request.prefix_token_ids, 90) if request.prefix_token_ids else (10,)
            return ModelTurn({"role": "assistant", "content": "Done."}, prompt, (20,), (-0.5,), "stop")

    engine = ShellboxRolloutEngine(
        Model().complete,
        {EnvironmentKind.DOCKER: Factory()},
        max_turns=1,
        command_timeout=5,
        convention=SubmissionConvention(id="plain", answer_format=AnswerFormat.PLAIN),
    )
    result = await engine.run(next(read_tasks(str(output))))
    assert (result.grade.status, result.grade.reward) == (
        (Outcome.GRADED, 1.0) if verification else (Outcome.SKIPPED, None)
    )
    assert result.response_token_ids == ((20, 90, 20) if staged else (20,))
    assert result.loss_mask == ((1, 0, 1) if staged else (1,))
    assert result.logprobs == ((-0.5, 0.0, -0.5) if staged else (-0.5,))


def test_packed_materializer_reuses_reader_across_batches(tmp_path: Path) -> None:
    dataset_path = tmp_path / "tasks.parquet"
    _write_dataset(dataset_path)
    references = select_task_references(_source(dataset_path, TaskTroveSelection(sources=("source-a",)))).references
    materializer = PackedTaskMaterializer(tmp_path / "cache")

    first = materializer.materialize_batch([references[0]])[references[0]]
    dataset_path.unlink()
    second = materializer.materialize_batch([references[1]])[references[1]]
    materializer.close()

    assert (first / "instruction.md").read_text() == "Do one"
    assert (second / "instruction.md").read_text() == "Do three"


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
