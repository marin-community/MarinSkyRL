"""Prepare source data as bare TaskSpecs before a training run."""

import argparse
import hashlib
import json
from collections.abc import Iterable, Mapping
from itertools import batched
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from skyrl_gym.source_task import source_task
from taskcompendium.models import EnvironmentRequirements, Source, TaskSpec
from marinskyrl.task_format import TASK_SPEC_COLUMN, CurriculumBin, TaskDatasetRow

TASK_SCHEMA = pa.schema(
    [
        pa.field(TASK_SPEC_COLUMN, pa.string(), nullable=False),
        pa.field("data_source", pa.string()),
        pa.field("teacher_route", pa.string()),
        pa.field("metric_tags", pa.list_(pa.string()), nullable=False),
        pa.field("extra_info", pa.struct([pa.field("data_source", pa.string()), pa.field("grade", pa.int64())])),
    ]
)
PARQUET_BATCH_SIZE = 1024


def source_row_task(
    row: Mapping[str, Any],
    index: int,
    *,
    source_name: str,
    config: Mapping[str, Any],
    environment: EnvironmentRequirements | None = None,
) -> TaskSpec:
    """Keep source prompts and private grader inputs without runtime settings."""
    content = json.dumps(dict(row), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    source = Source(
        dataset=row.get("data_source") or source_name,
        revision=f"sha256:{hashlib.sha256(content).hexdigest()}",
        row=str(index),
        importer_revision="skyrl-task-spec-v1",
    )
    extras = {
        key: value
        for key, value in row.items()
        if key not in {"prompt", "env_class", "teacher_route", "metric_tags", "data_source"}
    }
    return source_task(row["prompt"], extras, dict(config), source, environment=environment)


def write_tasks(path: Path, tasks: Iterable[TaskSpec]) -> None:
    """Write prepared TaskSpecs as a private input artifact."""
    write_task_rows(path, (TaskDatasetRow(task) for task in tasks))


def write_task_rows(path: Path, rows: Iterable[TaskDatasetRow]) -> None:
    """Keep task definitions and training labels in separate storage columns."""
    with pq.ParquetWriter(path, TASK_SCHEMA) as writer:
        path.chmod(0o600)
        for batch in batched(rows, PARQUET_BATCH_SIZE):
            writer.write_table(
                pa.Table.from_pylist(
                    [
                        {
                            TASK_SPEC_COLUMN: row.task.model_dump_json(),
                            "data_source": row.data_source,
                            "teacher_route": row.teacher_route,
                            "metric_tags": list(row.metric_tags),
                            "extra_info": None
                            if row.curriculum_bin is None
                            else {
                                "data_source": row.curriculum_bin.name,
                                "grade": row.curriculum_bin.grade,
                            },
                        }
                        for row in batch
                    ],
                    schema=TASK_SCHEMA,
                )
            )


def cache_tasks(tasks: Iterable[TaskSpec], cache_dir: Path) -> Path:
    """Save a prepared task artifact under its content digest."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=cache_dir) as directory:
        temporary_path = Path(directory) / "tasks.parquet"
        write_tasks(temporary_path, tasks)
        with temporary_path.open("rb") as contents:
            digest = hashlib.file_digest(contents, "sha256").hexdigest()
        path = cache_dir / f"{digest}.parquet"
        temporary_path.replace(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--config", type=Path, help="Private grader parameters in JSON")
    parser.add_argument("--environment", type=Path, help="Task EnvironmentRequirements in JSON")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = {} if args.config is None else json.loads(args.config.read_text())
    environment = (
        None if args.environment is None else EnvironmentRequirements.model_validate_json(args.environment.read_text())
    )
    format_name = "parquet" if args.input.suffix == ".parquet" else "json"
    rows = load_dataset(format_name, data_files=str(args.input), split="train")
    source_sessions = {}
    for row in rows:
        session_name = row.get("env_class")
        if session_name is None:
            continue
        source_name = row.get("data_source") or args.source
        if source_sessions.setdefault(source_name, session_name) != session_name:
            raise NotImplementedError(
                f"Source {source_name!r} contains different session types. Give each session type a separate source."
            )
    write_task_rows(
        args.output,
        (
            TaskDatasetRow(
                source_row_task(row, index, source_name=args.source, config=config, environment=environment),
                teacher_route=row.get("teacher_route"),
                data_source=row.get("data_source"),
                metric_tags=tuple(row.get("metric_tags") or ()),
                curriculum_bin=curriculum_bin(row),
            )
            for index, row in enumerate(rows)
        ),
    )


def curriculum_bin(row: Mapping[str, Any]) -> CurriculumBin | None:
    info = row.get("extra_info")
    if info is None or "data_source" not in info or "grade" not in info:
        return None
    return CurriculumBin(name=info["data_source"], grade=info["grade"])


if __name__ == "__main__":
    main()
