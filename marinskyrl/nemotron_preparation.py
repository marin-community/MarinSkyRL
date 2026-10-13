"""Prepare mixed Nemotron rows as portable tasks before rollout execution."""

import argparse
import json
from collections.abc import Iterable, Mapping
from pathlib import Path

import pyarrow.parquet as pq
from datasets import load_dataset
from skyrl_gym.source_task import session_parameters
from taskcompendium.models import EnvironmentRequirements, TaskSpec

from marinskyrl.task_preparation import PARQUET_BATCH_SIZE, source_row_task, write_task_rows
from marinskyrl.task_format import HARBOR_ID_PREFIX, TASK_SPEC_COLUMN, TaskDatasetRow


def terminal_task_index(path: Path) -> dict[str, TaskSpec]:
    """Index Harbor TaskSpecs by package and source instance identities."""
    result = {}
    with pq.ParquetFile(path) as parquet:
        for batch in parquet.iter_batches(batch_size=PARQUET_BATCH_SIZE, columns=[TASK_SPEC_COLUMN]):
            for row in batch.to_pylist():
                record = TaskSpec.model_validate_json(row[TASK_SPEC_COLUMN])
                identifiers = {
                    record.id.casefold(),
                    *(tag.removeprefix(HARBOR_ID_PREFIX) for tag in record.tags if tag.startswith(HARBOR_ID_PREFIX)),
                }
                for identifier in identifiers:
                    if identifier in result and result[identifier] != record:
                        raise ValueError(f"Duplicate terminal-bench task ID {identifier!r}")
                    result[identifier] = record
    return result


def resolve_terminal_task(task: TaskSpec, terminals: Mapping[str, TaskSpec]) -> TaskSpec:
    """Resolve a terminal source row before rollout execution."""
    parameters = session_parameters(task)
    extras = parameters.extras
    ultra = (extras.get("extra_info") or {}).get("nemotron_ultra")
    if ultra is None:
        return task
    if not all(isinstance(ultra.get(key), str) and ultra[key] for key in ("blend", "agent")):
        raise ValueError("Nemotron Ultra tasks require non-empty blend and agent fields")
    if ultra.get("route") != "terminal_bench":
        return task
    identifier = ultra.get("terminal_bench_instance_id")
    if not isinstance(identifier, str) or not identifier:
        raise ValueError("A terminal-bench row must provide terminal_bench_instance_id")
    if identifier.casefold() not in terminals:
        raise ValueError(f"Terminal-bench task {identifier!r} is absent from the configured task data")
    terminal = terminals[identifier.casefold()]
    return terminal.model_copy(update={"id": task.id})


def nemotron_task_rows(
    rows: Iterable[Mapping],
    *,
    source_name: str,
    config: dict,
    environments: Mapping[str, EnvironmentRequirements],
    terminals: Mapping[str, TaskSpec],
) -> Iterable[TaskDatasetRow]:
    for index, row in enumerate(rows):
        ultra = row["extra_info"]["nemotron_ultra"]
        task = source_row_task(
            row, index, source_name=source_name, config=config, environment=environments.get(ultra["agent"])
        )
        yield TaskDatasetRow(
            task=resolve_terminal_task(task, terminals),
            data_source=task.source.dataset,
            teacher_route=row.get("teacher_route"),
            metric_tags=(f"nemotron_ultra/coverage/{ultra['blend']}/{ultra['agent']}",),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--environments", type=Path, help="JSON map from agent names to EnvironmentRequirements")
    parser.add_argument("--terminal-tasks", type=Path, help="Prepared Harbor TaskSpec artifact")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = load_dataset("parquet", data_files=str(args.input), split="train")
    environment_values = {} if args.environments is None else json.loads(args.environments.read_text())
    tasks = nemotron_task_rows(
        rows,
        source_name=args.source,
        config=json.loads(args.config.read_text()),
        environments={key: EnvironmentRequirements.model_validate(value) for key, value in environment_values.items()},
        terminals={} if args.terminal_tasks is None else terminal_task_index(args.terminal_tasks),
    )
    write_task_rows(args.output, tasks)


if __name__ == "__main__":
    main()
