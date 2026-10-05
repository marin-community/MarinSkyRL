"""Prepare mixed Nemotron rows as portable tasks before rollout execution."""

from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from datasets import Dataset
from taskcompendium.environment import ExternalVerifierSpec
from taskcompendium.execution import TaskExecution
from taskcompendium.models import TaskSpec, VerifierSpec
from transformers import PreTrainedTokenizerBase

from skyrl_train.dataset.harbor import materialize_harbor_tasks
from skyrl_train.dataset.tasks import PARQUET_BATCH_SIZE, SourceTaskDataset, TaskRecord, source_row_task, task_prompt


def terminal_task_index(path: Path) -> dict[str, TaskRecord]:
    """Index portable Harbor tasks by their directory and source instance IDs."""
    result = {}
    with pq.ParquetFile(path) as parquet:
        columns = [name for name in ("task_spec", "task_execution") if name in parquet.schema_arrow.names]
        for batch in parquet.iter_batches(batch_size=PARQUET_BATCH_SIZE, columns=columns):
            for row in batch.to_pylist():
                task = TaskSpec.model_validate_json(row["task_spec"])
                execution = (
                    TaskExecution()
                    if row.get("task_execution") is None
                    else TaskExecution.model_validate_json(row["task_execution"])
                )
                record = TaskRecord(task, execution)
                for identifier in {task.id.casefold(), *task.metadata["harbor_task_ids"]}:
                    if identifier in result and result[identifier] != record:
                        raise ValueError(f"Duplicate terminal-bench task ID {identifier!r}")
                    result[identifier] = record
    return result


def resolve_terminal_task(task: TaskSpec, terminals: Mapping[str, TaskRecord]) -> TaskRecord:
    """Keep source metadata while replacing a terminal row with its executable task."""
    verifier = ExternalVerifierSpec.model_validate_json(task.verifier.parameters_json)
    extras = verifier.parameters["extras"]
    ultra = (extras.get("extra_info") or {}).get("nemotron_ultra")
    if ultra is None:
        return TaskRecord(task, TaskExecution())
    if not all(isinstance(ultra.get(key), str) and ultra[key] for key in ("blend", "agent")):
        raise ValueError("Nemotron Ultra tasks require non-empty blend and agent fields")
    if ultra.get("route") != "terminal_bench":
        return TaskRecord(task, TaskExecution())
    identifier = ultra.get("terminal_bench_instance_id")
    if not isinstance(identifier, str) or not identifier:
        raise ValueError("A terminal-bench row must provide terminal_bench_instance_id")
    if identifier.casefold() not in terminals:
        raise ValueError(f"Terminal-bench task {identifier!r} is absent from the configured task data")
    terminal = terminals[identifier.casefold()]
    return TaskRecord(
        terminal.task.model_copy(
            update={
                "id": task.id,
                "source": task.source,
                "metadata": {
                    **terminal.task.metadata,
                    **task.metadata,
                    "skyrl_extras": extras,
                    "harbor_source": terminal.task.source.model_dump(mode="json"),
                },
            }
        ),
        terminal.execution,
    )


def _nemotron_task_prompt(
    row: dict,
    index: int,
    *,
    source_name: str,
    environment_configs: Mapping[str, dict],
    terminals: Mapping[str, TaskRecord],
) -> dict:
    record = resolve_terminal_task(
        source_row_task(row, index, source_name=source_name, environment_configs=environment_configs), terminals
    )
    return task_prompt(record.task, record.execution)


class NemotronTaskDataset(SourceTaskDataset):
    """Convert answer, tool, and terminal source rows to portable tasks."""

    def __init__(
        self,
        datasets: Sequence[str],
        tokenizer: PreTrainedTokenizerBase,
        max_prompt_length: int,
        *,
        environment_configs: Mapping[str, dict],
        terminal_bench_data: Sequence[str | Mapping[str, Any]],
        cache_dir: Path,
        verifier_override: VerifierSpec | None = None,
        num_workers: int = 8,
    ):
        cache_dir = cache_dir.expanduser()
        self.terminals = terminal_task_index(
            materialize_harbor_tasks(terminal_bench_data, cache_dir=cache_dir, verifier_override=verifier_override)
        )
        super().__init__(
            datasets,
            tokenizer,
            max_prompt_length,
            environment_configs=environment_configs,
            num_workers=num_workers,
        )

    def prepare_dataset(self, dataset: Dataset) -> Dataset:
        return self._map_task_rows(
            dataset,
            partial(
                _nemotron_task_prompt,
                source_name=", ".join(self.datasets),
                environment_configs=self.environment_configs,
                terminals=self.terminals,
            ),
        )
