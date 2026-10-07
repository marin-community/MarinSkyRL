"""Convert source rows and lowered task specifications for the rollout loader."""

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import partial
from itertools import batched
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset
from rolloutengine.spec import LoweredTaskSpec, MachineRuntimeSpec, TaskRuntimeSpec, TaskSessionSpec
from rolloutengine.lowering import SHELLBOX_SESSION
from rolloutengine.task_session import session_start
from taskcompendium.importers.skyrl import ExternalVerifierSpec, source_task
from taskcompendium.models import EnvironmentRequirements, Source
from taskcompendium.submission import AnswerFormat, SubmissionConvention
from transformers import PreTrainedTokenizerBase

from skyrl_train.dataset.dataset import PromptDataset
from skyrl_train.rollouts.group_grader import task_group_grader

TASKCOMPENDIUM_ENVIRONMENT = "taskcompendium"
LOWERED_TASK_COLUMN = "lowered_task_spec"
TASK_SCHEMA = pa.schema([pa.field(LOWERED_TASK_COLUMN, pa.string(), nullable=False)])
PARQUET_BATCH_SIZE = 1024


def task_prompt(lowered: LoweredTaskSpec) -> dict:
    """Prepare public messages and private worker inputs from a lowered task."""
    task = lowered.task
    extras = (
        ExternalVerifierSpec.model_validate_json(task.verifier.parameters_json).parameters["extras"]
        if task.verifier.kind == "external"
        else {}
    )
    convention = SubmissionConvention(id="rollout", answer_format=AnswerFormat.PLAIN)
    return {
        **extras,
        LOWERED_TASK_COLUMN: lowered.model_dump_json(),
        "prompt": session_start(task, convention).messages,
        "env_class": TASKCOMPENDIUM_ENVIRONMENT
        if lowered.session.task_session == SHELLBOX_SESSION
        else lowered.session.task_session,
        "data_source": task.source.dataset,
        "group_grader": specification.model_dump_json()
        if (specification := task_group_grader(lowered)) is not None
        else None,
    }


class TaskDataset(PromptDataset):
    """Render public conversation fields and retain private lowered tasks for workers."""

    def prepare_dataset(self, dataset):
        return dataset.map(_stored_task_prompt, num_proc=self.num_workers)


def _stored_task_prompt(row: dict) -> dict:
    return task_prompt(LoweredTaskSpec.model_validate_json(row[LOWERED_TASK_COLUMN]))


def source_row_task(
    row: Mapping[str, Any], index: int, *, source_name: str, environment_configs: Mapping[str, dict]
) -> LoweredTaskSpec:
    """Convert a source row once, with explicit session and machine settings."""
    row = dict(row)
    content = json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    source = Source(
        dataset=row.get("data_source") or source_name,
        revision=f"sha256:{hashlib.sha256(content).hexdigest()}",
        row=str(index),
        importer_revision="skyrl-task-session-v2",
    )
    prompt = row.pop("prompt")
    session_name = row.pop("env_class")
    if not isinstance(session_name, str):
        raise ValueError("The source session identifier must be a string")
    config = dict(environment_configs.get(session_name, {}))
    machine = config.pop("machine", None)
    machines = config.pop("machines", {})
    if session_name == "nemotron_ultra":
        machine = machines.get(row["extra_info"]["nemotron_ultra"]["agent"], machine)
    elif session_name == "openenv":
        machine = machines.get(row["env_name"], machine)
    session = {**environment_configs["session"], **config.pop("session", {}), "task_session": session_name}
    requirements = (
        EnvironmentRequirements()
        if machine is None
        else EnvironmentRequirements.model_validate(machine["requirements"])
    )
    runtime = None if machine is None else MachineRuntimeSpec.model_validate(machine["runtime"])
    task = source_task(prompt, row, config, source, environment=requirements)
    return LoweredTaskSpec(
        task=task,
        runtime=TaskRuntimeSpec(task_machine=runtime, verifier_machine=None),
        session=TaskSessionSpec.model_validate(session),
    )


def _source_task_prompt(row: dict, index: int, *, source_name: str, environment_configs: Mapping[str, dict]) -> dict:
    return task_prompt(source_row_task(row, index, source_name=source_name, environment_configs=environment_configs))


def write_tasks(path: Path, records: Iterable[LoweredTaskSpec]) -> None:
    """Write private lowered tasks in bounded Parquet batches."""
    with pq.ParquetWriter(path, TASK_SCHEMA) as writer:
        path.chmod(0o600)
        for batch in batched(records, PARQUET_BATCH_SIZE):
            writer.write_table(
                pa.Table.from_pydict(
                    {LOWERED_TASK_COLUMN: [record.model_dump_json() for record in batch]},
                    TASK_SCHEMA,
                )
            )


def cache_tasks(records: Iterable[LoweredTaskSpec], cache_dir: Path) -> Path:
    """Cache private task Parquet under its content digest."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=cache_dir) as directory:
        temporary_path = Path(directory) / "tasks.parquet"
        write_tasks(temporary_path, records)
        with temporary_path.open("rb") as contents:
            digest = hashlib.file_digest(contents, "sha256").hexdigest()
        path = cache_dir / f"{digest}.parquet"
        temporary_path.chmod(0o600)
        temporary_path.replace(path)
    return path


class SourceTaskDataset(PromptDataset):
    """Prepare source rows as in-memory prompts and private lowered tasks."""

    def __init__(
        self,
        datasets: Sequence[str],
        tokenizer: PreTrainedTokenizerBase,
        max_prompt_length: int,
        *,
        environment_configs: Mapping[str, dict],
        num_workers: int = 8,
    ):
        self.environment_configs = environment_configs
        super().__init__(list(datasets), tokenizer, max_prompt_length, num_workers=num_workers)

    def prepare_dataset(self, dataset: Dataset) -> Dataset:
        return self._map_task_rows(
            dataset,
            partial(
                _source_task_prompt, source_name=", ".join(self.datasets), environment_configs=self.environment_configs
            ),
        )

    def _map_task_rows(self, dataset: Dataset, mapper: Callable[[dict, int], dict]) -> Dataset:
        return dataset.map(
            mapper,
            with_indices=True,
            remove_columns=dataset.column_names,
            num_proc=self.num_workers,
            keep_in_memory=True,
        )
