"""Convert source rows and stored task specifications for the rollout loader."""

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from functools import partial
from itertools import batched
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset
from rolloutengine.task_session import session_start
from taskcompendium.environment import EnvironmentSpec, ExternalVerifierSpec
from taskcompendium.importers.skyrl import source_task
from taskcompendium.models import Source, TaskSpec, VerifierKind
from taskcompendium.submission import AnswerFormat, SubmissionConvention
from transformers import PreTrainedTokenizerBase

from skyrl_train.dataset.dataset import PromptDataset
from skyrl_train.rollouts.group_grader import task_group_grader

TASKCOMPENDIUM_ENVIRONMENT = "taskcompendium"
TASK_SCHEMA = pa.schema([pa.field("task_spec", pa.string(), nullable=False)])
PARQUET_BATCH_SIZE = 1024


def task_prompt(task: TaskSpec) -> dict:
    """Prepare public messages and private worker inputs from a task."""
    verifier = (
        ExternalVerifierSpec.model_validate_json(task.verifier.parameters_json)
        if task.verifier.kind == VerifierKind.EXTERNAL
        else None
    )
    convention = SubmissionConvention(id="rollout", answer_format=AnswerFormat.PLAIN)
    result = {
        **(
            verifier.parameters["extras"]
            if verifier is not None and task.environment.interaction is not None
            else task.metadata.get("skyrl_extras", {})
        ),
        "prompt": session_start(task, convention).messages,
        "env_class": task.environment.interaction or TASKCOMPENDIUM_ENVIRONMENT,
        "data_source": task.source.dataset,
        "group_grader": (
            specification.model_dump_json() if (specification := task_group_grader(task)) is not None else None
        ),
    }
    if "teacher_route" in task.metadata:
        result["teacher_route"] = task.metadata["teacher_route"]
    return result


class TaskDataset(PromptDataset):
    """Prepare the public conversation while retaining the private task for workers."""

    def prepare_dataset(self, dataset):
        return dataset.map(_stored_task_prompt, num_proc=self.num_workers)


def _stored_task_prompt(row: dict) -> dict:
    return task_prompt(TaskSpec.model_validate_json(row["task_spec"]))


def source_row_task(
    row: Mapping[str, Any], index: int, *, source_name: str, environment_configs: Mapping[str, dict]
) -> TaskSpec:
    """Convert a source row with stable provenance and private verifier inputs."""
    row = dict(row)
    content = json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    source = Source(
        dataset=row.get("data_source") or source_name,
        revision=f"sha256:{hashlib.sha256(content).hexdigest()}",
        row=str(index),
        importer_revision="skyrl-task-session-v1",
    )
    prompt = row.pop("prompt")
    environment = row.pop("env_class")
    assert isinstance(environment, str)
    config = dict(environment_configs.get(environment, {}))
    machine = config.pop("machine", None)
    machines = config.pop("machines", {})
    if environment == "nemotron_ultra":
        ultra = row["extra_info"]["nemotron_ultra"]
        agent = ultra["agent"]
        machine = machines.get(agent, machine)
    elif environment == "openenv":
        machine = machines.get(row["env_name"], machine)
    return source_task(
        prompt,
        environment,
        row,
        config,
        source,
        environment=None if machine is None else EnvironmentSpec.model_validate(machine),
    )


def _source_task_prompt(row: dict, index: int, *, source_name: str, environment_configs: Mapping[str, dict]) -> dict:
    task = source_row_task(row, index, source_name=source_name, environment_configs=environment_configs)
    return {"task_spec": task.model_dump_json(), **task_prompt(task)}


def write_tasks(path: Path, tasks: Iterable[TaskSpec]) -> None:
    """Write a local task dataset in bounded Parquet batches."""
    with pq.ParquetWriter(path, TASK_SCHEMA) as writer:
        for batch in batched(tasks, PARQUET_BATCH_SIZE):
            writer.write_table(
                pa.Table.from_pydict({"task_spec": [task.model_dump_json() for task in batch]}, TASK_SCHEMA)
            )


def cache_tasks(tasks: Iterable[TaskSpec], cache_dir: Path) -> Path:
    """Write private task Parquet and use its content digest as the filename."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=cache_dir) as directory:
        temporary_path = Path(directory) / "tasks.parquet"
        write_tasks(temporary_path, tasks)
        with temporary_path.open("rb") as contents:
            digest = hashlib.file_digest(contents, "sha256").hexdigest()
        path = cache_dir / f"{digest}.parquet"
        temporary_path.chmod(0o600)
        temporary_path.replace(path)
    return path


class SourceTaskDataset(PromptDataset):
    """Convert source datasets to portable tasks with explicit session and machine inputs."""

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
        return dataset.map(
            partial(
                _source_task_prompt,
                source_name=", ".join(self.datasets),
                environment_configs=self.environment_configs,
            ),
            with_indices=True,
            remove_columns=dataset.column_names,
            num_proc=self.num_workers,
            keep_in_memory=True,
        )
