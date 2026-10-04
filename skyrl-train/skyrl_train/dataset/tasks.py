"""Unified task Parquet inputs for the rollout loader."""

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from datasets import Dataset
from transformers import PreTrainedTokenizerBase
from taskcompendium.environment import ExternalVerifierSpec
from taskcompendium.importers.skyrl import GYM_INTERACTION, gym_task
from taskcompendium.models import Source, TaskSpec, VerifierKind
from taskcompendium.parquet import write_tasks
from rolloutengine.task_session import session_start
from taskcompendium.submission import AnswerFormat, SubmissionConvention

from skyrl_train.dataset.dataset import PromptDataset
from skyrl_train.rollouts.group_grader import task_group_grader

TASKCOMPENDIUM_ENVIRONMENT = "taskcompendium"


def task_prompt(row: dict) -> dict:
    task = TaskSpec.model_validate_json(row["task_spec"])
    verifier = (
        ExternalVerifierSpec.model_validate_json(task.verifier.parameters_json)
        if task.verifier.kind == VerifierKind.EXTERNAL
        else None
    )
    convention = SubmissionConvention(id="rollout", answer_format=AnswerFormat.PLAIN)
    result = {
        **(
            verifier.parameters["extras"]
            if verifier is not None and task.environment.interaction == GYM_INTERACTION
            else task.metadata.get("skyrl_extras", {})
        ),
        "prompt": session_start(task, convention).messages,
        "env_class": verifier.name if verifier is not None else TASKCOMPENDIUM_ENVIRONMENT,
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
        return dataset.map(task_prompt, num_proc=self.num_workers)


def gym_tasks(dataset: Dataset, *, source_name: str, environment_configs: Mapping[str, dict]) -> Iterator[TaskSpec]:
    """Convert source rows with stable provenance and private verifier inputs."""
    for index, source_row in enumerate(dataset):
        row = dict(source_row)
        content = json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        yield _gym_task(
            row,
            environment_configs,
            Source(
                dataset=row.get("data_source") or source_name,
                revision=f"sha256:{hashlib.sha256(content).hexdigest()}",
                row=str(index),
                importer_revision="skyrl-gym-v1",
            ),
        )


def _gym_task(row: dict[str, Any], environment_configs: Mapping[str, dict], source: Source) -> TaskSpec:
    prompt = row.pop("prompt")
    environment = row.pop("env_class")
    assert isinstance(environment, str)
    return gym_task(prompt, environment, row, environment_configs.get(environment, {}), source)


def cache_tasks(tasks: Iterator[TaskSpec], cache_dir: Path) -> Path:
    """Write private task Parquet and use its content digest as the filename."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=cache_dir) as directory:
        temporary_path = Path(directory) / "tasks.parquet"
        write_tasks(str(temporary_path), tasks)
        with temporary_path.open("rb") as contents:
            digest = hashlib.file_digest(contents, "sha256").hexdigest()
        path = cache_dir / f"{digest}.parquet"
        temporary_path.chmod(0o600)
        temporary_path.replace(path)
    return path


class GymTaskDataset(TaskDataset):
    """Convert Gym source datasets to the same task format used by other importers."""

    def __init__(
        self,
        datasets: Sequence[str],
        tokenizer: PreTrainedTokenizerBase,
        max_prompt_length: int,
        *,
        environment_configs: Mapping[str, dict],
        cache_dir: Path,
        num_workers: int = 8,
    ):
        self.environment_configs = environment_configs
        self.cache_dir = cache_dir.expanduser()
        super().__init__(list(datasets), tokenizer, max_prompt_length, num_workers=num_workers)

    def _tasks(self, dataset: Dataset) -> Iterator[TaskSpec]:
        return gym_tasks(
            dataset,
            source_name=", ".join(self.datasets),
            environment_configs=self.environment_configs,
        )

    def prepare_dataset(self, dataset: Dataset) -> Dataset:
        self.task_path = cache_tasks(self._tasks(dataset), self.cache_dir)
        return super().prepare_dataset(Dataset.from_parquet(str(self.task_path)))
