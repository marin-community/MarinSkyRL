"""Load prepared TaskSpecs and apply run-specific execution settings."""

from collections.abc import Mapping, Sequence
from functools import partial

from datasets import Dataset
from rolloutengine.lowering import SHELLBOX_SESSION
from taskcompendium.models import TaskSpec
from transformers import PreTrainedTokenizerBase

from marinskyrl.task_format import TASK_SPEC_COLUMN, TASKCOMPENDIUM_ENVIRONMENT
from skyrl_train.dataset.dataset import PromptDataset
from skyrl_train.rollouts.group_grader import task_group_grader
from skyrl_train.rollouts.task_lowering import TaskExecutionSettings, lowered_task
from skyrl_train.task_inputs import TaskInput, TrajectoryMetadata, task_messages


class TaskSpecDataset(PromptDataset):
    """Load prepared TaskSpecs and add the run's machine and session settings."""

    def __init__(
        self,
        datasets: Sequence[str],
        tokenizer: PreTrainedTokenizerBase,
        max_prompt_length: int,
        *,
        execution: TaskExecutionSettings,
        source_executions: Mapping[str, TaskExecutionSettings],
        num_workers: int = 8,
    ):
        self.execution = execution
        self.source_executions = source_executions
        super().__init__(list(datasets), tokenizer, max_prompt_length, num_workers=num_workers)

    def prepare_dataset(self, dataset: Dataset) -> Dataset:
        if TASK_SPEC_COLUMN not in dataset.column_names:
            raise ValueError("Generated RL requires prepared TaskSpec JSON in the task_spec column")
        prepared = dataset.map(
            partial(_task_spec_prompt, execution=self.execution, source_executions=self.source_executions),
            num_proc=self.num_workers,
            keep_in_memory=True,
        )
        self.source_names = frozenset(prepared.unique("task_source"))
        return prepared.remove_columns("task_source")

    def uid(self, index: int) -> str:
        return self.dataframe[index]["task_id"]

    def __getitem__(self, index: int) -> TaskInput:
        row = self.dataframe[index]
        task = TaskSpec.model_validate_json(row[TASK_SPEC_COLUMN])
        lowered = lowered_task(task, self.source_executions.get(task.source.dataset, self.execution))
        return TaskInput(
            lowered=lowered,
            metadata=TrajectoryMetadata(
                uid=task.id,
                metric_class=TASKCOMPENDIUM_ENVIRONMENT
                if lowered.session.task_session == SHELLBOX_SESSION
                else lowered.session.task_session,
                data_source=row.get("data_source") or task.source.dataset,
                teacher_route=row.get("teacher_route"),
                metric_tags=tuple(row.get("metric_tags") or ()),
            ),
            group_grader=task_group_grader(lowered),
        )

    def collate_fn(self, rows: Sequence[TaskInput]) -> list[dict]:
        return [{"uid": row.metadata.uid, "input": row} for row in rows]


def _task_spec_prompt(
    row: dict,
    *,
    execution: TaskExecutionSettings,
    source_executions: Mapping[str, TaskExecutionSettings],
) -> dict:
    task = TaskSpec.model_validate_json(row[TASK_SPEC_COLUMN])
    return {
        "prompt": task_messages(lowered_task(task, source_executions.get(task.source.dataset, execution))),
        "task_id": task.id,
        "task_source": task.source.dataset,
    }
