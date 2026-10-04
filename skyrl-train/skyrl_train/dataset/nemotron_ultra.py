"""Prepare mixed Nemotron rows as portable tasks before rollout execution."""

from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from datasets import Dataset
from taskcompendium.environment import ExternalVerifierSpec
from taskcompendium.models import TaskSpec, VerifierSpec
from taskcompendium.parquet import read_tasks
from transformers import PreTrainedTokenizerBase

from skyrl_train.dataset.harbor import materialize_harbor_tasks
from skyrl_train.dataset.tasks import SourceTaskDataset


def terminal_task_index(path: Path) -> dict[str, TaskSpec]:
    """Index portable Harbor tasks by their directory and source instance IDs."""
    result = {}
    for task in read_tasks(str(path)):
        for identifier in {task.id.casefold(), *task.metadata["harbor_task_ids"]}:
            if identifier in result and result[identifier] != task:
                raise ValueError(f"Duplicate terminal-bench task ID {identifier!r}")
            result[identifier] = task
    return result


def resolve_terminal_task(task: TaskSpec, terminals: Mapping[str, TaskSpec]) -> TaskSpec:
    """Keep source metadata while replacing a terminal row with its executable task."""
    verifier = ExternalVerifierSpec.model_validate_json(task.verifier.parameters_json)
    extras = verifier.parameters["extras"]
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
    return terminal.model_copy(
        update={
            "id": task.id,
            "source": task.source,
            "metadata": {
                **terminal.metadata,
                **task.metadata,
                "skyrl_extras": extras,
                "harbor_source": terminal.source.model_dump(mode="json"),
            },
        }
    )


class NemotronTaskDataset(SourceTaskDataset):
    """Convert answer, tool, and terminal source rows to one task Parquet file."""

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
            cache_dir=cache_dir,
            num_workers=num_workers,
        )

    def _tasks(self, dataset: Dataset) -> Iterator[TaskSpec]:
        return (resolve_terminal_task(task, self.terminals) for task in super()._tasks(dataset))
