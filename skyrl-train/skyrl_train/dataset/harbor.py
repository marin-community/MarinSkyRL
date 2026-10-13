from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from contextlib import closing
from itertools import batched
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from transformers import PreTrainedTokenizerBase
from marinskyrl.harbor_import import harbor_task
from marinskyrl.harbor_preparation import TerminalBenchTaskDataset, harbor_task_ids
from marinskyrl.task_format import HARBOR_ID_PREFIX
from harbor_config.models.task.config import EnvironmentConfig, TaskConfig
from rolloutengine.lowering import SHELLBOX_SESSION
from rolloutengine.spec import LoweredTaskSpec, MachineRuntimeSpec, TaskRuntimeSpec, TaskSessionSpec
from shellbox.machine import Backend, NetworkPolicy
from taskcompendium.models import NoGrader, ScriptGrader, Source, TaskSpec, VerifyitGrader

from marinskyrl.packed_tasks import PackedTaskMaterializer, PackedTaskReference
from skyrl_train.dataset.tasks import LOWERED_TASK_COLUMN, TaskDataset, cache_tasks

MATERIALIZATION_BATCH_SIZE = 64


def harbor_machine(environment: EnvironmentConfig, user: str | int | None) -> MachineRuntimeSpec:
    return MachineRuntimeSpec(
        backend=Backend.DOCKER,
        network=NetworkPolicy.ALLOW if environment.allow_internet else NetworkPolicy.DENY,
        cpus=environment.cpus,
        memory_mb=environment.memory_mb,
        storage_mb=environment.storage_mb,
        gpus=0 if environment.gpus is None else environment.gpus,
        user=None if user is None else str(user),
        startup_timeout=environment.build_timeout_sec,
        cleanup_timeout=None,
    )


def lower_harbor_task(directory: Path, task: TaskSpec, *, session: TaskSessionSpec) -> LoweredTaskSpec:
    """Use package machine settings and phase deadlines, with other limits from the caller."""
    config = TaskConfig.model_validate_toml((directory / "task.toml").read_text())

    task_machine = harbor_machine(config.environment, config.agent.user)
    verifier_machine = (
        harbor_machine(config.verifier.environment or config.environment, config.verifier.user)
        if isinstance(task.grader, ScriptGrader)
        or (isinstance(task.grader, VerifyitGrader) and task.grader.environment is not None)
        else None
    )
    return LoweredTaskSpec(
        task=task,
        runtime=TaskRuntimeSpec(task_machine=task_machine, verifier_machine=verifier_machine),
        session=TaskSessionSpec.model_validate(
            {
                **session.model_dump(),
                "task_session": SHELLBOX_SESSION,
                "total_turn_timeout": config.agent.timeout_sec,
                "verifier_timeout": config.verifier.timeout_sec,
            }
        ),
    )


def materialize_harbor_tasks(
    data_files: Sequence[str | Mapping[str, Any]],
    *,
    cache_dir: Path,
    session: TaskSessionSpec,
    grader_override: NoGrader | None = None,
) -> Path:
    """Combine Harbor packages and pre-lowered task Parquet for one worker pool."""
    task_files = [source for source in data_files if isinstance(source, str) and source.endswith(".parquet")]
    sources = TerminalBenchTaskDataset([source for source in data_files if source not in task_files])
    if not len(sources) and not task_files:
        raise ValueError("No Harbor tasks matched the configured sources")

    def tasks() -> Iterator[LoweredTaskSpec]:
        with closing(PackedTaskMaterializer(cache_dir / "archives")) as materializer:
            for batch in batched(sources, MATERIALIZATION_BATCH_SIZE):
                references = [item for item in batch if isinstance(item, PackedTaskReference)]
                directories = materializer.materialize_batch(references)
                for item in batch:
                    if isinstance(item, Path):
                        directory = item
                        data_source = str(item)
                        uid = item.name
                    else:
                        directory = directories[item]
                        data_source = item.stable_uri()
                        uid = item.uid()
                    task = harbor_task(
                        directory,
                        source=Source(
                            dataset=data_source,
                            revision="unhashed",
                            row=uid,
                            importer_revision="skyrl-harbor-v1",
                        ),
                    )
                    if grader_override is not None:
                        task = task.model_copy(
                            update={
                                "grader": grader_override,
                                "resources": task.resources.model_copy(update={"verifier": ()}),
                            }
                        )
                    content = json.dumps(task.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode()
                    task = task.model_copy(
                        update={
                            "id": uid,
                            "tags": (
                                *task.tags,
                                *(
                                    f"{HARBOR_ID_PREFIX}{identifier}"
                                    for identifier in sorted(harbor_task_ids(directory))
                                ),
                            ),
                            "source": task.source.model_copy(
                                update={"revision": f"sha256:{hashlib.sha256(content).hexdigest()}"}
                            ),
                        }
                    )
                    yield lower_harbor_task(directory, task, session=session)
        for path in task_files:
            with pq.ParquetFile(path) as parquet:
                for batch in parquet.iter_batches(batch_size=MATERIALIZATION_BATCH_SIZE, columns=[LOWERED_TASK_COLUMN]):
                    for row in batch.to_pylist():
                        yield LoweredTaskSpec.model_validate_json(row[LOWERED_TASK_COLUMN])

    return cache_tasks(tasks(), cache_dir)


class HarborTaskDataset(TaskDataset):
    """Prepare Harbor source packages for the common task worker."""

    def __init__(
        self,
        data_files: Sequence[str | Mapping[str, Any]],
        tokenizer: PreTrainedTokenizerBase,
        max_prompt_length: int,
        *,
        cache_dir: Path,
        session: TaskSessionSpec,
        grader_override: NoGrader | None = None,
        num_workers: int = 8,
    ):
        self.task_path = materialize_harbor_tasks(
            data_files, cache_dir=cache_dir.expanduser(), session=session, grader_override=grader_override
        )
        super().__init__([str(self.task_path)], tokenizer, max_prompt_length, num_workers=num_workers)

    def uid(self, index: int) -> str:
        return LoweredTaskSpec.model_validate_json(self.dataframe[index][LOWERED_TASK_COLUMN]).task.id
