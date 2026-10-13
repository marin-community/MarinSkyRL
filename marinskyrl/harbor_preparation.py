"""Prepare Harbor packages as bare tasks and separate execution overrides."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from contextlib import closing
from itertools import batched
from pathlib import Path
from typing import Any

from loguru import logger
from harbor_config.models.task.config import EnvironmentConfig, TaskConfig
from marinskyrl.harbor_import import harbor_task
from taskcompendium.models import NoGrader, Source, TaskSpec

from marinskyrl.packed_tasks import PackedTaskMaterializer, PackedTaskReference, select_task_references
from marinskyrl.task_sources import DirectoryDataSource, TaskTroveParquetSource, data_source
from marinskyrl.task_preparation import cache_tasks
from marinskyrl.task_format import HARBOR_ID_PREFIX

MATERIALIZATION_BATCH_SIZE = 64


def _machine_overrides(environment: EnvironmentConfig, user: str | int | None) -> dict:
    return {
        "network": "allow" if environment.allow_internet else "deny",
        "cpus": environment.cpus,
        "memory_mb": environment.memory_mb,
        "storage_mb": environment.storage_mb,
        "gpus": environment.gpus or 0,
        "user": None if user is None else str(user),
        "startup_timeout": environment.build_timeout_sec,
    }


def harbor_execution_overrides(directory: Path) -> dict:
    config = TaskConfig.model_validate_toml((directory / "task.toml").read_text())
    return {
        "runtime": {
            "task_machine": _machine_overrides(config.environment, config.agent.user),
            "verifier_machine": _machine_overrides(
                config.verifier.environment or config.environment, config.verifier.user
            ),
        },
        "session": {
            "task_session": "shellbox",
            "total_turn_timeout": config.agent.timeout_sec,
            "verifier_timeout": config.verifier.timeout_sec,
        },
    }


class TerminalBenchTaskDataset:
    """Terminal-bench directory tasks and lazily materialized packed tasks."""

    def __init__(self, data_files: Sequence[str | Mapping[str, Any]]):
        self.data_files = data_files
        self._items = self._load_data_files()
        logger.info(f"TerminalBenchTaskDataset initialized with {len(self._items)} tasks")

    def _directory_tasks(self, source_path: Path) -> list[Path]:
        if not source_path.exists():
            logger.warning(f"Path does not exist: {source_path}")
            return []
        if not source_path.is_dir():
            logger.warning(f"File {source_path} cannot be a valid task directory (missing instruction.md)")
            return []

        all_dirs = [path for path in source_path.iterdir() if path.is_dir()]
        valid = [path for path in all_dirs if self._is_valid_task_directory(path)]
        if valid:
            logger.info(f"Found {len(valid)} valid task directories out of {len(all_dirs)} total directories")
            return valid
        if self._is_valid_task_directory(source_path):
            return [source_path]
        logger.warning(f"No valid task directories found in {source_path}")
        return []

    def _packed_tasks(self, source: TaskTroveParquetSource) -> list[PackedTaskReference]:
        summary = select_task_references(source, dataset_path=source.resolved_path())
        if source.snapshot is None:
            return list(summary.references)
        if source.snapshot.count != len(summary.references):
            raise ValueError(
                f"TaskTrove selection count changed: launch selected {source.snapshot.count}, "
                f"Ray selected {len(summary.references)}"
            )
        if source.snapshot.digest != summary.digest:
            raise ValueError("TaskTrove selection digest changed between launch and Ray")
        if source.snapshot.distinct_environment_count != summary.distinct_environment_count:
            raise ValueError("TaskTrove environment count changed between launch and Ray")
        return list(summary.references)

    def _load_data_files(self) -> list[Path | PackedTaskReference]:
        items: list[Path | PackedTaskReference] = []
        for value in self.data_files:
            if isinstance(value, str):
                items.extend(self._directory_tasks(Path(value)))
                continue
            source = data_source(value)
            if isinstance(source, DirectoryDataSource):
                items.extend(self._directory_tasks(Path(source.resolved_path())))
            else:
                items.extend(self._packed_tasks(source))
        if all(isinstance(item, Path) for item in items):
            return sorted(items)
        return items

    def _is_valid_task_directory(self, task_path: Path) -> bool:
        if not task_path.is_dir():
            return False
        return (task_path / "instruction.md").is_file() or (
            (task_path / "task.toml").is_file() and (task_path / "steps").is_dir()
        )

    def __getitem__(self, index: int) -> Path | PackedTaskReference:
        if index >= len(self._items):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self._items)}")
        return self._items[index]

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]


def harbor_task_ids(task_path: Path) -> set[str]:
    """Return Harbor's directory ID and any source-dataset ID in task metadata."""
    task_ids = {task_path.name}

    swe_gym_config = task_path / "tests" / "config.json"
    if swe_gym_config.is_file():
        config = json.loads(swe_gym_config.read_text())
        instance_id = config.get("instance_id")
        if isinstance(instance_id, str) and instance_id:
            task_ids.add(instance_id)

    r2e_info = task_path / "tests" / "test_info.json"
    if r2e_info.is_file():
        info = json.loads(r2e_info.read_text())
        github_repo = info.get("github_repo")
        base_commit = info.get("base_commit")
        if isinstance(github_repo, str) and github_repo and isinstance(base_commit, str) and base_commit:
            task_ids.add(f"{github_repo.replace('/', '__')}-{base_commit}")

    return {task_id.casefold() for task_id in task_ids}


def materialize_harbor_tasks(
    data_files: Sequence[str | Mapping[str, Any]],
    *,
    cache_dir: Path,
    grader_override: NoGrader | None = None,
    execution_output: Path | None = None,
) -> Path:
    """Prepare Harbor packages as a bare TaskSpec artifact."""
    sources = TerminalBenchTaskDataset(data_files)
    if not len(sources):
        raise ValueError("No Harbor tasks matched the configured sources")
    if execution_output is not None and execution_output.exists():
        raise FileExistsError(execution_output)
    execution = {}
    seen_task_ids = set()

    def tasks() -> Iterator[TaskSpec]:
        with closing(PackedTaskMaterializer(cache_dir / "archives")) as materializer:
            for batch in batched(sources, MATERIALIZATION_BATCH_SIZE):
                references = [item for item in batch if isinstance(item, PackedTaskReference)]
                directories = materializer.materialize_batch(references)
                for item in batch:
                    if isinstance(item, Path):
                        directory = item
                        uid = item.name
                        source_name = f"harbor/{uid}"
                    else:
                        directory = directories[item]
                        source_name = item.stable_uri()
                        uid = item.uid()
                    if uid in seen_task_ids:
                        raise ValueError(f"Duplicate Harbor task ID {uid!r}")
                    seen_task_ids.add(uid)
                    task = harbor_task(
                        directory,
                        source=Source(
                            dataset=source_name,
                            revision="unhashed",
                            row=uid,
                            importer_revision="skyrl-harbor-v1",
                        ),
                    )
                    if execution_output is not None:
                        execution[source_name] = harbor_execution_overrides(directory)
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
                    yield task

    path = cache_tasks(tasks(), cache_dir)
    if execution_output is not None:
        execution_output.write_text(json.dumps({"environment": {"task_sessions": {"sources": execution}}}))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Harbor package directory")
    parser.add_argument("--packed-source", type=Path, help="TaskTrove source descriptor in JSON")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execution-output", type=Path, required=True, help="Run settings in JSON")
    parser.add_argument("--disable-verification", action="store_true")
    args = parser.parse_args()
    if (args.input is None) == (args.packed_source is None):
        parser.error("Select one --input or --packed-source")
    sources = [str(args.input)] if args.packed_source is None else [json.loads(args.packed_source.read_text())]
    path = materialize_harbor_tasks(
        sources,
        cache_dir=args.output_dir,
        grader_override=NoGrader(reason="Verification is disabled") if args.disable_verification else None,
        execution_output=args.execution_output,
    )
    print(path)


if __name__ == "__main__":
    main()
