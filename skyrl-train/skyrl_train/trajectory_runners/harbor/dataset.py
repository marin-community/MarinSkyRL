from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from loguru import logger

from marinskyrl.packed_tasks import PackedTaskReference, select_task_references
from marinskyrl.task_sources import DirectoryDataSource, TaskTroveParquetSource, data_source


class TerminalBenchTaskDataset:
    """Terminal-bench directory tasks and lazily materialized packed tasks."""

    def __init__(self, data_files: Sequence[str | Mapping[str, Any]]):
        self.data_files = data_files
        self._items = self._load_data_files()
        self.task_paths = [item for item in self._items if isinstance(item, Path)]
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
        if source.selected_count is not None and source.selected_count != len(summary.references):
            raise ValueError(
                f"TaskTrove selection count changed: launch selected {source.selected_count}, "
                f"Ray selected {len(summary.references)}"
            )
        if source.selection_digest is not None and source.selection_digest != summary.digest:
            raise ValueError("TaskTrove selection digest changed between launch and Ray")
        if (
            source.distinct_environment_count is not None
            and source.distinct_environment_count != summary.distinct_environment_count
        ):
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
        instruction_file = task_path / "instruction.md"
        return instruction_file.exists() and instruction_file.is_file()

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index >= len(self._items):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self._items)}")
        item = self._items[index]
        if isinstance(item, Path):
            path = str(item)
            return {
                "prompt": path,
                "env_class": None,
                "env_extras": {"data_source": path},
                "uid": item.name,
            }
        uri = item.stable_uri()
        return {
            "prompt": uri,
            "env_class": None,
            "env_extras": {"data_source": uri, "packed_task": asdict(item)},
            "uid": item.uid(),
        }

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def get_task_paths(self) -> list[Path]:
        """Return directory-backed task paths; packed tasks have no path before rollout."""
        return self.task_paths.copy()

    def collate_fn(self, item_list):
        return item_list
