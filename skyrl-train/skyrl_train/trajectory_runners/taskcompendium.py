"""Dataset projection for independently lowered TaskCompendium tasks."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


ENV_CLASS = "taskcompendium_harbor"


class TaskCompendiumTaskDataset:
    """Load native lowering packages and bind them to one live policy endpoint."""

    def __init__(self, data_files: Sequence[str | Mapping[str, Any]], *, api_base: str, model_name: str):
        self._rows = self._load_rows(data_files, api_base=api_base, model_name=model_name)

    @staticmethod
    def _task_directories(root: Path) -> list[Path]:
        if not root.is_dir():
            raise ValueError(f"TaskCompendium data root does not exist: {root}")
        if (root / "manifest.json").is_file():
            return [root]
        tasks = sorted(path for path in root.iterdir() if (path / "manifest.json").is_file())
        if not tasks:
            raise ValueError(f"TaskCompendium data root has no lowering packages: {root}")
        return tasks

    @classmethod
    def _load_rows(
        cls,
        data_files: Sequence[str | Mapping[str, Any]],
        *,
        api_base: str,
        model_name: str,
    ) -> list[dict[str, Any]]:
        rows = []
        for value in data_files:
            if not isinstance(value, str):
                value = str(value["local_path"])
            for task_dir in cls._task_directories(Path(value)):
                execution_path = task_dir / "reference-execution.json"
                if not execution_path.is_file():
                    raise ValueError(f"TaskCompendium lowering lacks reference-execution.json: {task_dir}")
                execution = copy.deepcopy(json.loads(execution_path.read_text()))
                agent = execution["agent"]
                if agent.get("import_path", "").endswith(("ReplayAgent", "ActionOutputReplayAgent")):
                    raise ValueError(f"Replay agents cannot produce on-policy training data: {task_dir}")
                agent.setdefault("kwargs", {})["api_base"] = api_base
                agent["model_name"] = model_name
                rows.append(
                    {
                        "uid": task_dir.name,
                        "prompt": [{"role": "user", "content": str(task_dir)}],
                        "env_class": ENV_CLASS,
                        "env_extras": {"task_dir": str(task_dir), "execution": execution},
                    }
                )
        return rows

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._rows[index]

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self):
        return iter(self._rows)

    def collate_fn(self, item_list):
        return item_list
