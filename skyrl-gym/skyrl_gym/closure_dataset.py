"""Lower audited closure cases to exact-token task sessions."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from rolloutengine.models import TaskRolloutSpec
from taskcompendium.models import Source

from skyrl_gym.closure_task import STRUCTURE_CLOSURE_SESSION, ClosurePrompt
from skyrl_gym.source_task import source_task
from skyrl_gym.rollout_specs import rollout_spec


class ConversionFailure(TypedDict):
    row: int
    error: str


@dataclass(frozen=True)
class ClosureConversion:
    tasks: tuple[TaskRolloutSpec, ...]
    failures: tuple[ConversionFailure, ...]


def closure_tasks(directory: Path, split: str, session: dict) -> ClosureConversion:
    """Validate every row, preserving all conversion failures for the caller.

    Consume one immutable output of the conditional-token preparer. Callers must
    reject the entire conversion when failures are present, before submission.
    """
    if split not in ("train", "validation"):
        raise ValueError("Choose the train or validation partition")
    manifest = json.loads((directory / "manifest.json").read_text())
    contents = (directory / f"{split}.jsonl").read_bytes()
    digest = hashlib.sha256(contents).hexdigest()
    if digest != manifest["hashes"][split] or manifest["status"] != "local-token-roundtrip-passed":
        raise ValueError("Closure input does not match a passed immutable token audit")
    tasks = []
    failures: list[ConversionFailure] = []
    seen = set()
    for index, line in enumerate(contents.splitlines()):
        try:
            record = json.loads(line)
            case_id = record["case_id"]
            if record["split"] != split or case_id in seen:
                raise ValueError("Closure case partition mismatch or duplicate identity")
            seen.add(case_id)
            prompt = ClosurePrompt.model_validate({key: record[key] for key in ClosurePrompt.model_fields})
            task = source_task(
                [{"role": "user", "content": record["instruction"]}],
                prompt.model_dump(mode="json"),
                {},
                Source(
                    dataset="structure-closure",
                    revision=digest,
                    row=case_id,
                    importer_revision="closure-replay-v1",
                ),
            )
            tasks.append(rollout_spec(task, {**session, "task_session": STRUCTURE_CLOSURE_SESSION, "max_turns": 1}))
        except (ValueError, KeyError, TypeError) as error:
            failures.append({"row": index, "error": str(error)})
    expected_count = sum(count for key, count in manifest["counts"].items() if key.startswith(split + "/"))
    if len(contents.splitlines()) != expected_count:
        failures.append({"row": -1, "error": "Closure row count differs from the token audit"})
    return ClosureConversion(tuple(tasks), tuple(failures))
