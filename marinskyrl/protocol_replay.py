"""Convert a versioned training or validation replay to the common task format."""

import argparse
import hashlib
import json
from enum import StrEnum
from pathlib import Path

from skyrl_gym.source_task import source_task
from skyrl_gym.tool_protocol_task import ReplayPrompt
from taskcompendium.models import Source, TaskSpec

from marinskyrl.task_preparation import write_tasks


class ReplayPartition(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"


def replay_tasks(cases_path: Path, manifest_path: Path, *, partition: ReplayPartition) -> list[TaskSpec]:
    """Validate replay identity and the selected task partition before preparation."""
    contents = cases_path.read_bytes()
    manifest = json.loads(manifest_path.read_text())
    digest = hashlib.sha256(contents).hexdigest()
    if digest != manifest["cases_sha256"]:
        raise ValueError("Replay cases do not match their immutable manifest")
    training = set(manifest["train_task_ids"])
    validation = set(manifest["diagnostic_reserved_task_ids"])
    parity = set(manifest["parity_task_ids"])
    if training & validation or (training | validation) & parity:
        raise ValueError("Replay task partitions overlap training, diagnostic, or BFCL-Parity")
    reserved = training if partition is ReplayPartition.TRAIN else validation
    count_key = "training_cases" if partition is ReplayPartition.TRAIN else "diagnostic_cases"
    tasks = []
    seen = set()
    for line in contents.splitlines():
        case = json.loads(line)
        task_id = case["task_source_id"]
        if task_id not in reserved or task_id in seen:
            raise ValueError(f"Unreserved or duplicated replay task: {task_id}")
        seen.add(task_id)
        prompt = ReplayPrompt(prompt_token_ids=case["prompt_token_ids"])
        prefix_digest = hashlib.sha256(json.dumps(prompt.prompt_token_ids, separators=(",", ":")).encode()).hexdigest()
        if prefix_digest != case["prefix_sha256"]:
            raise ValueError(f"Replay prefix hash mismatch: {task_id}")
        task = source_task(
            [{"role": "user", "content": f"Retained token-prefix replay for BFCL task {task_id}."}],
            prompt.model_dump(mode="json"),
            {},
            Source(
                dataset="tool-protocol-replay", revision=digest, row=task_id, importer_revision="protocol-replay-v1"
            ),
        )
        tasks.append(task)
    if not tasks or len(tasks) != manifest[count_key]:
        raise ValueError("Replay count does not match the selected partition manifest")
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partition", type=ReplayPartition, choices=list(ReplayPartition), required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    tasks = replay_tasks(args.cases, args.manifest, partition=args.partition)
    write_tasks(args.output, tasks)


if __name__ == "__main__":
    main()
