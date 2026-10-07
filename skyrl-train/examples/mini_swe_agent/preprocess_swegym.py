"""Materialize SWE-Gym and SWE-Bench rows as executable TaskCompendium tasks."""

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

import datasets
from rolloutengine.spec import LoweredTaskSpec, MachineRuntimeSpec, TaskRuntimeSpec, TaskSessionSpec
from shellbox.machine import NetworkPolicy
from taskcompendium.importers.swe import SWEInstance, swe_task
from taskcompendium.models import EnvironmentRequirements, Source
from skyrl_train.dataset.tasks import write_tasks

TRAIN_DATASET = "SumanthRH/SWE-Gym-Subset"
EVAL_DATASET = "SumanthRH/SWE-bench_Verified"
ENVIRONMENT_VARIABLES = {
    "PAGER": "cat",
    "MANPAGER": "cat",
    "LESS": "-R",
    "PIP_PROGRESS_BAR": "off",
    "TQDM_DISABLE": "1",
}


def materialize(
    dataset: str,
    revision: str,
    split: str,
    output: Path,
    *,
    images: Mapping[str, str],
    runtime: TaskRuntimeSpec,
    session: TaskSessionSpec,
) -> None:
    rows = datasets.load_dataset(dataset, "default", revision=revision, split=split)
    instances = (SWEInstance.model_validate(row) for row in rows)
    tasks = (
        swe_task(
            row,
            source=Source(dataset=dataset, revision=revision, row=str(index), importer_revision="swe-v1"),
            environment=EnvironmentRequirements(
                docker_image=images[row.instance_id],
                working_directory="/testbed",
                environment_variables=ENVIRONMENT_VARIABLES,
            ),
        )
        for index, row in enumerate(instances)
    )
    write_tasks(output, (LoweredTaskSpec(task=task, runtime=runtime, session=session) for task in tasks))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--train_revision", required=True, help="Pinned training dataset revision")
    parser.add_argument("--eval_revision", required=True, help="Pinned evaluation dataset revision")
    parser.add_argument(
        "--image_manifest", type=Path, required=True, help="JSON map of instance IDs to digest-pinned images"
    )
    args = parser.parse_args()
    output = args.output_dir.expanduser()
    output.mkdir(parents=True, exist_ok=True)
    images = json.loads(args.image_manifest.read_text())
    machine = MachineRuntimeSpec(
        backend="docker",
        network=NetworkPolicy.ALLOW,
        cpus=1,
        memory_mb=8192,
        storage_mb=None,
        gpus=0,
        user=None,
        startup_timeout=600,
        cleanup_timeout=None,
    )
    runtime = TaskRuntimeSpec(task_machine=machine, verifier_machine=machine)
    session = TaskSessionSpec(
        task_session="shellbox",
        max_turns=50,
        model_turn_timeout=None,
        tool_turn_timeout=120,
        total_turn_timeout=3600,
        attempt_timeout=None,
        verifier_timeout=3600,
        cleanup_timeout=30,
    )
    materialize(
        TRAIN_DATASET,
        args.train_revision,
        "train",
        output / "train.parquet",
        images=images,
        runtime=runtime,
        session=session,
    )
    materialize(
        EVAL_DATASET,
        args.eval_revision,
        "test",
        output / "validation.parquet",
        images=images,
        runtime=runtime,
        session=session,
    )


if __name__ == "__main__":
    main()
