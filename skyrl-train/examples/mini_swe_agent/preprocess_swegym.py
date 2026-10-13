"""Materialize SWE-Gym and SWE-Bench rows as executable TaskCompendium tasks."""

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

import datasets
from taskcompendium.importers.swe import SWEInstance, swe_task
from taskcompendium.models import CommandSemantics, EnvironmentRequirements, Source
from marinskyrl.task_preparation import write_tasks

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
) -> None:
    rows = datasets.load_dataset(dataset, "default", revision=revision, split=split)
    instances = (SWEInstance.model_validate(row) for row in rows)
    tasks = (
        swe_task(
            row,
            source=Source(dataset=dataset, revision=revision, row=str(index), importer_revision="swe-v1"),
            environment=EnvironmentRequirements(
                command_semantics=CommandSemantics.LINUX_PROCESS,
                docker_image=images[row.instance_id],
                working_directory="/testbed",
                environment_variables=ENVIRONMENT_VARIABLES,
            ),
        )
        for index, row in enumerate(instances)
    )
    write_tasks(output, tasks)


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
    materialize(
        TRAIN_DATASET,
        args.train_revision,
        "train",
        output / "train.parquet",
        images=images,
    )
    materialize(
        EVAL_DATASET,
        args.eval_revision,
        "test",
        output / "validation.parquet",
        images=images,
    )


if __name__ == "__main__":
    main()
