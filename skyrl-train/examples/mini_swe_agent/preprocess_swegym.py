"""Materialize SWE-Gym and SWE-Bench rows as executable TaskCompendium tasks."""

import argparse
from pathlib import Path

import datasets
from taskcompendium.environment import EnvironmentKind, EnvironmentSpec, RegistryImage
from taskcompendium.importers.swe import SWEInstance, swe_image, swe_task
from taskcompendium.models import Source
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


def materialize(dataset: str, revision: str, split: str, output: Path) -> None:
    rows = datasets.load_dataset(dataset, "default", revision=revision, split=split)
    instances = (SWEInstance.model_validate(row) for row in rows)
    tasks = (
        swe_task(
            row,
            source=Source(dataset=dataset, revision=revision, row=str(index), importer_revision="swe-v1"),
            environment=EnvironmentSpec(
                kind=EnvironmentKind.DOCKER,
                image=RegistryImage(reference=swe_image(row, dataset)),
                workdir="/testbed",
                env=ENVIRONMENT_VARIABLES,
                network=True,
            ),
            verifier_timeout=3600,
        )
        for index, row in enumerate(instances)
    )
    write_tasks(output, tasks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--train_revision", required=True, help="Pinned training dataset revision")
    parser.add_argument("--eval_revision", required=True, help="Pinned evaluation dataset revision")
    args = parser.parse_args()
    output = args.output_dir.expanduser()
    output.mkdir(parents=True, exist_ok=True)
    materialize(TRAIN_DATASET, args.train_revision, "train", output / "train.parquet")
    materialize(EVAL_DATASET, args.eval_revision, "test", output / "validation.parquet")


if __name__ == "__main__":
    main()
