"""Convert the pinned DeepMath prompt set to MarinSkyRL's prompt-only schema."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import datasets
import pyarrow as pa

from training_plan import OPD_DATASET, OPD_DATASET_REVISION

PROMPT_ONLY_ENV = "prompt_only"


def convert_rows(rows: Iterable[Mapping[str, Any]]) -> pa.Table:
    """Return DeepMath questions as zero-reward, single-user-turn prompts."""
    converted = []
    for index, row in enumerate(rows):
        question = row.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"DeepMath row {index} has no non-empty question")
        converted.append(
            {
                "data_source": OPD_DATASET,
                "prompt": [{"role": "user", "content": question}],
                "env_class": PROMPT_ONLY_ENV,
                "extra_info": {"source_index": index},
            }
        )
    return pa.Table.from_pylist(converted)


def materialize_dataset(path: Path, row_limit: int | None = None) -> int:
    """Write the pinned DeepMath split and return its materialized row count."""
    source = datasets.load_dataset(OPD_DATASET, split="train", revision=OPD_DATASET_REVISION)
    if row_limit is not None:
        if row_limit <= 0:
            raise ValueError("row_limit must be positive")
        source = source.select(range(min(row_limit, len(source))))
    rows = (source[index] for index in range(len(source)))
    table = convert_rows(rows)
    datasets.Dataset(table).to_parquet(path)
    return table.num_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-rows", type=int)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        materialize_dataset(args.output_dir / "train.parquet", args.max_rows)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
