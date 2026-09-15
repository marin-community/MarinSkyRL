"""Convert the pinned DeepMath prompt set to MarinSkyRL's prompt-only schema."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import datasets
import pyarrow as pa

DATASET = "zwhe99/DeepMath-103K"
DATASET_REVISION = "5cf055d1fe3d7a2eb19719ac020211469736ae44"


def convert_rows(rows: Iterable[Mapping[str, Any]]) -> pa.Table:
    """Return DeepMath questions as zero-reward, single-user-turn prompts."""
    converted = []
    for index, row in enumerate(rows):
        question = row.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"DeepMath row {index} has no non-empty question")
        converted.append(
            {
                "data_source": DATASET,
                "prompt": [{"role": "user", "content": question}],
                "env_class": "prompt_only",
                "extra_info": {"source_index": index},
            }
        )
    return pa.Table.from_pylist(converted)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-rows", type=int)
    args = parser.parse_args()

    source = datasets.load_dataset(DATASET, split="train", revision=DATASET_REVISION)
    if args.max_rows is not None:
        if args.max_rows <= 0:
            parser.error("--max-rows must be positive")
        source = source.select(range(min(args.max_rows, len(source))))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = (source[index] for index in range(len(source)))
    datasets.Dataset(convert_rows(rows)).to_parquet(args.output_dir / "train.parquet")


if __name__ == "__main__":
    main()
