"""Materialize the pinned AIME 2024 prompts for native training and evaluation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import datasets
import pyarrow as pa

from marinskyrl.task_format import TASK_SPEC_COLUMN
from marinskyrl.task_preparation import source_row_task

from aime24_protocol import (
    AIME24_DATASET,
    AIME24_REVISION,
    AIME24_SIZE,
    AIME_SOURCE,
    MAX_TOKENS,
    SYSTEM_PROMPT,
    USER_INSTRUCTION,
)


def convert_rows(rows: Iterable[Mapping[str, Any]]) -> pa.Table:
    """Return bare TaskSpecs with private AIME answers and grader parameters."""
    converted = []
    for index, row in enumerate(rows):
        problem = row.get("problem")
        answer = row.get("answer")
        if not isinstance(problem, str) or not problem.strip():
            raise ValueError(f"AIME 2024 row {index} has no non-empty problem")
        try:
            normalized_answer = str(int(str(answer).strip()))
        except (TypeError, ValueError) as error:
            raise ValueError(f"AIME 2024 row {index} has an invalid answer") from error
        if not 0 <= int(normalized_answer) <= 999:
            raise ValueError(f"AIME 2024 row {index} has an answer outside [0, 999]")
        task = source_row_task(
            {
                "data_source": AIME_SOURCE,
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"{problem.strip()}\n\n{USER_INSTRUCTION}"},
                ],
                "reward_model": {"ground_truth": normalized_answer},
                "extra_info": {"source_id": str(row.get("id", index))},
            },
            index,
            source_name=AIME_SOURCE,
            config={"evaluation_token_budget": MAX_TOKENS, "strict_box_verify": True, "max_gen_length": MAX_TOKENS},
        )
        converted.append({TASK_SPEC_COLUMN: task.model_dump_json()})
    return pa.Table.from_pylist(converted)


def materialize_dataset(path: Path, row_limit: int | None = None) -> int:
    """Write the pinned AIME split and return the selected row count."""
    source = datasets.load_dataset(AIME24_DATASET, split="train", revision=AIME24_REVISION)
    if len(source) != AIME24_SIZE:
        raise RuntimeError(f"Expected {AIME24_SIZE} AIME 2024 rows, found {len(source)}")
    if row_limit is not None:
        if not 0 < row_limit <= AIME24_SIZE:
            raise ValueError(f"row_limit must be between 1 and {AIME24_SIZE}")
        source = source.select(range(row_limit))
    table = convert_rows(source[index] for index in range(len(source)))
    datasets.Dataset(table).to_parquet(path)
    return table.num_rows
