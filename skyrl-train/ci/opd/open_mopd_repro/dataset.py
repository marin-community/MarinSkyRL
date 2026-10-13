"""Convert pinned Open-MOPD prompts to explicit native teacher routes."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import pyarrow as pa

from cloud.iris.open_mopd_fidelity import DOMAINS
from marinskyrl.task_format import TASK_SPEC_COLUMN
from marinskyrl.task_preparation import source_row_task
from skyrl_train.domain_sampling import ROUTE_COLUMN


def convert_rows(rows: Iterable[Mapping[str, Any]]) -> pa.Table:
    """Preserve released chat prompts and map each domain to its teacher."""
    converted = []
    for index, row in enumerate(rows):
        domain = row.get("domain")
        if domain not in DOMAINS:
            raise ValueError(f"Open-MOPD row {index} has unknown domain {domain!r}")
        prompt = row.get("prompt")
        if not isinstance(prompt, list) or not prompt:
            raise ValueError(f"Open-MOPD row {index} has no chat prompt")
        if any(
            not isinstance(message, dict)
            or not isinstance(message.get("role"), str)
            or not isinstance(message.get("content"), str)
            for message in prompt
        ):
            raise ValueError(f"Open-MOPD row {index} has invalid chat messages")
        data_source = row.get("data_source")
        if not isinstance(data_source, str) or not data_source:
            raise ValueError(f"Open-MOPD row {index} has no data_source")
        task = source_row_task(
            {
                "data_source": data_source,
                "prompt": prompt,
                "source_index": index,
            },
            index,
            source_name=data_source,
            config={},
        )
        converted.append({TASK_SPEC_COLUMN: task.model_dump_json(), ROUTE_COLUMN: domain})
    return pa.Table.from_pylist(converted)
