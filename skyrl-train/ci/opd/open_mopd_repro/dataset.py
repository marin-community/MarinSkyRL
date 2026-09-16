"""Convert pinned Open-MOPD prompts to explicit native teacher routes."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import pyarrow as pa

from cloud.iris.open_mopd_fidelity import DOMAINS
from skyrl_train.domain_sampling import ROUTE_COLUMN

PROMPT_ONLY_ENV = "prompt_only"


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
        converted.append(
            {
                "data_source": data_source,
                "prompt": prompt,
                "env_class": PROMPT_ONLY_ENV,
                ROUTE_COLUMN: domain,
                "source_index": index,
            }
        )
    return pa.Table.from_pylist(converted)
