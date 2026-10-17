from __future__ import annotations

from collections.abc import Callable, Iterable, Set
from typing import Any


def prefetch_items_from_tasks(
    tasks: Iterable[Any], lazy_keys: Set[str], expert_index: Callable[[str], int]
) -> list[tuple[str, int | None]]:
    """Return rank-owned HF keys and expert slices in conversion-task order."""
    return [
        (key, expert_index(task.mapping.megatron_param) if key in lazy_keys else None)
        for task in tasks
        if task.megatron_module is not None
        for key in (
            [task.mapping.hf_param] if isinstance(task.mapping.hf_param, str) else task.mapping.hf_param.values()
        )
    ]


def lazy_first_dim_patterns_for_bridge(bridge: object) -> tuple[str, ...]:
    """Return remote slice patterns from an AutoBridge's registered model bridge."""
    registered_bridge = getattr(bridge, "_model_bridge", bridge)
    return tuple(getattr(registered_bridge, "REMOTE_FIRST_DIM_SLICE_PATTERNS", ()))
