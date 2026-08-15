"""Shared pytest helpers for Iris launcher behavior tests."""

from collections.abc import Callable, Iterable
from typing import Any

import pytest
from hydra.core.override_parser.overrides_parser import OverridesParser


@pytest.fixture
def parse_hydra_overrides() -> Callable[[Iterable[str]], dict[str, Any]]:
    """Parse encoded overrides with Hydra and return their final values."""

    def parse(encoded: Iterable[str]) -> dict[str, Any]:
        parsed = OverridesParser.create().parse_overrides(list(encoded))
        return {override.key_or_group: override.value() for override in parsed}

    return parse
