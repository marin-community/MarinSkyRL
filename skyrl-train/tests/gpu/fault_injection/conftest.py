"""Command-line options for opt-in distributed fault tests."""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("distributed fault injection")
    group.addoption(
        "--node-agent-command-prefix",
        help="Command prepended to every remote Slurm node agent, such as the production container launcher.",
    )
