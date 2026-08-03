"""NCCL environment controls shared by distributed contract tests."""

from __future__ import annotations

from collections.abc import MutableMapping


_NCCL_COMMUNICATOR_NONBLOCKING_VARIABLE = "TORCH_NCCL_USE_COMM_NONBLOCKING"
_NCCL_COMMUNICATOR_TIMEOUT_VARIABLE = "TORCH_NCCL_NONBLOCKING_TIMEOUT"
_NCCL_COMMUNICATOR_NONBLOCKING_VARIABLES = (
    _NCCL_COMMUNICATOR_NONBLOCKING_VARIABLE,
    _NCCL_COMMUNICATOR_TIMEOUT_VARIABLE,
)


def nccl_communicator_nonblocking_environment(timeout_seconds: int) -> dict[str, str]:
    return {
        _NCCL_COMMUNICATOR_NONBLOCKING_VARIABLE: "1",
        _NCCL_COMMUNICATOR_TIMEOUT_VARIABLE: f"{timeout_seconds:d}",
    }


def disable_nccl_communicator_nonblocking(environment: MutableMapping[str, str]) -> None:
    for variable in _NCCL_COMMUNICATOR_NONBLOCKING_VARIABLES:
        environment.pop(variable, None)
