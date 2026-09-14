"""Receiver-observed proof for one named-weight installation."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator
from typing import Any, Literal, TypedDict


class WeightInstallReceipt(TypedDict):
    """Wire format returned after one vLLM worker finalizes a reload."""

    kind: Literal["weight_install_receipt"]
    finalized: bool
    received_weight_count: int
    received_name_digest: str
    loaded_parameter_count: int
    loaded_parameter_digest: str
    host: str


def weight_name_digest(names: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def flatten_install_receipts(value: Any) -> Iterator[WeightInstallReceipt]:
    if isinstance(value, dict) and value.get("kind") == "weight_install_receipt":
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from flatten_install_receipts(item)


__all__ = ["WeightInstallReceipt", "flatten_install_receipts", "weight_name_digest"]
