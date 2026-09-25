"""Persistent handoff for completed rollout groups.

FineStore commits each group before its ID is made visible to a trainer. The
payload is the normalized trajectory batch, including optional evidence that
cannot be reconstructed from tokens after generation.
"""

from __future__ import annotations

import io
import uuid
from collections import defaultdict
from dataclasses import dataclass
from typing import Protocol, Sequence

import pyarrow as pa
import torch
from finestore.store import DataStore
from finestore.reader import ReadView

from skyrl_train.async_rollout_state import GeneratedOutputGroup, RolloutReference
from skyrl_train.trajectory_runners.base import TrajectoryBatch


ROLLOUT_TABLE = "rollouts"
ROLLOUT_SCHEMA = pa.schema(
    [
        pa.field("rollout_id", pa.string()),
        pa.field("uid", pa.string()),
        pa.field("model_step", pa.int64()),
        pa.field("payload", pa.binary()),
    ]
)


class RolloutWriter(Protocol):
    def write_rollout(self, rollout: GeneratedOutputGroup | SynchronousRollout) -> str: ...


@dataclass
class SynchronousRollout:
    """One complete synchronous generation request before trainer admission."""

    trajectory_batch: TrajectoryBatch
    uids: list[str]
    source_prompts: list[dict]
    model_step: int
    rollout_id: str | None = None
    rollout_store_path: str | None = None


@dataclass(frozen=True)
class FineStoreRolloutWriter:
    store: DataStore

    def write_rollout(self, rollout: GeneratedOutputGroup | SynchronousRollout) -> str:
        """Commit one immutable generation result and return its stable ID."""
        rollout_id = rollout.rollout_id or uuid.uuid4().hex
        uid = rollout.uid if isinstance(rollout, GeneratedOutputGroup) else rollout.uids[0]
        model_step = rollout.earliest_model_step if isinstance(rollout, GeneratedOutputGroup) else rollout.model_step
        payload = io.BytesIO()
        torch.save(rollout, payload)
        with self.store.unbounded_transaction() as transaction:
            transaction.table(ROLLOUT_TABLE).add(
                {
                    "rollout_id": rollout_id,
                    "uid": uid,
                    "model_step": model_step,
                    "payload": payload.getvalue(),
                }
            )
        return rollout_id


class FineStoreRolloutBuffer:
    """Read and write completed groups in a run-scoped FineStore archive."""

    def __init__(self, path: str):
        self.path = path
        self.store = DataStore.open(path)
        self.store.table(ROLLOUT_TABLE, primary_key=("rollout_id",), schema=ROLLOUT_SCHEMA)

    def writer(self) -> RolloutWriter:
        return FineStoreRolloutWriter(self.store)

    def read_rollout(
        self, rollout_id: str, *, store_path: str | None = None
    ) -> GeneratedOutputGroup | SynchronousRollout:
        """Read a committed rollout by ID from a fresh FineStore snapshot."""
        view = self.store.read_view() if store_path is None or store_path == self.path else ReadView(store_path)
        row = view.point(ROLLOUT_TABLE, rollout_id=rollout_id)
        if row is None:
            raise KeyError(f"rollout {rollout_id} was not committed")
        return self._decode_rollout(row, store_path or self.path)

    def read_rollouts(self, references: Sequence[RolloutReference]) -> list[GeneratedOutputGroup]:
        """Read one bounded admission scan from a snapshot per archive."""
        by_path: dict[str, list[str]] = defaultdict(list)
        for reference in references:
            by_path[reference.store_path].append(reference.rollout_id)
        groups = {}
        for path, rollout_ids in by_path.items():
            view = self.store.read_view() if path == self.path else ReadView(path)
            for row in view.iter_rows(ROLLOUT_TABLE, where=[("rollout_id", "in", rollout_ids)]):
                rollout = self._decode_rollout(row, path)
                if not isinstance(rollout, GeneratedOutputGroup):
                    raise ValueError(f"rollout {row['rollout_id']} is not a generated output group")
                groups[path, row["rollout_id"]] = rollout
        return [groups[reference.store_path, reference.rollout_id] for reference in references]

    @staticmethod
    def _decode_rollout(row: dict, store_path: str) -> GeneratedOutputGroup | SynchronousRollout:
        rollout_id = row["rollout_id"]
        rollout = torch.load(io.BytesIO(row["payload"]), map_location="cpu", weights_only=False)
        if not isinstance(rollout, (GeneratedOutputGroup, SynchronousRollout)):
            raise ValueError(f"rollout {rollout_id} has an unexpected payload type")
        rollout.rollout_id = rollout_id
        rollout.rollout_store_path = store_path
        return rollout

    def close(self) -> None:
        self.store.close()
