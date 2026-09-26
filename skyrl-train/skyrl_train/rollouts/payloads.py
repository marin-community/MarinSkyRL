"""Rollout payload stores: where a written group waits between its worker's write and the trainer's read.

``MemoryPayloads`` keeps each trainable group in Ray's object store, owned by the buffer actor so it outlives the
worker that wrote it; a checkpoint copies the groups. ``FineStorePayloads`` commits each trainable group to a
FineStore archive before its verdict reaches the buffer, so the archive keeps every trainable group the run
generated, and a checkpoint records only their URIs.
"""

from __future__ import annotations

import asyncio
import pickle
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import ray
from finestore.reader import ReadView
from finestore.store import DataStore
from ray.actor import ActorHandle

from skyrl_train.rollouts.buffer import RolloutContentPolicy, RolloutGroup, RolloutLease, RolloutWriter

ROLLOUT_OBJECT_PREFIX = "rollouts/"

_stores: dict[str, DataStore] = {}
_stores_lock = threading.Lock()


class PayloadStore(Protocol):
    """How rollout payloads are written, read by the trainer, and carried through a checkpoint.

    A ``ReadyRollout.payload`` holds this store's references: at most one, and none for a group whose verdict
    excludes it from training.
    """

    @property
    def archive_root(self) -> str | None:
        """The FineStore archive that holds the payloads, or None when they live only in Ray's object store."""
        ...

    def writer(self, buffer: ActorHandle, content_policy: RolloutContentPolicy) -> RolloutWriter:
        """A picklable writer that rollout workers use to commit groups to ``buffer``."""
        ...

    async def fetch(self, payloads: Sequence) -> list[RolloutGroup]: ...

    async def checkpoint(self, payloads: Sequence) -> list:
        """The durable form of ``payloads`` for a checkpoint."""
        ...

    def restore(self, payloads: Sequence, buffer: ActorHandle) -> list:
        """References for a new ``buffer`` from a checkpoint's durable payloads."""
        ...


@dataclass(frozen=True)
class MemoryRolloutWriter:
    """Store payloads in Ray's object store, owned by the buffer actor, and commit their verdicts."""

    buffer: ActorHandle
    content_policy: RolloutContentPolicy

    async def write_rollout(self, lease: RolloutLease, group: RolloutGroup) -> None:
        verdict = self.content_policy.verdict(group)
        payload = []
        if verdict.trainable:
            payload.append(await asyncio.to_thread(ray.put, group, _owner=self.buffer))
        # Nested in a list so Ray passes the reference instead of resolving it.
        await self.buffer.commit.remote(lease.lease_id, group.prompt, verdict, payload)


class MemoryPayloads:
    """Payloads in Ray's object store; a checkpoint holds the groups themselves."""

    archive_root = None

    def writer(self, buffer: ActorHandle, content_policy: RolloutContentPolicy) -> RolloutWriter:
        return MemoryRolloutWriter(buffer, content_policy)

    async def fetch(self, payloads: Sequence) -> list[RolloutGroup]:
        return list(await asyncio.gather(*payloads))

    async def checkpoint(self, payloads: Sequence) -> list:
        return await self.fetch(payloads)

    def restore(self, payloads: Sequence, buffer: ActorHandle) -> list:
        return [ray.put(group, _owner=buffer) for group in payloads]


@dataclass(frozen=True)
class FineStoreRolloutWriter:
    """Commit each trainable group to the FineStore archive, then commit its verdict with the group's URI."""

    buffer: ActorHandle
    content_policy: RolloutContentPolicy
    archive_root: str

    async def write_rollout(self, lease: RolloutLease, group: RolloutGroup) -> None:
        verdict = self.content_policy.verdict(group)
        payload = []
        if verdict.trainable:
            payload.append(await asyncio.to_thread(_commit_group, self.archive_root, lease.lease_id, group))
        await self.buffer.commit.remote(lease.lease_id, group.prompt, verdict, payload)


@dataclass(frozen=True)
class FineStorePayloads:
    """Payloads in the FineStore archive at ``archive_root``; a checkpoint holds their URIs."""

    archive_root: str

    def writer(self, buffer: ActorHandle, content_policy: RolloutContentPolicy) -> RolloutWriter:
        return FineStoreRolloutWriter(buffer, content_policy, self.archive_root)

    async def fetch(self, payloads: Sequence) -> list[RolloutGroup]:
        return await asyncio.to_thread(_read_groups, self.archive_root, list(payloads))

    async def checkpoint(self, payloads: Sequence) -> list:
        return list(payloads)

    def restore(self, payloads: Sequence, buffer: ActorHandle) -> list:
        return list(payloads)


def _archive(root: str) -> DataStore:
    """This process's writer for the archive at ``root``, shared by every rollout it commits."""
    with _stores_lock:
        store = _stores.get(root)
        if store is None:
            store = _stores[root] = DataStore.open(root)
        return store


def _commit_group(root: str, lease_id: str, group: RolloutGroup) -> str:
    # One transaction per group: its commit returns only once this group, and no other, is durable.
    with _archive(root).unbounded_transaction() as transaction:
        uri = transaction.write_object(
            f"{ROLLOUT_OBJECT_PREFIX}{lease_id}", pickle.dumps(group, protocol=pickle.HIGHEST_PROTOCOL)
        )
    return uri


def _read_groups(root: str, uris: list[str]) -> list[RolloutGroup]:
    view = ReadView(root)
    groups = []
    for uri in uris:
        data = view.resolve(uri)
        if data is None:
            raise KeyError(f"rollout payload {uri} is not committed to the FineStore archive at {root}")
        groups.append(pickle.loads(data))
    return groups
