"""Backend-neutral handoff of completed rollout records to a trainer."""

from __future__ import annotations

import asyncio
import io
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Generic, Literal, Protocol, TypeVar, cast, runtime_checkable

import pyarrow as pa
import torch
from finestore.reader import ReadView
from finestore.store import DataStore

from marinskyrl.resource_locator import join_resource_path
from skyrl_train.async_rollout_state import RolloutBufferBackend, RolloutBufferSnapshot
from skyrl_train.trajectory_runners.types import TrajectoryBatch, TrajectoryRequestBatch


ROLLOUT_TABLE = "rollouts"
ROLLOUT_BUFFER_SUBDIR = "rollout_buffer"
ROLLOUT_SCHEMA = pa.schema(
    [
        pa.field("rollout_id", pa.string()),
        pa.field("uid", pa.string()),
        pa.field("model_step", pa.int64()),
        pa.field("payload", pa.binary()),
    ]
)


@dataclass
class SynchronousRollout:
    """One completed synchronous generation request and its replay evidence."""

    trajectory_batch: TrajectoryBatch
    uids: list[str]
    source_prompts: list[dict]
    model_step: int
    rollout_id: str | None = None
    request_batch: TrajectoryRequestBatch | None = None

    @property
    def rollout_uids(self) -> tuple[str, ...]:
        return tuple(self.uids)

    @property
    def rollout_model_step(self) -> int:
        return self.model_step


@dataclass(frozen=True)
class RolloutRequest:
    """One producer assignment and the record shape expected by its reader."""

    trajectory_request: TrajectoryRequestBatch
    source_prompts: list[dict]
    uids: list[str]
    model_step: int
    kind: Literal["group", "batch"]


@runtime_checkable
class Rollout(Protocol):
    trajectory_batch: TrajectoryBatch
    rollout_id: str | None
    request_batch: TrajectoryRequestBatch | None

    @property
    def rollout_uids(self) -> tuple[str, ...]: ...

    @property
    def rollout_model_step(self) -> int: ...


RolloutT = TypeVar("RolloutT", bound=Rollout)


class RolloutWriter(Protocol[RolloutT]):
    async def stage_rollout(self, rollout: RolloutT) -> RolloutReceipt: ...


@dataclass(frozen=True)
class RolloutReceipt:
    """Small completion token returned by a producer after its payload is stored."""

    rollout_id: str
    uids: tuple[str, ...]
    model_step: int
    store_path: str | None = None


class RolloutSlotPolicy(Protocol):
    async def acquire_submission_slot(self) -> None: ...

    async def on_rollout_accepted(self) -> None: ...

    async def cancel_submission_slot(self) -> None: ...


class RolloutSlot(Generic[RolloutT]):
    """Reserve capacity before generation and release it on failure or rejection."""

    def __init__(self, buffer: RolloutBuffer[RolloutT], policy: RolloutSlotPolicy | None):
        self._buffer = buffer
        self._policy = policy
        self._reserved = False

    async def __aenter__(self) -> RolloutSlot[RolloutT]:
        if self._policy is not None:
            await self._policy.acquire_submission_slot()
        self._reserved = True
        return self

    async def publish(self, receipt: RolloutReceipt) -> None:
        self._buffer.publish(receipt)
        await self._accept()

    async def _accept(self) -> None:
        if self._policy is not None:
            accepted = asyncio.create_task(self._policy.on_rollout_accepted())
            try:
                await asyncio.shield(accepted)
            except asyncio.CancelledError:
                await accepted
                self._reserved = False
                raise
        self._reserved = False

    async def __aexit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        if self._reserved and self._policy is not None:
            await self._policy.cancel_submission_slot()
        self._reserved = False


class RolloutBuffer(Protocol[RolloutT]):
    """A queue of committed rollout records; backend state stays behind this interface."""

    def writer(self) -> RolloutWriter[RolloutT]: ...

    def remote_writer(self) -> RolloutWriter[RolloutT]: ...

    def publish(self, receipt: RolloutReceipt) -> None: ...

    def request_slot(self) -> RolloutSlot[RolloutT]: ...

    async def next_batch(self, max_items: int) -> list[RolloutT]: ...

    def requeue(self, rollout: RolloutT) -> None: ...

    def snapshot(self) -> RolloutBufferSnapshot: ...

    def restore(self, snapshot: RolloutBufferSnapshot) -> None: ...

    def pending_count(self) -> int: ...

    def capacity(self) -> int: ...

    def close(self) -> None: ...

    def empty(self) -> bool: ...

    def full(self) -> bool: ...


class _MemoryWriter(Generic[RolloutT]):
    def __init__(self, buffer: MemoryRolloutBuffer[RolloutT]):
        self.buffer = buffer

    async def stage_rollout(self, rollout: RolloutT) -> RolloutReceipt:
        rollout_id = rollout.rollout_id or uuid.uuid4().hex
        rollout.rollout_id = rollout_id
        self.buffer._staged[rollout_id] = rollout
        return RolloutReceipt(rollout_id, rollout.rollout_uids, rollout.rollout_model_step)


class MemoryRolloutBuffer(Generic[RolloutT]):
    """In-process FIFO implementation, including checkpointable pending records."""

    def __init__(self, capacity: int = 0, slot_policy: RolloutSlotPolicy | None = None):
        self._pending: deque[RolloutT] = deque()
        self._staged: dict[str, RolloutT] = {}
        self._capacity = capacity
        self._slot_policy = slot_policy

    def writer(self) -> RolloutWriter[RolloutT]:
        return _MemoryWriter(self)

    def remote_writer(self) -> RolloutWriter[RolloutT]:
        raise ValueError("memory rollout buffer cannot be written from a remote producer")

    def publish(self, receipt: RolloutReceipt) -> None:
        if self.full():
            raise asyncio.QueueFull
        self.requeue(self._staged.pop(receipt.rollout_id))

    def request_slot(self) -> RolloutSlot[RolloutT]:
        return RolloutSlot(self, self._slot_policy)

    async def next_batch(self, max_items: int) -> list[RolloutT]:
        # Scan all ready groups so admission can reject stale work and retain surplus.
        result = list(self._pending)
        self._pending.clear()
        return result

    def requeue(self, rollout: RolloutT) -> None:
        if self.full():
            raise asyncio.QueueFull
        self._pending.append(rollout)

    def snapshot(self) -> RolloutBufferSnapshot:
        return RolloutBufferSnapshot(
            RolloutBufferBackend.MEMORY,
            tuple(uid for r in self._pending for uid in r.rollout_uids),
            list(self._pending),
        )

    def restore(self, snapshot: RolloutBufferSnapshot) -> None:
        if snapshot.backend != RolloutBufferBackend.MEMORY:
            raise ValueError(f"cannot restore {snapshot.backend} snapshot into memory buffer")
        records = snapshot.state
        if not isinstance(records, list):
            raise ValueError("memory buffer snapshot has invalid state")
        if self._capacity and len(records) > self._capacity:
            raise ValueError(
                f"checkpoint contains {len(records)} completed groups, exceeding buffer capacity {self._capacity}"
            )
        self._pending = deque(records)

    def pending_count(self) -> int:
        return len(self._pending)

    def capacity(self) -> int:
        return self._capacity

    def empty(self) -> bool:
        return not self._pending

    def full(self) -> bool:
        return self._capacity > 0 and len(self._pending) >= self._capacity

    def close(self) -> None:
        pass


@dataclass(frozen=True)
class _FineStorePointer:
    rollout_id: str
    uids: tuple[str, ...]
    store_path: str


class _FineStoreWriter(Generic[RolloutT]):
    def __init__(self, path: str):
        self.path = path

    async def stage_rollout(self, rollout: RolloutT) -> RolloutReceipt:
        write = asyncio.create_task(asyncio.to_thread(self._commit, rollout))
        try:
            receipt = await asyncio.shield(write)
        except asyncio.CancelledError:
            # A storage commit cannot be cancelled; wait before closing the store.
            await write
            raise
        rollout.rollout_id = receipt.rollout_id
        return receipt

    def _commit(self, rollout: RolloutT) -> RolloutReceipt:
        rollout_id = rollout.rollout_id or uuid.uuid4().hex
        payload = io.BytesIO()
        torch.save(rollout, payload)
        store = DataStore.open(self.path)
        try:
            store.table(ROLLOUT_TABLE, primary_key=("rollout_id",), schema=ROLLOUT_SCHEMA)
            with store.unbounded_transaction() as transaction:
                transaction.table(ROLLOUT_TABLE).add(
                    {
                        "rollout_id": rollout_id,
                        "uid": rollout.rollout_uids[0],
                        "model_step": rollout.rollout_model_step,
                        "payload": payload.getvalue(),
                    }
                )
        finally:
            store.close()
        return RolloutReceipt(rollout_id, rollout.rollout_uids, rollout.rollout_model_step, self.path)


class FineStoreRolloutBuffer(Generic[RolloutT]):
    """Commit payloads to FineStore and queue only private storage pointers."""

    def __init__(self, path: str, capacity: int = 0, slot_policy: RolloutSlotPolicy | None = None):
        self.path = path
        self.store = DataStore.open(path)
        self.store.table(ROLLOUT_TABLE, primary_key=("rollout_id",), schema=ROLLOUT_SCHEMA)
        # Local FineStore creates HEAD before taking its file lock. Establish it
        # once here, before independent producers can open the archive.
        if self.store.read_view().token is None:
            self.store.write_object("__rollout_buffer_initialized__", b"")
            self.store.flush()
        self._pending: deque[_FineStorePointer] = deque()
        self._scanned: dict[str, _FineStorePointer] = {}
        self._capacity = capacity
        self._slot_policy = slot_policy

    def writer(self) -> RolloutWriter[RolloutT]:
        return _FineStoreWriter(self.path)

    def remote_writer(self) -> RolloutWriter[RolloutT]:
        return self.writer()

    def publish(self, receipt: RolloutReceipt) -> None:
        if receipt.store_path != self.path:
            raise ValueError("rollout receipt belongs to a different FineStore buffer")
        self._append(_FineStorePointer(receipt.rollout_id, receipt.uids, self.path))

    def request_slot(self) -> RolloutSlot[RolloutT]:
        return RolloutSlot(self, self._slot_policy)

    def _append(self, pointer: _FineStorePointer) -> None:
        if self.full():
            raise asyncio.QueueFull
        self._pending.append(pointer)

    def _read(self, pointers: list[_FineStorePointer]) -> list[RolloutT]:
        views: dict[str, ReadView] = {}
        result = []
        for pointer in pointers:
            view = views.get(pointer.store_path)
            if view is None:
                view = self.store.read_view() if pointer.store_path == self.path else ReadView(pointer.store_path)
                views[pointer.store_path] = view
            row = view.point(ROLLOUT_TABLE, rollout_id=pointer.rollout_id)
            if row is None:
                raise KeyError(f"rollout {pointer.rollout_id} was not committed")
            rollout = torch.load(io.BytesIO(row["payload"]), map_location="cpu", weights_only=False)
            if not isinstance(rollout, Rollout):
                raise ValueError(f"rollout {pointer.rollout_id} has an unexpected payload type")
            rollout.rollout_id = pointer.rollout_id
            result.append(cast(RolloutT, rollout))
        return result

    async def next_batch(self, max_items: int) -> list[RolloutT]:
        pointers = [self._pending.popleft() for _ in range(min(max_items, len(self._pending)))]
        if not pointers:
            return []
        try:
            read = asyncio.create_task(asyncio.to_thread(self._read, pointers))
            try:
                result = await asyncio.shield(read)
            except asyncio.CancelledError:
                await read
                raise
            self._scanned = {pointer.rollout_id: pointer for pointer in pointers}
            return result
        except BaseException:
            self._pending.extendleft(reversed(pointers))
            raise

    def requeue(self, rollout: RolloutT) -> None:
        if rollout.rollout_id is None:
            raise ValueError("cannot requeue an uncommitted FineStore rollout")
        self._append(self._scanned[rollout.rollout_id])

    def snapshot(self) -> RolloutBufferSnapshot:
        return RolloutBufferSnapshot(
            RolloutBufferBackend.FINESTORE,
            tuple(uid for p in self._pending for uid in p.uids),
            list(self._pending),
        )

    def restore(self, snapshot: RolloutBufferSnapshot) -> None:
        if snapshot.backend != RolloutBufferBackend.FINESTORE:
            raise ValueError(f"cannot restore {snapshot.backend} snapshot into FineStore buffer")
        pointers = snapshot.state
        if not isinstance(pointers, list) or not all(isinstance(p, _FineStorePointer) for p in pointers):
            raise ValueError("FineStore buffer snapshot has invalid state")
        if self._capacity and len(pointers) > self._capacity:
            raise ValueError(
                f"checkpoint contains {len(pointers)} completed groups, exceeding buffer capacity {self._capacity}"
            )
        self._pending = deque(pointers)
        self._scanned.clear()

    def pending_count(self) -> int:
        return len(self._pending)

    def capacity(self) -> int:
        return self._capacity

    def empty(self) -> bool:
        return not self._pending

    def full(self) -> bool:
        return self._capacity > 0 and len(self._pending) >= self._capacity

    def close(self) -> None:
        self.store.close()


def create_rollout_buffer(
    backend: str, checkpoint_root: str, capacity: int = 0, slot_policy: RolloutSlotPolicy | None = None
) -> RolloutBuffer:
    selected = RolloutBufferBackend(backend)
    if selected is RolloutBufferBackend.MEMORY:
        return MemoryRolloutBuffer(capacity, slot_policy)
    if selected is RolloutBufferBackend.FINESTORE:
        return FineStoreRolloutBuffer(join_resource_path(checkpoint_root, ROLLOUT_BUFFER_SUBDIR), capacity, slot_policy)
    raise AssertionError(f"unhandled rollout buffer backend: {selected}")
