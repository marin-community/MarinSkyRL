"""Bounded, representation-preserving observability for async rollout retention."""

from __future__ import annotations

import asyncio
import itertools
import math
import sys
import threading
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Callable


_KNOWN_FIELDS = frozenset(
    {
        "prompt_token_ids",
        "response_ids",
        "rewards",
        "unshaped_rewards",
        "unshaped_reward_available",
        "reward_shaping_components",
        "reward_shaping_loop_spans",
        "loop_advantages",
        "reward_shaping_versions",
        "verifier_tests",
        "loss_masks",
        "stop_reasons",
        "exception_types",
        "error_treatments",
        "rollout_metrics",
        "rollout_logprobs",
        "student_topk_indices",
        "behavior_topk_logprobs",
        "rollout_routed_experts",
        "teacher_evidence",
        "distillation",
        "token_level_shaping",
        "response_span_tags",
        "trajectory_ids",
        "teacher_route_keys",
        "is_last_step",
        "exclude_from_baseline",
        "actual_global_step",
        "source_prompts",
        "other",
    }
)


@dataclass(frozen=True)
class PayloadEstimate:
    """Primitive-only estimate of one retained group's logical Python payload."""

    rows: int
    total_bytes: int
    field_bytes: tuple[tuple[str, int], ...]
    sampled_nodes: int
    truncated: bool


@dataclass(frozen=True)
class RetentionSnapshot:
    """Primitive-only ownership snapshot safe to retain or export."""

    surface: str
    groups: dict[str, int]
    rows: dict[str, int]
    estimated_bytes: dict[str, int]
    field_estimated_bytes: dict[tuple[str, str], int]
    producers: dict[str, int]
    settled_groups: int
    settled_rows: int


@dataclass(frozen=True)
class _RetainedEstimate:
    owner: str
    estimate: PayloadEstimate
    token_id: int | None = None


@dataclass
class _Budget:
    remaining: int
    sampled: int = 0

    def take(self) -> bool:
        if self.remaining <= 0:
            return False
        self.remaining -= 1
        self.sampled += 1
        return True


def _evenly_spaced_indices(length: int, count: int) -> range | list[int]:
    if count >= length:
        return range(length)
    if count == 1:
        return [length // 2]
    return [round(index * (length - 1) / (count - 1)) for index in range(count)]


def _estimate_value(value: object, budget: _Budget) -> tuple[int, bool]:
    """Estimate logical bytes without mutating or retaining ``value``.

    Large containers are sampled at evenly spaced positions and extrapolated. The
    estimate is intentionally approximate; its purpose is ownership attribution and
    trend comparison, not heap accounting.
    """

    if not budget.take():
        return 0, True

    base_size = sys.getsizeof(value)
    if value is None or isinstance(value, (bool, int, float, complex, str, bytes, bytearray)):
        return base_size, False

    numel = getattr(value, "numel", None)
    element_size = getattr(value, "element_size", None)
    if callable(numel) and callable(element_size):
        try:
            return base_size + int(numel()) * int(element_size()), False
        except (TypeError, ValueError, RuntimeError):
            pass

    nbytes = getattr(value, "nbytes", None)
    if isinstance(nbytes, int):
        return base_size + nbytes, False

    if isinstance(value, Mapping):
        length = len(value)
        if not length:
            return base_size, False
        sample_count = min(length, 8, max(1, budget.remaining // 2))
        sampled_size = 0
        truncated = sample_count < length
        for key, item in itertools.islice(value.items(), sample_count):
            key_size, key_truncated = _estimate_value(key, budget)
            item_size, item_truncated = _estimate_value(item, budget)
            sampled_size += key_size + item_size
            truncated = truncated or key_truncated or item_truncated
        return base_size + math.ceil(sampled_size * length / sample_count), truncated

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        length = len(value)
        if not length:
            return base_size, False
        sample_count = min(length, 8, max(1, budget.remaining))
        sampled_size = 0
        truncated = sample_count < length
        for index in _evenly_spaced_indices(length, sample_count):
            item_size, item_truncated = _estimate_value(value[index], budget)
            sampled_size += item_size
            truncated = truncated or item_truncated
        return base_size + math.ceil(sampled_size * length / sample_count), truncated

    return base_size, False


def estimate_group_payload(
    trajectory_batch: Mapping[str, object],
    source_prompts: Sequence[Mapping[str, object]],
    *,
    max_nodes_per_field: int = 256,
) -> PayloadEstimate:
    """Return a bounded logical-size estimate without changing either input."""

    if max_nodes_per_field <= 0:
        raise ValueError("max_nodes_per_field must be positive")

    field_sizes: Counter[str] = Counter()
    sampled_nodes = 0
    truncated = False
    for raw_name, value in trajectory_batch.items():
        name = raw_name if raw_name in _KNOWN_FIELDS else "other"
        budget = _Budget(max_nodes_per_field)
        size, field_truncated = _estimate_value(value, budget)
        field_sizes[name] += size
        sampled_nodes += budget.sampled
        truncated = truncated or field_truncated

    source_budget = _Budget(max_nodes_per_field)
    source_size, source_truncated = _estimate_value(source_prompts, source_budget)
    field_sizes["source_prompts"] += source_size
    sampled_nodes += source_budget.sampled
    truncated = truncated or source_truncated

    response_ids = trajectory_batch.get("response_ids")
    rows = len(response_ids) if isinstance(response_ids, Sequence) else 0
    return PayloadEstimate(
        rows=rows,
        total_bytes=sum(field_sizes.values()),
        field_bytes=tuple(sorted(field_sizes.items())),
        sampled_nodes=sampled_nodes,
        truncated=truncated,
    )


def estimate_agent_loop_payload(output: object, *, max_nodes_per_field: int = 64) -> PayloadEstimate:
    """Estimate one pre-projection ``AgentLoopOutput`` without retaining it.

    The synchronous Snowball path fans out thousands of these objects through
    ``tqdm.gather``.  Keeping the estimate primitive-only lets us measure that
    actual retention seam without adding another reference to the payload.
    """

    if max_nodes_per_field <= 0:
        raise ValueError("max_nodes_per_field must be positive")

    evidence = getattr(output, "evidence", None)
    fields = {
        "prompt_token_ids": getattr(evidence, "prompt_token_ids", None),
        "response_ids": getattr(evidence, "response_token_ids", None),
        "rollout_logprobs": getattr(evidence, "behavior_logprobs", None),
        "student_topk_indices": getattr(evidence, "student_topk_indices", None),
        "behavior_topk_logprobs": getattr(evidence, "behavior_topk_logprobs", None),
        "rollout_routed_experts": getattr(evidence, "routed_experts", None),
        "source_prompts": getattr(evidence, "messages", None),
        "loss_masks": getattr(output, "loss_mask", None),
        "other": (
            getattr(evidence, "response", None),
            getattr(output, "verification", None),
            getattr(output, "reward", None),
            getattr(output, "disposition", None),
            getattr(output, "env_metrics", None),
        ),
    }
    field_sizes: Counter[str] = Counter()
    sampled_nodes = 0
    truncated = False
    for name, value in fields.items():
        budget = _Budget(max_nodes_per_field)
        size, field_truncated = _estimate_value(value, budget)
        field_sizes[name] += size
        sampled_nodes += budget.sampled
        truncated = truncated or field_truncated
    return PayloadEstimate(
        rows=1,
        total_bytes=sum(field_sizes.values()),
        field_bytes=tuple(sorted(field_sizes.items())),
        sampled_nodes=sampled_nodes,
        truncated=truncated,
    )


def _default_publish(snapshot: RetentionSnapshot, boundary: str | None) -> None:
    from skyrl_train.telemetry import record_generation_retention_snapshot

    record_generation_retention_snapshot(
        surface=snapshot.surface,
        groups=snapshot.groups,
        rows=snapshot.rows,
        estimated_bytes=snapshot.estimated_bytes,
        field_estimated_bytes=snapshot.field_estimated_bytes,
        producers=snapshot.producers,
        settled_groups=snapshot.settled_groups,
        settled_rows=snapshot.settled_rows,
        boundary=boundary,
    )


class GenerationRetentionObserver:
    """Track primitive ownership estimates while never retaining rollout objects."""

    def __init__(
        self,
        *,
        publish_interval_seconds: float = 10.0,
        max_nodes_per_field: int = 256,
        publish: Callable[[RetentionSnapshot, str | None], None] = _default_publish,
        monotonic: Callable[[], float] = time.monotonic,
        surface: str = "fully_async_queue",
    ) -> None:
        if publish_interval_seconds <= 0:
            raise ValueError("publish_interval_seconds must be positive")
        self._publish_interval_seconds = publish_interval_seconds
        self._max_nodes_per_field = max_nodes_per_field
        self._publish_callback = publish
        self._monotonic = monotonic
        self._surface = surface
        self._lock = threading.Lock()
        self._retained: dict[int, _RetainedEstimate] = {}
        self._producer_states: dict[int, str] = {}
        self._settled_groups = 0
        self._settled_rows = 0
        self._last_published = float("-inf")
        self._stop_event: asyncio.Event | None = None
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._active_leases = 0

    def start(self) -> None:
        """Acquire a heartbeat lease for one overlapping collection."""

        self._active_leases += 1
        if self._active_leases > 1:
            return
        stop_event = asyncio.Event()
        self._stop_event = stop_event
        self._heartbeat_task = asyncio.create_task(self._heartbeat(stop_event))
        self.publish(force=True, boundary="observer_started")

    async def stop(self) -> None:
        """Release a heartbeat lease and stop after the last collection."""

        if self._active_leases == 0:
            return
        self._active_leases -= 1
        if self._active_leases > 0:
            return
        task = self._heartbeat_task
        stop_event = self._stop_event
        if task is None or stop_event is None:
            return
        stop_event.set()
        await task
        self.publish(force=True, boundary="observer_stopped")
        if self._heartbeat_task is task:
            self._heartbeat_task = None
            self._stop_event = None

    async def _heartbeat(self, stop_event: asyncio.Event) -> None:
        while not stop_event.is_set():
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=self._publish_interval_seconds)
            except asyncio.TimeoutError:
                self.publish(force=True)

    def producer_started(self, producer: object, *, state: str = "waiting_input") -> None:
        with self._lock:
            self._producer_states[id(producer)] = state
        self.publish()

    def producer_state(self, producer: object, state: str) -> None:
        with self._lock:
            producer_id = id(producer)
            if producer_id not in self._producer_states:
                raise KeyError("producer was not registered")
            self._producer_states[producer_id] = state
        self.publish()

    def producer_finished(self, producer: object) -> None:
        with self._lock:
            self._producer_states.pop(id(producer), None)
        self.publish()

    def register_group(
        self,
        group: object,
        *,
        trajectory_batch: Mapping[str, object],
        source_prompts: Sequence[Mapping[str, object]],
        owner: str,
        token: object | None = None,
    ) -> PayloadEstimate:
        estimate = estimate_group_payload(
            trajectory_batch,
            source_prompts,
            max_nodes_per_field=self._max_nodes_per_field,
        )
        with self._lock:
            group_id = id(group)
            if group_id in self._retained:
                raise ValueError("group is already registered")
            self._retained[group_id] = _RetainedEstimate(
                owner=owner, estimate=estimate, token_id=None if token is None else id(token)
            )
            self._settled_groups += 1
            self._settled_rows += estimate.rows
        self.publish()
        return estimate

    def register_estimate(
        self, payload: object, *, estimate: PayloadEstimate, owner: str, token: object | None = None
    ) -> None:
        """Track a primitive estimate while deliberately not retaining ``payload``."""

        with self._lock:
            payload_id = id(payload)
            if payload_id in self._retained:
                raise ValueError("payload is already registered")
            self._retained[payload_id] = _RetainedEstimate(
                owner=owner, estimate=estimate, token_id=None if token is None else id(token)
            )
            self._settled_groups += 1
            self._settled_rows += estimate.rows
        self.publish()

    def transfer_group(self, group: object, *, owner: str) -> None:
        with self._lock:
            group_id = id(group)
            retained = self._retained.get(group_id)
            if retained is None:
                raise KeyError("group was not registered")
            self._retained[group_id] = _RetainedEstimate(
                owner=owner, estimate=retained.estimate, token_id=retained.token_id
            )
        self.publish()

    def release_group(self, group: object) -> None:
        with self._lock:
            self._retained.pop(id(group), None)
        self.publish()

    def release_token(self, token: object) -> None:
        """Release only one call's estimates, preserving overlapping calls."""
        token_id = id(token)
        with self._lock:
            self._retained = {
                payload_id: retained for payload_id, retained in self._retained.items() if retained.token_id != token_id
            }
        self.publish()

    def snapshot(self) -> RetentionSnapshot:
        with self._lock:
            groups: Counter[str] = Counter()
            rows: Counter[str] = Counter()
            estimated_bytes: Counter[str] = Counter()
            field_estimated_bytes: Counter[tuple[str, str]] = Counter()
            for retained in self._retained.values():
                owner = retained.owner
                estimate = retained.estimate
                groups[owner] += 1
                rows[owner] += estimate.rows
                estimated_bytes[owner] += estimate.total_bytes
                for field_name, field_bytes in estimate.field_bytes:
                    field_estimated_bytes[(owner, field_name)] += field_bytes
            producers = Counter(self._producer_states.values())
            return RetentionSnapshot(
                surface=self._surface,
                groups=dict(groups),
                rows=dict(rows),
                estimated_bytes=dict(estimated_bytes),
                field_estimated_bytes=dict(field_estimated_bytes),
                producers=dict(producers),
                settled_groups=self._settled_groups,
                settled_rows=self._settled_rows,
            )

    def publish(self, *, force: bool = False, boundary: str | None = None) -> None:
        now = self._monotonic()
        with self._lock:
            if not force and now - self._last_published < self._publish_interval_seconds:
                return
            self._last_published = now
        self._publish_callback(self.snapshot(), boundary)
