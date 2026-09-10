"""Bounded rollout load with raw snapshots and compact request receipts.

This module does no model or sampling configuration. Callers provide the same
InferenceEngineClient and inputs that they want to measure. Artifact capture is
explicit; ordinary training does not instantiate this harness.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from skyrl_train.inference_engines.vllm.stats import InferenceStatsSnapshot, IntervalReadMode


def json_value(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return json_value(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "+Inf" if value == math.inf else str(value)
    return value


@dataclass(frozen=True)
class BenchmarkRequest:
    request_id: str
    group_id: str
    repetition_id: int
    session_id: str
    prompt_token_ids: list[int]


class InferenceBenchmark:
    def __init__(self, client, artifact_dir: Path, *, sinks=(), poll_seconds: float = 5.0):
        self.client = client
        self.artifact_dir = artifact_dir
        artifact_dir.mkdir(parents=True, exist_ok=True)
        # Exclusive creation prevents accidentally mixing two benchmark runs.
        self._file = (artifact_dir / "events.jsonl").open("x")
        self.sinks = sinks
        self.poll_seconds = poll_seconds
        self._poll_lock = asyncio.Lock()
        self._stop = asyncio.Event()
        self._poll_task = None
        self._engine_ids = None
        self._treatment = None

    def record(self, event: str, **fields) -> None:
        self._file.write(json.dumps(json_value({"event": event, "at": time.time(), **fields})) + "\n")
        self._file.flush()

    async def snapshot(self, boundary: str, *, publish: bool = False):
        async with self._poll_lock:
            started_at = time.time()
            started_mono = time.perf_counter()

            async def read(index, engine):
                before = time.time()
                before_mono = time.perf_counter()
                snapshot = await engine.get_stats(IntervalReadMode.PEEK)
                return {
                    "engine_index": index,
                    "poll_started_at": before,
                    "poll_finished_at": time.time(),
                    "poll_started_monotonic": before_mono,
                    "poll_finished_monotonic": time.perf_counter(),
                    "snapshot": snapshot,
                }

            try:
                readings = await asyncio.gather(*(read(i, engine) for i, engine in enumerate(self.client.engines)))
                snapshots = tuple(item["snapshot"] for item in readings)
                identities = tuple(snapshot.engine_id for snapshot in snapshots)
                if len(set(identities)) != len(self.client.engines):
                    raise RuntimeError(f"Incomplete or duplicate producer identities: {identities}")
                if self._engine_ids is None:
                    self._engine_ids = identities
                elif identities != self._engine_ids:
                    raise RuntimeError(f"Producer identities changed: {identities}")
            except Exception as exc:
                self.record("collection_error", treatment=self._treatment, boundary=boundary, error=repr(exc))
                raise
            collection_seconds = time.perf_counter() - started_mono
            publish_started = time.perf_counter()
            if publish:
                for sink in self.sinks:
                    sink.publish(InferenceStatsSnapshot(snapshots), step=0)
            self.record(
                "snapshot",
                treatment=self._treatment,
                boundary=boundary,
                poll_started_at=started_at,
                poll_finished_at=time.time(),
                collection_seconds=collection_seconds,
                publication_seconds=time.perf_counter() - publish_started,
                published=publish,
                readings=readings,
            )
            return snapshots

    async def _poll(self):
        while not self._stop.is_set():
            before = time.perf_counter()
            await self.snapshot("periodic", publish=True)
            delay = max(0.0, self.poll_seconds - (time.perf_counter() - before))
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=delay)
            except TimeoutError:
                pass

    async def __aenter__(self):
        try:
            await self.snapshot("initial")
        except BaseException:
            self._file.close()
            raise
        self._poll_task = asyncio.create_task(self._poll())
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self._stop.set()
        try:
            if self._poll_task is not None:
                await self._poll_task
            await self.snapshot("final")
        finally:
            self._file.close()

    async def run(
        self,
        name: str,
        requests: Sequence[BenchmarkRequest],
        *,
        concurrency: int,
        mode: Literal["burst", "refill"],
        sampling_params: dict[str, Any],
    ) -> dict[str, Any]:
        if concurrency < 1 or mode not in ("burst", "refill"):
            raise ValueError("Expected positive concurrency and burst/refill mode")
        if len({request.request_id for request in requests}) != len(requests):
            raise ValueError("Request identities must be unique within a treatment")
        if self._poll_task is not None and self._poll_task.done():
            await self._poll_task
        self._treatment = name
        await self.snapshot("before")
        self.record("treatment_start", treatment=name, mode=mode, concurrency=concurrency, requests=len(requests))
        started = time.perf_counter()
        receipts = []

        async def generate(request):
            submitted_at = time.time()
            submitted_mono = time.perf_counter()
            self.record("submission", treatment=name, request_id=request.request_id, submitted_at=submitted_at)
            try:
                output = await self.client.generate(
                    {
                        "prompt_token_ids": [request.prompt_token_ids],
                        "session_ids": [request.session_id],
                        "sampling_params": sampling_params,
                    }
                )
            except BaseException as exc:
                self.record("request_error", treatment=name, request_id=request.request_id, error=repr(exc))
                raise
            receipt = {
                "treatment": name,
                "request_id": request.request_id,
                "group_id": request.group_id,
                "repetition_id": request.repetition_id,
                "session_id": request.session_id,
                "submitted_at": submitted_at,
                "completed_at": time.time(),
                "latency_seconds": time.perf_counter() - submitted_mono,
                "prompt_tokens": len(request.prompt_token_ids),
                "output_tokens": len(output["response_ids"][0]),
                "finish_reason": output["stop_reasons"][0],
                "attempts": output.get("request_timings", [[]])[0],
            }
            self.record("completion", **receipt)
            receipts.append(receipt)

        async def joined(batch):
            tasks = [asyncio.create_task(generate(request)) for request in batch]
            try:
                await asyncio.gather(*tasks)
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

        if mode == "burst":
            for offset in range(0, len(requests), concurrency):
                wave_started = time.perf_counter()
                await joined(requests[offset : offset + concurrency])
                self.record("wave_end", treatment=name, offset=offset, seconds=time.perf_counter() - wave_started)
        else:
            queue = iter(requests)

            async def worker():
                for request in queue:
                    await generate(request)

            tasks = [asyncio.create_task(worker()) for _ in range(min(concurrency, len(requests)))]
            try:
                await asyncio.gather(*tasks)
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = time.perf_counter() - started
        self.record("treatment_end", treatment=name, seconds=elapsed, completed=len(receipts))
        await self.snapshot("after")
        self._treatment = None
        return {"name": name, "seconds": elapsed, "requests": receipts}
