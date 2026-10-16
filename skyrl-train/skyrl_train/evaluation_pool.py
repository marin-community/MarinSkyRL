"""Fair, bounded admission for a fixed set of evaluation requests."""

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar

T = TypeVar("T")


@dataclass
class _Request:
    run: Callable[[], Awaitable[Any]]
    result: asyncio.Future


class EvaluationPool:
    """Replenish free slots across sources without batch or cohort barriers.

    Each request retains its own result, including failures. Source limits constrain
    expensive sandbox providers without reserving idle capacity from other sources.
    The caller owns the frozen-policy session around this pool.
    """

    def __init__(self, concurrency: int):
        if concurrency < 1:
            raise ValueError("evaluation concurrency must be positive")
        self.concurrency = concurrency
        self._queues: dict[str, deque[_Request]] = {}
        self._limits: dict[str, int] = {}
        self._active: dict[str, int] = {}
        self._sources: deque[str] = deque()
        self._changed = asyncio.Condition()
        self._workers: list[asyncio.Task] = []

    async def __aenter__(self):
        self._workers = [asyncio.create_task(self._worker()) for _ in range(self.concurrency)]
        return self

    async def __aexit__(self, *_args):
        for worker in self._workers:
            worker.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        for queue in self._queues.values():
            for request in queue:
                request.result.cancel()

    async def run(self, source: str, run: Callable[[], Awaitable[T]], *, limit: int | None = None) -> T:
        result = asyncio.get_running_loop().create_future()
        async with self._changed:
            source_limit = self.concurrency if limit is None else limit
            if source_limit < 1:
                raise ValueError("evaluation source concurrency must be positive")
            if source not in self._queues:
                self._queues[source] = deque()
                self._limits[source] = source_limit
                self._active[source] = 0
                self._sources.append(source)
            if self._limits[source] != source_limit:
                raise ValueError(f"inconsistent concurrency limit for {source}")
            self._queues[source].append(_Request(run, result))
            self._changed.notify_all()
        return await result

    def _next(self) -> tuple[str, _Request] | None:
        for _ in range(len(self._sources)):
            source = self._sources[0]
            self._sources.rotate(-1)
            queue = self._queues[source]
            while queue and queue[0].result.cancelled():
                queue.popleft()
            if queue and self._active[source] < self._limits[source]:
                self._active[source] += 1
                return source, queue.popleft()
        return None

    async def _worker(self):
        while True:
            async with self._changed:
                source, request = await self._changed.wait_for(self._next)
            try:
                value = await request.run()
            except asyncio.CancelledError:
                request.result.cancel()
                raise
            except Exception as error:
                if not request.result.done():
                    request.result.set_exception(error)
            else:
                if not request.result.done():
                    request.result.set_result(value)
            finally:
                async with self._changed:
                    self._active[source] -= 1
                    self._changed.notify_all()
