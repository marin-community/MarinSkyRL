from collections.abc import Callable, Mapping
from dataclasses import dataclass
import multiprocessing
from multiprocessing.connection import Connection, wait
import threading
import time
from typing import Any, Protocol


@dataclass(frozen=True)
class PublicationRequest:
    request_id: str
    operation: str
    output_path: str
    archive_path: str | None = None
    archive_payload: bytes | None = None
    ledger: Mapping[str, Any] | None = None
    retention_config: Mapping[str, Any] | None = None
    record_count: int = 0


@dataclass(frozen=True)
class PublicationResult:
    request_id: str
    record_count: int
    ledger: dict[str, Any] | None = None
    error: str | None = None
    timed_out: bool = False


class TrajectoryPublisher(Protocol):
    def execute(self, request: PublicationRequest) -> PublicationResult: ...

    def submit(self, request: PublicationRequest) -> bool: ...

    def poll(self) -> PublicationResult | None: ...

    def wait_pending(self) -> PublicationResult | None: ...

    def close(self) -> PublicationResult | None: ...


PublisherWorker = Callable[[PublicationRequest, Connection], None]


class ProcessTrajectoryPublisher:
    """Run each storage operation in a process that can be killed at its deadline."""

    def __init__(
        self,
        worker: PublisherWorker,
        *,
        publish_timeout_seconds: float,
        shutdown_timeout_seconds: float,
    ):
        self._worker = worker
        self._publish_timeout_seconds = publish_timeout_seconds
        self._shutdown_timeout_seconds = shutdown_timeout_seconds
        self._lock = threading.Lock()
        self._pending_thread: threading.Thread | None = None
        self._pending_event: threading.Event | None = None
        self._pending_result: PublicationResult | None = None
        self._child: multiprocessing.Process | None = None

    def execute(self, request: PublicationRequest) -> PublicationResult:
        return self._execute(request)

    def submit(self, request: PublicationRequest) -> bool:
        with self._lock:
            if self._pending_thread is not None:
                return False
            self._pending_event = threading.Event()
            self._pending_result = None
            self._pending_thread = threading.Thread(
                target=self._execute_in_background,
                args=(request,),
                name=f"trajectory-publisher-{request.request_id[:12]}",
                daemon=True,
            )
            self._pending_thread.start()
            return True

    def poll(self) -> PublicationResult | None:
        with self._lock:
            event = self._pending_event
        if event is None or not event.is_set():
            return None
        return self._take_pending_result()

    def wait_pending(self) -> PublicationResult | None:
        with self._lock:
            event = self._pending_event
        if event is None:
            return None
        event.wait(self._publish_timeout_seconds + 1)
        return self._take_pending_result()

    def close(self) -> PublicationResult | None:
        with self._lock:
            event = self._pending_event
        if event is None:
            return None
        if not event.wait(self._shutdown_timeout_seconds):
            self._terminate_child()
            event.wait(1)
        return self._take_pending_result()

    def _execute_in_background(self, request: PublicationRequest) -> None:
        result = self._execute(request)
        with self._lock:
            self._pending_result = result
            assert self._pending_event is not None
            self._pending_event.set()

    def _execute(self, request: PublicationRequest) -> PublicationResult:
        context = multiprocessing.get_context("spawn")
        receiver, sender = context.Pipe(duplex=False)
        process = context.Process(target=self._worker, args=(request, sender), daemon=True)
        with self._lock:
            self._child = process
        try:
            process.start()
            sender.close()
            deadline = time.monotonic() + self._publish_timeout_seconds
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._stop_process(process)
                    return PublicationResult(
                        request_id=request.request_id,
                        record_count=request.record_count,
                        error=f"storage operation exceeded {self._publish_timeout_seconds:g} seconds",
                        timed_out=True,
                    )
                ready = wait((receiver, process.sentinel), timeout=remaining)
                if receiver in ready or receiver.poll():
                    result = receiver.recv()
                    process.join(timeout=1)
                    return result
                if process.sentinel in ready:
                    process.join(timeout=1)
                    return PublicationResult(
                        request_id=request.request_id,
                        record_count=request.record_count,
                        error=f"storage worker exited with code {process.exitcode} without a result",
                    )
        finally:
            receiver.close()
            sender.close()
            with self._lock:
                if self._child is process:
                    self._child = None

    def _take_pending_result(self) -> PublicationResult | None:
        with self._lock:
            event = self._pending_event
            if event is None or not event.is_set():
                return None
            result = self._pending_result
            thread = self._pending_thread
            self._pending_event = None
            self._pending_result = None
            self._pending_thread = None
        if thread is not None:
            thread.join(timeout=1)
        return result

    def _terminate_child(self) -> None:
        with self._lock:
            process = self._child
        if process is not None:
            self._stop_process(process)

    @staticmethod
    def _stop_process(process: multiprocessing.Process) -> None:
        if not process.is_alive():
            process.join(timeout=1)
            return
        process.terminate()
        process.join(timeout=1)
        if process.is_alive():
            process.kill()
            process.join(timeout=1)
