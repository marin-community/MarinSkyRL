from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
import multiprocessing
from multiprocessing.connection import Connection, wait
import threading
import time
import weakref
from typing import Any, Protocol


class PublicationOperation(StrEnum):
    INITIALIZE = "initialize"
    PUBLISH = "publish"


@dataclass(frozen=True)
class PublicationRequest:
    request_id: str
    operation: PublicationOperation
    output_path: str
    archives: Mapping[str, bytes] | None = None
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

    def submit(self, request: PublicationRequest) -> bool:
        """Submit a best-effort request, returning false when the single slot is occupied."""
        ...

    def poll(self) -> PublicationResult | None: ...

    def close(self) -> PublicationResult | None: ...


PublisherWorker = Callable[[PublicationRequest], PublicationResult]


def _publication_loop(worker: PublisherWorker, connection: Connection) -> None:
    """Keep imports and storage clients warm between acknowledged operations."""
    try:
        while True:
            request = connection.recv()
            connection.send(worker(request))
    except EOFError:
        return
    finally:
        connection.close()


def _close_storage_process(process: multiprocessing.Process, connection: Connection) -> None:
    if process.is_alive():
        process.terminate()
        process.join(timeout=1)
        if process.is_alive():
            process.kill()
    process.join(timeout=1)
    connection.close()


class ProcessTrajectoryPublisher:
    """Reuse one storage process, killing and replacing it when an operation fails."""

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
        self._execution_lock = threading.Lock()
        self._pending_thread: threading.Thread | None = None
        self._pending_event: threading.Event | None = None
        self._pending_result: PublicationResult | None = None
        self._child: multiprocessing.Process | None = None
        self._connection: Connection | None = None
        self._cleanup: weakref.finalize | None = None

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

    def close(self) -> PublicationResult | None:
        with self._lock:
            event = self._pending_event
        if event is not None and not event.wait(self._shutdown_timeout_seconds):
            self._terminate_child()
            event.wait(1)
        result = self._take_pending_result() if event is not None else None
        self._terminate_child()
        return result

    def _execute_in_background(self, request: PublicationRequest) -> None:
        result = self._execute(request)
        with self._lock:
            self._pending_result = result
            assert self._pending_event is not None
            self._pending_event.set()

    def _execute(self, request: PublicationRequest) -> PublicationResult:
        with self._execution_lock:
            return self._execute_locked(request)

    def _storage_process(self) -> tuple[multiprocessing.Process, Connection]:
        with self._lock:
            if self._child is not None and self._child.is_alive():
                assert self._connection is not None
                return self._child, self._connection
        self._terminate_child()
        context = multiprocessing.get_context("spawn")
        connection, child_connection = context.Pipe(duplex=True)
        process = context.Process(target=_publication_loop, args=(self._worker, child_connection), daemon=True)
        with self._lock:
            process.start()
            child_connection.close()
            self._child = process
            self._connection = connection
            self._cleanup = weakref.finalize(self, _close_storage_process, process, connection)
        return process, connection

    def _execute_locked(self, request: PublicationRequest) -> PublicationResult:
        deadline = time.monotonic() + self._publish_timeout_seconds
        process, connection = self._storage_process()
        send_errors = []

        def send_request() -> None:
            try:
                connection.send(request)
            except Exception as error:
                send_errors.append(error)

        # A large ledger can fill the pipe while a fresh child imports its runtime.
        # Include that wait in the operation deadline instead of blocking in send().
        sender = threading.Thread(target=send_request, name="trajectory-publication-send", daemon=True)
        sender.start()
        try:
            sender.join(max(0, deadline - time.monotonic()))
            if send_errors:
                return PublicationResult(request.request_id, request.record_count, error=str(send_errors[0]))
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._terminate_child()
                    sender.join(timeout=1)
                    return PublicationResult(
                        request_id=request.request_id,
                        record_count=request.record_count,
                        error=f"storage operation exceeded {self._publish_timeout_seconds:g} seconds",
                        timed_out=True,
                    )
                ready = wait((connection, process.sentinel), timeout=remaining)
                if connection in ready or connection.poll():
                    result = connection.recv()
                    if result.request_id != request.request_id:
                        self._terminate_child()
                        raise ValueError("storage worker returned a result for another publication")
                    if result.error is not None:
                        self._terminate_child()
                    return result
                if process.sentinel in ready:
                    process.join(timeout=1)
                    return PublicationResult(
                        request_id=request.request_id,
                        record_count=request.record_count,
                        error=f"storage worker exited with code {process.exitcode} without a result",
                    )
        except (OSError, EOFError) as error:
            self._terminate_child()
            return PublicationResult(
                request.request_id,
                request.record_count,
                error=f"storage worker disconnected without a result: {type(error).__name__}: {error}",
            )
        finally:
            if send_errors or not process.is_alive():
                self._terminate_child()

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
            cleanup = self._cleanup
            self._child = None
            self._connection = None
            self._cleanup = None
        if cleanup is not None:
            cleanup()
