from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
import multiprocessing
from multiprocessing.connection import Connection, wait
import threading
import time
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


PublicationOperationRunner = Callable[[PublicationRequest], PublicationResult]


def _send_publication_result(run: PublicationOperationRunner, request: PublicationRequest, sender: Connection) -> None:
    try:
        sender.send(run(request))
    finally:
        sender.close()


def _serve_publications(run: PublicationOperationRunner, connection: Connection) -> None:
    try:
        while (request := connection.recv()) is not None:
            connection.send(run(request))
    except EOFError:
        pass
    finally:
        connection.close()


class InlineTrajectoryPublisher:
    """Run each storage operation synchronously in the calling thread, with no deadline.

    A best-effort submission completes before ``submit`` returns, so it suits storage that cannot hang, such as a
    local filesystem.
    """

    def __init__(self, run: PublicationOperationRunner):
        self._run = run
        self._pending_result: PublicationResult | None = None

    def execute(self, request: PublicationRequest) -> PublicationResult:
        return self._run(request)

    def submit(self, request: PublicationRequest) -> bool:
        if self._pending_result is not None:
            return False
        self._pending_result = self._run(request)
        return True

    def poll(self) -> PublicationResult | None:
        result, self._pending_result = self._pending_result, None
        return result

    def close(self) -> PublicationResult | None:
        return self.poll()


class ProcessTrajectoryPublisher:
    """Run each storage operation in a process that can be killed at its deadline."""

    def __init__(
        self,
        run: PublicationOperationRunner,
        *,
        publish_timeout_seconds: float,
        shutdown_timeout_seconds: float,
        reuse_process: bool = False,
    ):
        self._run = run
        self._publish_timeout_seconds = publish_timeout_seconds
        self._shutdown_timeout_seconds = shutdown_timeout_seconds
        self._lock = threading.Lock()
        self._pending_thread: threading.Thread | None = None
        self._pending_event: threading.Event | None = None
        self._pending_result: PublicationResult | None = None
        self._child: multiprocessing.Process | None = None
        self._connection: Connection | None = None
        self._reuse_process = reuse_process
        self._operation_lock = threading.Lock()

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
        if event is None:
            self._terminate_child()
            return None
        if not event.wait(self._shutdown_timeout_seconds):
            self._terminate_child()
            event.wait(1)
        result = self._take_pending_result()
        self._terminate_child()
        return result

    def _execute_in_background(self, request: PublicationRequest) -> None:
        result = self._execute(request)
        with self._lock:
            self._pending_result = result
            assert self._pending_event is not None
            self._pending_event.set()

    def _execute(self, request: PublicationRequest) -> PublicationResult:
        if self._reuse_process:
            with self._operation_lock:
                return self._execute_reused(request)
        context = multiprocessing.get_context("spawn")
        receiver, sender = context.Pipe(duplex=False)
        process = context.Process(target=_send_publication_result, args=(self._run, request, sender), daemon=True)
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
        except (EOFError, BrokenPipeError, OSError) as error:
            self._stop_process(process)
            return PublicationResult(request.request_id, request.record_count, error=str(error))
        finally:
            receiver.close()
            sender.close()
            with self._lock:
                if self._child is process:
                    self._child = None

    def _execute_reused(self, request: PublicationRequest) -> PublicationResult:
        context = multiprocessing.get_context("spawn")
        with self._lock:
            process, connection = self._child, self._connection
        if process is None or not process.is_alive():
            self._terminate_child()
            connection, child_connection = context.Pipe()
            process = context.Process(target=_serve_publications, args=(self._run, child_connection), daemon=True)
            with self._lock:
                self._child, self._connection = process, connection
            process.start()
            child_connection.close()
        assert connection is not None
        deadline = time.monotonic() + self._publish_timeout_seconds

        # A large archive can fill the pipe before the worker finishes importing.
        # Keep that write off this thread so the publication deadline still applies.
        def send_request() -> None:
            try:
                connection.send(request)
            except (EOFError, BrokenPipeError, OSError):
                pass  # Worker exit or the deadline closes the pipe below.

        sender = threading.Thread(target=send_request, daemon=True)
        try:
            sender.start()
            remaining = max(0, deadline - time.monotonic())
            ready = wait((connection, process.sentinel), timeout=remaining)
            if connection in ready and connection.poll():
                result = connection.recv()
                assert result.request_id == request.request_id
                return result
            if process.sentinel in ready:
                self._terminate_child()
                return PublicationResult(
                    request.request_id, request.record_count, error="storage worker exited without a result"
                )
            self._terminate_child()
            return PublicationResult(
                request.request_id,
                request.record_count,
                error=f"storage operation exceeded {self._publish_timeout_seconds:g} seconds",
                timed_out=True,
            )
        except (EOFError, BrokenPipeError, OSError) as error:
            self._terminate_child()
            return PublicationResult(request.request_id, request.record_count, error=str(error))
        finally:
            sender.join(timeout=1)

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
            connection = self._connection
            self._child, self._connection = None, None
        if process is not None:
            self._stop_process(process)
        if connection is not None:
            connection.close()

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
