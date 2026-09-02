"""No-op Rigging surface for installations without the telemetry extra."""

from collections.abc import Mapping
from dataclasses import dataclass


class _Instrument:
    def add(self, value: float = 1.0, *, attributes: Mapping[str, str] | None = None) -> None:
        pass

    def set(self, value: float, *, attributes: Mapping[str, str] | None = None) -> None:
        pass

    def record(self, value: float, *, attributes: Mapping[str, str] | None = None) -> None:
        pass


@dataclass(frozen=True)
class _Status:
    configured: bool = False
    queued_records: int = 0
    lost_records: int = 0


_instrument = _Instrument()
_status = _Status()


def counter(name: str, *, unit: str = "") -> _Instrument:
    return _instrument


def gauge(name: str, *, unit: str = "") -> _Instrument:
    return _instrument


def histogram(name: str, *, unit: str = "") -> _Instrument:
    return _instrument


def configure(*, endpoint: str, service: str, attributes: Mapping[str, str]) -> None:
    pass


def event(name: str, body: object, *, attributes: Mapping[str, str] | None = None) -> None:
    """Signature mirrors rigging's ``event(name, body: EventBody, *, attributes)``.

    It previously named the second parameter ``fields: Mapping[str, object]``, which type-checked a
    caller passing a bare dict and hid a real ``AttributeError`` on the live path.
    """


def runtime_status() -> _Status:
    return _status


def flush(timeout: float = 5.0) -> bool:
    """Nothing is queued, so the queue is trivially settled."""
    return True


def shutdown(timeout: float = 5.0) -> None:
    pass
