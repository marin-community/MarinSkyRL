"""Typed choices shared by the launcher and training runtime."""

from enum import StrEnum


class R3Transport(StrEnum):
    BY_VALUE = "by_value"
    RESIDENT = "resident"
    DECENTRAL = "decentral"


class WeightSyncTransport(StrEnum):
    AUTO = "auto"
    BROADCAST = "broadcast"
    EXPERT_BLOCK = "expert_block"


class PauseMode(StrEnum):
    """What the engines do with in-flight requests while weights are reloaded."""

    ABORT = "abort"
    KEEP = "keep"


class GDNBackend(StrEnum):
    TORCH = "torch"
    FLASHQLA = "flashqla"
