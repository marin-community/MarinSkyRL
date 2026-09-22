"""Typed choices shared by the launcher and training runtime."""

from enum import StrEnum


class R3Transport(StrEnum):
    BY_VALUE = "by_value"
    RESIDENT = "resident"
    DECENTRAL = "decentral"


class GDNBackend(StrEnum):
    TORCH = "torch"
    FLASHQLA = "flashqla"
