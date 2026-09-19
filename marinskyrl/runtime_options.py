"""Typed choices shared by the launcher and training runtime."""

from enum import StrEnum


class R3Transport(StrEnum):
    BY_VALUE = "by_value"
    RESIDENT = "resident"
    DECENTRAL = "decentral"


class WeightSyncTransport(StrEnum):
    BROADCAST = "broadcast"
    EXPERT_BLOCK = "expert_block"


class GDNBackend(StrEnum):
    TORCH = "torch"
    FLASHQLA = "flashqla"


class NodeLocalPlacement(StrEnum):
    """Values of ``generator.inference_engine_node_local``.

    ``auto`` packs a replica when that cannot leave nodes partly used. ``require`` packs or
    refuses the run. ``off`` never packs.
    """

    AUTO = "auto"
    REQUIRE = "require"
    OFF = "off"
