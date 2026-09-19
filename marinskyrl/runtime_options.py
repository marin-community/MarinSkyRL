"""Typed choices shared by the launcher and training runtime."""

from enum import StrEnum


class R3Transport(StrEnum):
    BY_VALUE = "by_value"
    RESIDENT = "resident"
    DECENTRAL = "decentral"


class GDNBackend(StrEnum):
    TORCH = "torch"
    FLASHQLA = "flashqla"


class NodeLocalPlacement(StrEnum):
    """``generator.inference_engine_node_local``: whether each stage of a DP/EP replica is packed on one node.

    ``auto`` packs when the engine can hold such a replica and its own placement group cannot leave
    nodes partly used; ``require`` packs or refuses; ``off`` never packs.
    """

    AUTO = "auto"
    REQUIRE = "require"
    OFF = "off"
