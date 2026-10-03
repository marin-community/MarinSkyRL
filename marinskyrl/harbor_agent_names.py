"""Canonical Harbor agent names used at configuration boundaries."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

DEFAULT_HARBOR_AGENT_NAME = "terminus-2"
TERMINUS_KIRA_HARBOR_AGENT_NAME = "terminus-kira"
OPENCODE_HARBOR_AGENT_NAME = "opencode"
PI_HARBOR_AGENT_NAME = "pi"
MINI_SWE_HARBOR_AGENT_NAME = "mini-swe-agent"


@dataclass(frozen=True)
class HarborAgentProfile:
    name: str
    settings: Mapping[str, Any]


def configured_harbor_profiles(harbor: Mapping[str, Any]) -> tuple[HarborAgentProfile, ...]:
    """Read the ordered harness panel used for stable task-index assignment."""
    profiles = []
    for entry in harbor.get("agent_profiles") or ():
        if not isinstance(entry, Mapping) or not isinstance(entry.get("name"), str) or not entry["name"].strip():
            raise ValueError("Each Harbor agent profile requires an explicit name")
        name = entry["name"].strip().lower().replace("_", "-")
        profiles.append(HarborAgentProfile(name, {key: value for key, value in entry.items() if key != "name"}))
    if len({profile.name for profile in profiles}) != len(profiles):
        raise ValueError("Harbor agent profile names must be distinct")
    return tuple(profiles)
