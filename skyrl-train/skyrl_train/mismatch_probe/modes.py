"""Trainer settings scoped to one frozen-token probe forward."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Any, Protocol

NATIVE_MODE = "native"
REPEAT_MODE = "repeat"
REPLAY_MODE = "router_replay"
FILTERED_REPLAY_MODE = "router_replay_filtered"
RESPONSE_REPLAY_MODE = "router_replay_response"
NATIVE_AGAIN_MODE = "native_again"
REPEAT_REPLAY_MODE = "repeat_replay"
REREAD_REPLAY_MODE = "reread_replay"
REPEAT_REREAD_REPLAY_MODE = "repeat_reread_replay"


class ProbeWorker(Protocol):
    model: Any


def _native(worker: ProbeWorker, settings: Mapping[str, Any]) -> AbstractContextManager:
    return nullcontext()


def _replay(worker: ProbeWorker, settings: Mapping[str, Any]) -> AbstractContextManager:
    controller = worker.model.router_replay
    if controller is None:
        raise ValueError("router replay probe mode requires an installed controller")
    filtered = TRAINER_MODES[settings["probe_mode"]].requires_keep_fraction
    return controller.scoring_mode(FILTERED_REPLAY_MODE if filtered else REPLAY_MODE, settings["probe_keep_fraction"])


@dataclass(frozen=True)
class ModeSpec:
    context: Callable[[ProbeWorker, Mapping[str, Any]], AbstractContextManager]
    requires_routes: bool = False
    requires_keep_fraction: bool = False
    # False scores with captured routes on response positions only; prompt positions route natively.
    replays_prompt: bool = True
    # True scores the batch in reversed order with larger micro-batches, the batch-layout control.
    repeat_layout: bool = False
    # "generation" replays the routes vLLM chose while sampling; "reread" replays the routes of
    # vLLM's cache-off prefill re-read of the same tokens.
    route_source: str = "generation"


TRAINER_MODES: dict[str, ModeSpec] = {
    NATIVE_MODE: ModeSpec(_native, replays_prompt=False),
    NATIVE_AGAIN_MODE: ModeSpec(_native, replays_prompt=False),
    REPEAT_MODE: ModeSpec(_native, replays_prompt=False, repeat_layout=True),
    REPLAY_MODE: ModeSpec(_replay, requires_routes=True),
    REPEAT_REPLAY_MODE: ModeSpec(_replay, requires_routes=True, repeat_layout=True),
    REREAD_REPLAY_MODE: ModeSpec(_replay, requires_routes=True, route_source="reread"),
    REPEAT_REREAD_REPLAY_MODE: ModeSpec(_replay, requires_routes=True, repeat_layout=True, route_source="reread"),
    RESPONSE_REPLAY_MODE: ModeSpec(_replay, requires_routes=True, replays_prompt=False),
    FILTERED_REPLAY_MODE: ModeSpec(_replay, requires_routes=True, requires_keep_fraction=True),
}
