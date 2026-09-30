"""Trainer settings scoped to one frozen-token probe forward."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Protocol

from skyrl_train.mismatch_probe.numerics import NUMERICS_FLAGS, grug_numerics

NATIVE_MODE = "native"
REPEAT_MODE = "repeat"
REPLAY_MODE = "router_replay"
FILTERED_REPLAY_MODE = "router_replay_filtered"
RESPONSE_REPLAY_MODE = "router_replay_response"
FILTERED_RESPONSE_REPLAY_MODE = "router_replay_filtered_response"
NATIVE_AGAIN_MODE = "native_again"
REPEAT_REPLAY_MODE = "repeat_replay"
REREAD_REPLAY_MODE = "reread_replay"
NATIVE_CAPTURE_MODE = "native_capture"
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
    # True records the trainer's per-region activations at trainer.mismatch_probe.capture_layers.
    captures_layers: bool = False


TRAINER_MODES: dict[str, ModeSpec] = {
    NATIVE_MODE: ModeSpec(_native, replays_prompt=False),
    NATIVE_AGAIN_MODE: ModeSpec(_native, replays_prompt=False),
    NATIVE_CAPTURE_MODE: ModeSpec(_native, replays_prompt=False, captures_layers=True),
    REPEAT_MODE: ModeSpec(_native, replays_prompt=False, repeat_layout=True),
    REPLAY_MODE: ModeSpec(_replay, requires_routes=True),
    REPEAT_REPLAY_MODE: ModeSpec(_replay, requires_routes=True, repeat_layout=True),
    REREAD_REPLAY_MODE: ModeSpec(_replay, requires_routes=True, route_source="reread"),
    REPEAT_REREAD_REPLAY_MODE: ModeSpec(_replay, requires_routes=True, repeat_layout=True, route_source="reread"),
    RESPONSE_REPLAY_MODE: ModeSpec(_replay, requires_routes=True, replays_prompt=False),
    FILTERED_REPLAY_MODE: ModeSpec(_replay, requires_routes=True, requires_keep_fraction=True),
    FILTERED_RESPONSE_REPLAY_MODE: ModeSpec(
        _replay, requires_routes=True, requires_keep_fraction=True, replays_prompt=False
    ),
}

# Candidate numerics (see ``mismatch_probe/numerics.py``), each scored under re-read replay (the prefill
# metric) and, except ``ep_sum``, under native routing (route agreement). ``compiled_stack`` holds the
# fusion-map rounding chains without ``route_weight``, whose fp32 per-expert projection is slow;
# ``compiled_stack_all`` adds it. The ``+fa3_attention``, ``+ep_sum``, ``+vllm_gemm`` and ``+vllm_experts``
# candidates add one of vLLM's kernels or orders to a stack; ``+vllm_kernels`` adds vLLM's attention, dense
# GEMM and expert kernels together. ``vllm_kernel_stack`` is the compiled stack with ``route_weight``,
# ``fa3_attention`` and ``ep_sum``.
COMPILED_STACK = "compiled_stack"
COMPILED_STACK_ALL = "compiled_stack_all"
VLLM_KERNEL_STACK = "vllm_kernel_stack"
VLLM_KERNELS = f"{COMPILED_STACK}+vllm_kernels"
_COMPILED_STACK_FLAGS = (
    "gated_norm",
    "qk_rope",
    "xsa_gate",
    "shared_swiglu",
    "router_gemm",
    "mlp_residual",
    "input_norm_variance",
    "final_norm_fp32",
)
_KERNEL_FLAGS = ("fa3_attention", "ep_sum")
_VLLM_KERNEL_FLAGS = ("fa3_attention", "vllm_gemm", "vllm_experts")
# ``ep_sum`` needs each row's vLLM data-parallel rank, which the probe knows for replay modes only.
_NEEDS_PLACEMENT = "ep_sum"


def _enabled(*flags: str) -> dict[str, bool]:
    return dict.fromkeys(flags, True)


NUMERICS_CANDIDATES = {
    **{flag: {flag: True} for flag in NUMERICS_FLAGS},
    COMPILED_STACK: _enabled(*_COMPILED_STACK_FLAGS),
    COMPILED_STACK_ALL: _enabled(*_COMPILED_STACK_FLAGS, "route_weight"),
    f"{COMPILED_STACK}+fa3_attention": _enabled(*_COMPILED_STACK_FLAGS, "fa3_attention"),
    f"{COMPILED_STACK}+ep_sum": _enabled(*_COMPILED_STACK_FLAGS, "ep_sum"),
    f"{COMPILED_STACK_ALL}+ep_sum": _enabled(*_COMPILED_STACK_FLAGS, "route_weight", "ep_sum"),
    VLLM_KERNEL_STACK: _enabled(*_COMPILED_STACK_FLAGS, "route_weight", *_KERNEL_FLAGS),
    f"{COMPILED_STACK}+vllm_gemm": _enabled(*_COMPILED_STACK_FLAGS, "vllm_gemm"),
    f"{COMPILED_STACK}+vllm_experts": _enabled(*_COMPILED_STACK_FLAGS, "vllm_experts"),
    VLLM_KERNELS: _enabled(*_COMPILED_STACK_FLAGS, *_VLLM_KERNEL_FLAGS),
    f"{VLLM_KERNELS}+ep_sum": _enabled(*_COMPILED_STACK_FLAGS, *_VLLM_KERNEL_FLAGS, "ep_sum"),
}


def _with_numerics(base: Callable, flags: Mapping[str, bool]):
    @contextmanager
    def scope(worker: ProbeWorker, settings: Mapping[str, Any]) -> Iterator[None]:
        with grug_numerics(**flags), base(worker, settings):
            yield

    return scope


@contextmanager
def probe_mode_scope(worker: ProbeWorker, settings: Mapping[str, Any]) -> Iterator[None]:
    """Scope of ``settings["probe_mode"]`` from the trainer's own numerics, whatever the process default."""
    with (
        grug_numerics(**dict.fromkeys(NUMERICS_FLAGS, False)),
        TRAINER_MODES[settings["probe_mode"]].context(worker, settings),
    ):
        yield


# The compiled stacks on generation-route replay, comparable with the replay rows against generation.
for _candidate in (
    COMPILED_STACK,
    COMPILED_STACK_ALL,
    f"{COMPILED_STACK}+fa3_attention",
    f"{COMPILED_STACK}+ep_sum",
    VLLM_KERNEL_STACK,
    f"{COMPILED_STACK}+vllm_gemm",
    f"{COMPILED_STACK}+vllm_experts",
    VLLM_KERNELS,
    f"{VLLM_KERNELS}+ep_sum",
):
    TRAINER_MODES[f"{REPLAY_MODE}+{_candidate}"] = ModeSpec(
        _with_numerics(_replay, NUMERICS_CANDIDATES[_candidate]), requires_routes=True
    )

# The batch-layout control of a stack: the trainer against itself in reversed order with larger micro-batches,
# the re-read's routes replayed, so the prefill metric's layout floor is measured under that stack.
for _candidate in (COMPILED_STACK, f"{COMPILED_STACK}+fa3_attention", VLLM_KERNELS, f"{VLLM_KERNELS}+ep_sum"):
    TRAINER_MODES[f"{REPEAT_REREAD_REPLAY_MODE}+{_candidate}"] = ModeSpec(
        _with_numerics(_replay, NUMERICS_CANDIDATES[_candidate]),
        requires_routes=True,
        repeat_layout=True,
        route_source="reread",
    )

for _candidate, _flags in NUMERICS_CANDIDATES.items():
    TRAINER_MODES[f"{REREAD_REPLAY_MODE}+{_candidate}"] = ModeSpec(
        _with_numerics(_replay, _flags), requires_routes=True, route_source="reread"
    )
    if not _flags.get(_NEEDS_PLACEMENT):
        TRAINER_MODES[f"{NATIVE_MODE}+{_candidate}"] = ModeSpec(_with_numerics(_native, _flags), replays_prompt=False)
