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
    # True scores each sequence as the logged vLLM step of its re-read computed it (``vllm_steps``).
    needs_step_plan: bool = False


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
# ``compiled_stack_all`` adds it. The ``+fa3_attention`` and ``+ep_sum`` candidates add one of vLLM's
# kernel orders to a stack, and ``vllm_kernel_stack`` is the compiled stack with ``route_weight``,
# ``fa3_attention`` and ``ep_sum``. ``KEPT_STACK`` (``compiled_stack+fa3_attention``) is the kept numerics;
# its ``+vllm_gemm`` and ``+vllm_experts`` candidates add vLLM's dense GEMM shapes and expert kernels, and
# ``+router_rows`` computes the router GEMM at a full vLLM prefill step's row count. ``KEPT_STACK_2`` (those three on
# ``KEPT_STACK``) is the kept numerics since J4; its ``+vllm_xsa``, ``+vllm_qk``, ``+vllm_norms``, ``+vllm_swiglu``
# and ``+vllm_log_softmax`` candidates take the XSA, q/k, norm, shared-activation and log-softmax values from compiled
# vLLM's own kernels, ``VLLM_FORWARD`` adds all five, and ``VLLM_FORWARD_EP_SUM`` adds vLLM's expert-parallel
# addition order as well. ``KEPT_STACK_3`` replaces ``router_rows`` with ``vllm_steps``: each sequence is scored as the
# logged vLLM step of the re-read computed it (FA3 split counts, router GEMM rows, LM-head rows), so its candidates
# (``STEP_CANDIDATES``) score re-read replay only; ``VLLM_STEP_FORWARD_EP_SUM`` is ``VLLM_FORWARD_EP_SUM`` so scored.
# ``INVARIANT_STACK`` is the trainer side of a decode-invariant vLLM engine (``generator.decode_invariant``): the
# router GEMM is the engine's row-invariant kernel and sliding-window rows past the window are one-row FA3 requests, so
# no logged step is needed and generation-route replay scores it too; ``VLLM_INVARIANT_FORWARD`` adds the region flags
# and ``ep_sum``.
COMPILED_STACK = "compiled_stack"
COMPILED_STACK_ALL = "compiled_stack_all"
VLLM_KERNEL_STACK = "vllm_kernel_stack"
KEPT_STACK = f"{COMPILED_STACK}+fa3_attention"
KEPT_VLLM_GEMM = f"{KEPT_STACK}+vllm_gemm"
KEPT_VLLM_EXPERTS = f"{KEPT_STACK}+vllm_experts"
KEPT_VLLM_KERNELS = f"{KEPT_STACK}+vllm_gemm+vllm_experts"
KEPT_VLLM_KERNELS_ROUTER_ROWS = f"{KEPT_VLLM_KERNELS}+router_rows"
KEPT_STACK_2 = KEPT_VLLM_KERNELS_ROUTER_ROWS
_VLLM_REGION_FLAGS = ("vllm_xsa", "vllm_qk", "vllm_norms", "vllm_swiglu", "vllm_log_softmax")
VLLM_FORWARD = "+".join((KEPT_STACK_2, *_VLLM_REGION_FLAGS))
VLLM_FORWARD_EP_SUM = f"{VLLM_FORWARD}+ep_sum"
KEPT_STACK_3 = f"{KEPT_VLLM_KERNELS}+vllm_steps"
VLLM_STEP_FORWARD = "+".join((KEPT_STACK_3, *_VLLM_REGION_FLAGS))
VLLM_STEP_FORWARD_EP_SUM = f"{VLLM_STEP_FORWARD}+ep_sum"
_INVARIANT_FLAGS = ("fa3_attention", "vllm_gemm", "vllm_experts", "invariant_router", "fa3_window_rows")
INVARIANT_STACK = f"{KEPT_VLLM_KERNELS}+invariant_router+fa3_window_rows"
VLLM_INVARIANT_FORWARD = "+".join((INVARIANT_STACK, *_VLLM_REGION_FLAGS, "ep_sum"))
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
_KEPT_2_FLAGS = ("fa3_attention", "vllm_gemm", "vllm_experts", "router_rows")
_KEPT_3_FLAGS = ("fa3_attention", "vllm_gemm", "vllm_experts", "vllm_steps")
# ``vllm_steps`` needs a logged step for every sequence, which only re-read replay has.
_STEP_FLAG = "vllm_steps"
# ``ep_sum`` needs each row's vLLM data-parallel rank, which the probe knows for replay modes only.
_NEEDS_PLACEMENT = "ep_sum"


def _enabled(*flags: str) -> dict[str, bool]:
    return dict.fromkeys(flags, True)


NUMERICS_CANDIDATES = {
    **{flag: {flag: True} for flag in NUMERICS_FLAGS if flag != _STEP_FLAG},
    COMPILED_STACK: _enabled(*_COMPILED_STACK_FLAGS),
    COMPILED_STACK_ALL: _enabled(*_COMPILED_STACK_FLAGS, "route_weight"),
    KEPT_STACK: _enabled(*_COMPILED_STACK_FLAGS, "fa3_attention"),
    f"{COMPILED_STACK}+ep_sum": _enabled(*_COMPILED_STACK_FLAGS, "ep_sum"),
    f"{KEPT_STACK}+ep_sum": _enabled(*_COMPILED_STACK_FLAGS, "fa3_attention", "ep_sum"),
    f"{COMPILED_STACK_ALL}+ep_sum": _enabled(*_COMPILED_STACK_FLAGS, "route_weight", "ep_sum"),
    VLLM_KERNEL_STACK: _enabled(*_COMPILED_STACK_FLAGS, "route_weight", *_KERNEL_FLAGS),
    KEPT_VLLM_GEMM: _enabled(*_COMPILED_STACK_FLAGS, "fa3_attention", "vllm_gemm"),
    KEPT_VLLM_EXPERTS: _enabled(*_COMPILED_STACK_FLAGS, "fa3_attention", "vllm_experts"),
    KEPT_VLLM_KERNELS: _enabled(*_COMPILED_STACK_FLAGS, "fa3_attention", "vllm_gemm", "vllm_experts"),
    KEPT_VLLM_KERNELS_ROUTER_ROWS: _enabled(
        *_COMPILED_STACK_FLAGS, "fa3_attention", "vllm_gemm", "vllm_experts", "router_rows"
    ),
    # Every vLLM kernel and vLLM's expert-parallel addition order: the stack the harness finds byte-equal to
    # compiled vLLM apart from the XSA reductions, a few RoPE and norm elements and the router GEMM's row count.
    f"{KEPT_VLLM_KERNELS}+ep_sum": _enabled(
        *_COMPILED_STACK_FLAGS, "fa3_attention", "vllm_gemm", "vllm_experts", "ep_sum"
    ),
    **{
        f"{KEPT_STACK_2}+{flag}": _enabled(*_COMPILED_STACK_FLAGS, *_KEPT_2_FLAGS, flag)
        for flag in (*_VLLM_REGION_FLAGS, "ep_sum")
    },
    VLLM_FORWARD: _enabled(*_COMPILED_STACK_FLAGS, *_KEPT_2_FLAGS, *_VLLM_REGION_FLAGS),
    VLLM_FORWARD_EP_SUM: _enabled(*_COMPILED_STACK_FLAGS, *_KEPT_2_FLAGS, *_VLLM_REGION_FLAGS, "ep_sum"),
    INVARIANT_STACK: _enabled(*_COMPILED_STACK_FLAGS, *_INVARIANT_FLAGS),
    VLLM_INVARIANT_FORWARD: _enabled(*_COMPILED_STACK_FLAGS, *_INVARIANT_FLAGS, *_VLLM_REGION_FLAGS, "ep_sum"),
}


# Candidates scored as the logged re-read's vLLM steps computed each sequence: re-read replay modes only.
STEP_CANDIDATES = {
    KEPT_STACK_3: _enabled(*_COMPILED_STACK_FLAGS, *_KEPT_3_FLAGS),
    f"{KEPT_STACK_3}+ep_sum": _enabled(*_COMPILED_STACK_FLAGS, *_KEPT_3_FLAGS, "ep_sum"),
    VLLM_STEP_FORWARD: _enabled(*_COMPILED_STACK_FLAGS, *_KEPT_3_FLAGS, *_VLLM_REGION_FLAGS),
    VLLM_STEP_FORWARD_EP_SUM: _enabled(*_COMPILED_STACK_FLAGS, *_KEPT_3_FLAGS, *_VLLM_REGION_FLAGS, "ep_sum"),
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
    KEPT_STACK,
    f"{COMPILED_STACK}+ep_sum",
    f"{KEPT_STACK}+ep_sum",
    VLLM_KERNEL_STACK,
    KEPT_VLLM_GEMM,
    KEPT_VLLM_EXPERTS,
    KEPT_VLLM_KERNELS,
    KEPT_VLLM_KERNELS_ROUTER_ROWS,
    f"{KEPT_VLLM_KERNELS}+ep_sum",
    *(f"{KEPT_STACK_2}+{flag}" for flag in (*_VLLM_REGION_FLAGS, "ep_sum")),
    VLLM_FORWARD,
    VLLM_FORWARD_EP_SUM,
    INVARIANT_STACK,
    VLLM_INVARIANT_FORWARD,
):
    TRAINER_MODES[f"{REPLAY_MODE}+{_candidate}"] = ModeSpec(
        _with_numerics(_replay, NUMERICS_CANDIDATES[_candidate]), requires_routes=True
    )

# The batch-layout control of a stack: the trainer against itself in reversed order with larger micro-batches,
# the re-read's routes replayed, so the prefill metric's layout floor is measured under that stack.
for _candidate in (
    KEPT_STACK,
    KEPT_VLLM_GEMM,
    KEPT_VLLM_KERNELS,
    KEPT_VLLM_KERNELS_ROUTER_ROWS,
    VLLM_FORWARD,
    VLLM_FORWARD_EP_SUM,
    VLLM_INVARIANT_FORWARD,
):
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

for _candidate, _flags in STEP_CANDIDATES.items():
    for _mode, _repeat in ((REREAD_REPLAY_MODE, False), (REPEAT_REREAD_REPLAY_MODE, True)):
        TRAINER_MODES[f"{_mode}+{_candidate}"] = ModeSpec(
            _with_numerics(_replay, _flags),
            requires_routes=True,
            repeat_layout=_repeat,
            route_source="reread",
            needs_step_plan=True,
        )
