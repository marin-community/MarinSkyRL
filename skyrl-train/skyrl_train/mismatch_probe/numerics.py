"""Switchable Grug rounding points that follow compiled vLLM.

Compiled vLLM runs Grug through Inductor, which keeps each fused elementwise chain in fp32 and rounds
to bf16 once when it stores the result. The Megatron trainer runs op by op and rounds after most
operations. Each flag makes one chain compute in fp32 and round once, as compiled vLLM does:

- ``gated_norm``: ``norm(x) * sigmoid(gate)``;
- ``qk_rope``: q/k RMS norm, RoPE with the bf16 cos/sin table, and the query scale;
- ``xsa_gate``: XSA followed by the ``2 * sigmoid`` head gate;
- ``shared_swiglu``: ``silu(gate) * up`` in the shared expert;
- ``router_input``: the router reads the unrounded gated-norm product in fp32, with an fp32
  router GEMM (requires ``gated_norm``);
- ``route_weight``: each routed expert's down projection accumulates in fp32 and is multiplied by
  its route weight before one rounding, instead of weighting the activation before the projection.

Probe modes set flags for one scoring forward; the flags default to the current trainer numerics.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, fields, replace


@dataclass(frozen=True)
class GrugNumerics:
    gated_norm: bool = False
    qk_rope: bool = False
    xsa_gate: bool = False
    shared_swiglu: bool = False
    router_input: bool = False
    route_weight: bool = False


NUMERICS_FLAGS = tuple(field.name for field in fields(GrugNumerics))
_active = GrugNumerics()


def active_numerics() -> GrugNumerics:
    return _active


@contextmanager
def grug_numerics(**flags: bool) -> Iterator[GrugNumerics]:
    """Enable the named rounding points for the duration of the block."""
    global _active
    unknown = set(flags) - set(NUMERICS_FLAGS)
    if unknown:
        raise ValueError(f"unknown Grug numerics flags: {sorted(unknown)}")
    updated = replace(_active, **flags)
    if updated.router_input and not updated.gated_norm:
        raise ValueError("router_input reads the fp32 gated-norm product and requires gated_norm")
    previous, _active = _active, updated
    try:
        yield updated
    finally:
        _active = previous
