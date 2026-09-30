"""Switchable Grug rounding points that follow compiled vLLM.

Compiled vLLM runs Grug through Inductor, which keeps each fused elementwise chain in fp32 and rounds
to bf16 once when it stores the result. The Megatron trainer runs op by op and rounds after most
operations. Each flag up to ``final_norm_fp32`` makes one chain compute in fp32 and round once, as
compiled vLLM does; the last two reproduce vLLM's attention kernel and its expert-parallel addition order:

- ``gated_norm``: ``norm(x) * sigmoid(gate)``;
- ``qk_rope``: q/k RMS norm, RoPE with the bf16 cos/sin table, and the query scale: k rounds once, q's
  rotary half rounds after RoPE and again after the two scale factors, q's pass-through half once;
- ``xsa_gate``: XSA followed by the ``2 * sigmoid`` head gate;
- ``shared_swiglu``: ``silu(gate) * up`` in the shared expert;
- ``router_gemm``: router logits from an fp32 GEMM on the bf16 router input and the bf16-valued weight;
- ``route_weight``: each routed expert's down projection accumulates in fp32 and is multiplied by
  its route weight before one rounding, instead of weighting the activation before the projection;
- ``mlp_residual``: the next residual is ``h + (routed + shared)`` in fp32, rounded once;
- ``input_norm_variance``: each layer's input norm takes its variance from the unrounded residual sum
  (the unrounded embedding gated-norm product for layer 0) and normalizes the rounded residual;
- ``final_norm_fp32``: the final norm normalizes the unrounded residual sum;
- ``fa3_attention``: attention runs vLLM's own FA3 forward kernel (unsplit, or with a scoring plan's
  split counts); the backward stays the trainer's cuDNN attention;
- ``ep_sum``: each token's routed expert outputs are added as vLLM's expert-parallel combine adds them,
  per vLLM EP rank in fp32, then across ranks in bf16 in NCCL's ring order for the token's vLLM
  data-parallel rank (router replay supplies that rank).

Probe modes set flags for one scoring forward. The process default is the current trainer numerics, or
the set named by ``trainer.mismatch_probe.train_numerics``, which then applies to training too.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, fields, replace


@dataclass(frozen=True)
class GrugNumerics:
    gated_norm: bool = False
    qk_rope: bool = False
    xsa_gate: bool = False
    shared_swiglu: bool = False
    router_gemm: bool = False
    route_weight: bool = False
    mlp_residual: bool = False
    input_norm_variance: bool = False
    final_norm_fp32: bool = False
    fa3_attention: bool = False
    ep_sum: bool = False


NUMERICS_FLAGS = tuple(field.name for field in fields(GrugNumerics))
_active = GrugNumerics()


def active_numerics() -> GrugNumerics:
    return _active


def _check_flags(flags: Mapping[str, bool]) -> None:
    unknown = set(flags) - set(NUMERICS_FLAGS)
    if unknown:
        raise ValueError(f"unknown Grug numerics flags: {sorted(unknown)}")


def set_default_numerics(flags: Mapping[str, bool]) -> None:
    """Make ``flags`` this process's numerics for every forward and backward outside a probe mode."""
    global _active
    _check_flags(flags)
    _active = GrugNumerics(**flags)


@contextmanager
def grug_numerics(**flags: bool) -> Iterator[GrugNumerics]:
    """Enable the named rounding points for the duration of the block."""
    global _active
    _check_flags(flags)
    updated = replace(_active, **flags)
    previous, _active = _active, updated
    try:
        yield updated
    finally:
        _active = previous
