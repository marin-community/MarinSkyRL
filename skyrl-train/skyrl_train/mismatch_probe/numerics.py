"""Switchable Grug rounding points that follow compiled vLLM.

Compiled vLLM runs Grug through Inductor, which keeps each fused elementwise chain in fp32 and rounds
to bf16 once when it stores the result. The Megatron trainer runs op by op and rounds after most
operations. Each flag up to ``final_norm_fp32`` makes one chain compute in fp32 and round once, as
compiled vLLM does; the others reproduce vLLM's kernels (attention, dense GEMMs, routed experts) and its
expert-parallel addition order:

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
- ``fa3_attention``: attention runs vLLM's own FA3 forward kernel (unsplit, with a scoring plan's split counts, or
  as a decode-invariant engine runs it under ``fa3_window_rows``); the backward stays the trainer's cuDNN attention;
- ``ep_sum``: each token's routed expert outputs are added as vLLM's expert-parallel combine adds them,
  per vLLM EP rank in fp32, then across ranks in bf16 in NCCL's ring order for the token's vLLM
  data-parallel rank (router replay supplies that rank);
- ``vllm_gemm``: the dense projections run as compiled vLLM issues them, one ``torch.mm`` each in vLLM's
  shapes (q, k and v separately, the head-gate projection padded to 24 outputs, the shared expert's gate
  and up separately), instead of Transformer Engine's fused GEMMs;
- ``vllm_experts``: the routed experts' values come from vLLM's fused-MoE Triton kernels with vLLM's
  configs (route weight inside the fp32 down-projection accumulator, one rounding); the backward stays
  the trainer's grouped-GEMM experts. It replaces ``route_weight``;
- ``router_rows``: the fp32 router GEMM (``router_gemm``'s) runs in calls of a full vLLM prefill step's
  8,192 rows, so its summation order does not follow the micro-batch size;
- ``vllm_xsa``: XSA and the head gate take their values from compiled vLLM's Inductor kernel (its reduction
  order and fused multiply-adds); the gradient is ``xsa_gate``'s. It replaces ``xsa_gate``;
- ``vllm_qk``: the q/k norm, RoPE and query scale take their values from compiled vLLM's Inductor kernels; the
  gradient is ``qk_rope``'s. It replaces ``qk_rope``;
- ``vllm_norms``: every RMS norm and gated-norm product takes its value from compiled vLLM's Inductor kernels,
  each input norm from the unrounded sum that formed its input (as ``input_norm_variance``) and the final norm
  from the unrounded last-layer sum (as ``final_norm_fp32``); the gradients are those flags' and ``gated_norm``'s;
- ``vllm_swiglu``: the shared expert's activation ``silu(gate) * up`` takes its value from compiled vLLM's Inductor
  kernel (its exponential and division); the gradient is ``shared_swiglu``'s. It replaces ``shared_swiglu``;
- ``vllm_log_softmax``: the log-probabilities come from vLLM's model runner V2 log-probability kernel on the bf16
  logits (``compute_token_logprobs``), which computes every prompt and sampled log-probability the probe's engines
  return; the gradient is the trainer's own log-softmax's;
- ``vllm_steps``: in a scoring forward of a logged re-read (re-read replay), each sequence is computed as the vLLM
  engine step that ran its prefill alone: FA3 with that step's split counts, the fp32 router GEMM at the step's row
  count, and the LM head at the row counts of model runner V2's prompt and sampled log-probabilities. It replaces
  ``router_rows`` and has no training forward: the step log exists only for the re-read;
- ``invariant_router``: the router logits come from the row-invariant Triton GEMM that a decode-invariant vLLM engine
  runs (``grug_invariant_kernels.invariant_router_logits``), so they do not depend on the rows computed with them; the
  gradient is ``router_gemm``'s fp32 GEMM. It replaces ``router_rows`` and ``vllm_steps``' router rows;
- ``fa3_window_rows``: ``fa3_attention`` computes every row as a decode-invariant engine computes it in its decode steps
  and its prefills alike: every request with the engine's fixed FA3 split count, and on sliding-window layers every row
  past the window as a one-row request.

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
    vllm_gemm: bool = False
    vllm_experts: bool = False
    router_rows: bool = False
    vllm_xsa: bool = False
    vllm_qk: bool = False
    vllm_norms: bool = False
    vllm_swiglu: bool = False
    vllm_log_softmax: bool = False
    vllm_steps: bool = False
    invariant_router: bool = False
    fa3_window_rows: bool = False


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
