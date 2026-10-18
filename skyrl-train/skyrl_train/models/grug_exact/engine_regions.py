"""The decode-invariant engine's fused regions of Grug, compiled by the trainer from the torch code vLLM compiles.

vLLM compiles Grug with Inductor, which fuses the elementwise and reduction code between two GEMMs, attention calls or
custom ops into kernels whose arithmetic follows from that code, the Inductor settings and the row count the compile
takes as its size hint. Each region below copies that code from the pinned vLLM wheel and compiles it with the engine's
settings (``INDUCTOR_COMPILE_CONFIG``), so on the trainer's bf16 rows it computes the engine's values bit for bit, under
three conditions this module enforces: the call records no autograd, the settings select kernels without timing them
(``deterministic``), and the first compile takes the engines' ``max_num_batched_tokens`` as its row hint, which
``set_compile_rows`` gives it. A region compiles once per process; the callers attach the gradients (``layers``).
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch._dynamo.source import LocalSource, Source, TensorProperty, TensorPropertySource
from torch._inductor.compile_fx import compile_fx, compile_fx_inner
from torch._inductor.output_code import OutputCode

from skyrl_train.inference_engines.vllm.decode_invariant.engine import INDUCTOR_COMPILE_CONFIG
from skyrl_train.models.grug_moe import GRUG_ATTN_GATE_SCALE, GRUG_QK_RMS_NORM_EPS, GRUG_XSA_EPS

# The row hint of every region this process compiles; torch.compile keeps compiled code per process.
_compile_rows: int | None = None


def set_compile_rows(rows: int) -> None:
    """Compile this process's regions at ``rows`` rows, the decode-invariant engines' ``max_num_batched_tokens``, at
    which vLLM compiles theirs."""
    global _compile_rows
    if not INDUCTOR_COMPILE_CONFIG.get("deterministic"):
        raise ValueError(f"the engines' Inductor settings {INDUCTOR_COMPILE_CONFIG} choose kernels by timing them")
    if _compile_rows not in (None, rows):
        raise ValueError(f"this process compiles its regions at {_compile_rows} rows, not {rows}")
    _compile_rows = rows


class _Region:
    """A region compiled at its first call and run without autograd. Its first ``row_inputs`` inputs hold the rows of
    the first input's leading dimensions, each row a vector or one value, and go in as contiguous ``[rows, width]`` or
    ``[rows]``, as the engine's regions read theirs; the outputs come back with those leading dimensions.

    The first call compiles the region with ``torch.compile``, its rows marked dynamic with the engines' row count as
    their size hint, and keeps the guards Dynamo built and the graph Inductor compiled. Every later call evaluates
    those guards and runs the compiled graph on the call's tensors and row count, without Dynamo's and AOTAutograd's
    per-call wrappers; a call whose inputs fail a guard, which Dynamo would compile again, raises."""

    def __init__(self, function: Callable, row_inputs: int):
        self._function = function
        self._row_inputs = row_inputs
        code = function.__code__
        self._parameters = code.co_varnames[: code.co_argcount]
        self._guards = None
        self._graph: OutputCode | None = None
        # Inductor's generated function of the compiled graph: graph inputs in a list, outputs in a tuple.
        self._run: Callable | None = None
        # Each graph input: the index of the argument it is, or None for the row count.
        self._graph_inputs: tuple[int | None, ...] = ()
        self._returns_tuple = False

    def __call__(self, *inputs):
        leading = inputs[0].shape[:-1]
        rows = leading.numel()
        start = len(leading)
        count = self._row_inputs
        arguments = [value.detach().reshape(rows, *value.shape[start:]).contiguous() for value in inputs[:count]]
        arguments.extend([value.detach() if isinstance(value, torch.Tensor) else value for value in inputs[count:]])
        if self._guards is None:
            with torch.no_grad():
                outputs = self._compile(arguments)
        else:
            grad = torch.is_grad_enabled()
            torch._C._set_grad_enabled(False)
            try:
                named = dict(zip(self._parameters, arguments))
                if not self._guards.check(named):
                    reason = self._guards.check_verbose(named).verbose_code_parts
                    raise RuntimeError(f"{self._function.__name__} would compile again: {reason}")
                outputs = self._run([rows if index is None else arguments[index] for index in self._graph_inputs])
            finally:
                torch._C._set_grad_enabled(grad)
        if self._returns_tuple:
            return tuple(value.view(*leading, *value.shape[1:]) for value in outputs)
        return outputs[0].view(*leading, *outputs[0].shape[1:])

    def _compile(self, arguments: list) -> tuple[torch.Tensor, ...]:
        """The first call's outputs; keeps the compiled graph, where its inputs come from and the guards."""
        if _compile_rows is None:
            raise RuntimeError("set_compile_rows sets the engines' row count before a region compiles")
        for tensor in arguments[: self._row_inputs]:
            torch._dynamo.mark_dynamic(tensor, 0, hint_override=_compile_rows)
        output = torch.compile(self._function, fullgraph=True, backend=self._inductor)(*arguments)
        (entry,) = torch._dynamo.eval_frame._debug_get_cache_entry_list(self._function.__code__)
        self._returns_tuple = isinstance(output, tuple)
        outputs = output if self._returns_tuple else (output,)
        rows = arguments[0].shape[0]
        self._run = self._graph.current_callable
        graph_outputs = self._run([rows if index is None else arguments[index] for index in self._graph_inputs])
        if len(graph_outputs) != len(outputs) or not all(
            torch.equal(value.view(torch.uint8), graph_value.view(torch.uint8))
            for value, graph_value in zip(outputs, graph_outputs, strict=False)
        ):
            raise RuntimeError(f"{self._function.__name__}'s compiled graph does not return the region's values")
        self._guards = entry.guard_manager
        return outputs

    def _inductor(self, graph: torch.fx.GraphModule, example_inputs: list) -> Callable:
        """Dynamo's backend: Inductor's ``compile_fx`` under the engines' settings, as ``torch.compile`` with
        ``options=INDUCTOR_COMPILE_CONFIG`` calls it. Keeps where each graph input comes from and the compiled graph
        ``compile_fx_inner`` returns; the AOTAutograd cache is off, so that a cached compile also reaches it."""
        sources = [node.meta["grapharg"].source for node in graph.graph.find_nodes(op="placeholder")]
        self._graph_inputs = tuple(self._graph_input(source) for source in sources)
        compiled: list[OutputCode] = []

        def inner_compile(*args, **kwargs) -> OutputCode:
            compiled.append(compile_fx_inner(*args, **kwargs))
            return compiled[-1]

        with torch._functorch.config.patch(enable_autograd_cache=False):
            function = compile_fx(
                graph, example_inputs, inner_compile=inner_compile, config_patches=dict(INDUCTOR_COMPILE_CONFIG)
            )
        (self._graph,) = compiled
        return function

    def _graph_input(self, source: Source) -> int | None:
        """The index of the argument that graph input ``source`` is, or None for a row input's row count."""
        if isinstance(source, LocalSource):
            return self._parameters.index(source.local_name)
        if (
            isinstance(source, TensorPropertySource)
            and source.prop is TensorProperty.SIZE
            and source.idx == 0
            and isinstance(source.base, LocalSource)
            and self._parameters.index(source.base.local_name) < self._row_inputs
        ):
            return None
        raise ValueError(f"{self._function.__name__}'s graph reads {source.name}, not an argument or its row count")


def _rms_norm(x, weight, epsilon):
    """``vllm.ir.ops.rms_norm``'s native implementation (``vllm/ir/ops/layernorm.py:9-21``), vLLM's ``RMSNorm``."""
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + epsilon)
    x = x.to(weight.dtype) * weight
    return x.to(orig_dtype)


def _rms_norm_from_square_sum(x, square_sum, weight, epsilon):
    """``_rms_norm`` of the stored bf16 rows ``x`` with the variance from the sum of squares the kernel that formed
    them reduced: the rest of the kernel that ``_residual_square_sum`` and ``_gated_product_square_sum`` begin."""
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    x = x * torch.rsqrt(square_sum[:, None] / x.shape[-1] + epsilon)
    x = x.to(weight.dtype) * weight
    return x.to(orig_dtype)


def _residual_square_sum(residual, mlp):
    """A layer's output ``residual + mlp`` (``grugmoe.py:1138``) and the fp32 sum of squares of each unrounded row,
    which the next layer's input ``RMSNorm`` (``:1124``) reduces in the same kernel."""
    summed = residual + mlp
    return summed, summed.to(torch.float32).pow(2).sum(dim=-1)


def _final_norm(residual, mlp, weight, epsilon):
    """The last layer's ``residual + mlp`` and the final ``RMSNorm`` (``grugmoe.py:1138, 1250``), one kernel."""
    return _rms_norm(residual + mlp, weight, epsilon)


def _gate_silu(gate_hidden):
    """``GrugMoeGatedNorm.forward``'s activation (``grugmoe.py:483``)."""
    return F.silu(gate_hidden)


def _gated_product(x, gate):
    """``GrugMoeGatedNorm.forward``'s product (``grugmoe.py:485``)."""
    return x * torch.sigmoid(gate).to(x.dtype)


def _gated_product_square_sum(x, gate):
    """The embedding gated norm's product (``grugmoe.py:1232``) and the fp32 sum of squares of each unrounded row,
    which layer 0's input ``RMSNorm`` (``:1124``) reduces in the same kernel."""
    product = _gated_product(x, gate)
    return product, product.to(torch.float32).pow(2).sum(dim=-1)


def _shared_activation(gate, up):
    """``GrugMoeDenseMLP.forward``'s activation (``grugmoe.py:529``)."""
    return F.silu(gate) * up


def _align_kv_heads(value, num_q_heads):
    """``grugmoe._align_kv_heads`` (``grugmoe.py:98-108``) for grouped KV heads."""
    num_kv_heads = value.shape[1]
    repeat = num_q_heads // num_kv_heads
    value = value.unsqueeze(2).expand(value.shape[0], num_kv_heads, repeat, value.shape[2])
    return value.reshape(value.shape[0], num_q_heads, value.shape[-1]).contiguous()


def _xsa_head_gate(attn_output, value_states, gate_logits, head_dim):
    """``GrugMoeAttention.apply_xsa`` (``grugmoe.py:971-990``) after the head gate's GEMM, whose fp32 output the
    engine's ``InvariantLinear`` rounds to bf16 (``decode_invariant/model.py:27``)."""
    num_tokens, num_heads = gate_logits.shape
    value_heads = value_states.view(num_tokens, -1, head_dim)
    aligned_value = _align_kv_heads(value_heads, num_heads)
    attn_heads = attn_output.view(num_tokens, num_heads, head_dim)
    dot = torch.sum(attn_heads * aligned_value, dim=-1, keepdim=True)
    value_norm_sq = torch.sum(aligned_value * aligned_value, dim=-1, keepdim=True)
    attn_heads = attn_heads - (dot / (value_norm_sq + GRUG_XSA_EPS)) * aligned_value
    gate = GRUG_ATTN_GATE_SCALE * torch.sigmoid(gate_logits.to(attn_output.dtype))
    attn_heads = gate[..., None].to(attn_heads.dtype) * attn_heads
    return attn_heads.reshape(num_tokens, -1)


def _head_rms_norm(x):
    """``grugmoe._rms_norm`` (``grugmoe.py:91-95``), the weightless q/k norm."""
    dtype = x.dtype
    x_float = x.float()
    variance = torch.mean(torch.square(x_float), dim=-1, keepdim=True)
    return (x_float * torch.rsqrt(variance + GRUG_QK_RMS_NORM_EPS)).to(dtype)


def _normalized_query_key(q, k, head_dim):
    """``GrugMoeAttention.forward``'s q/k norms (``grugmoe.py:1003-1012``)."""
    num_tokens = q.shape[0]
    q = _head_rms_norm(q.view(num_tokens, -1, head_dim))
    k = _head_rms_norm(k.view(num_tokens, -1, head_dim))
    return q.reshape(num_tokens, -1), k.reshape(num_tokens, -1)


def _query_key(q, k, head_dim, qk_mult, qk_mult_scale):
    """``GrugMoeAttention.forward`` from the q and k projections to attention (``grugmoe.py:1003-1016``) on a layer
    without RoPE."""
    q, k = _normalized_query_key(q, k, head_dim)
    return q * qk_mult * qk_mult_scale, k


def _query_key_rope(q, k, positions, cos_sin_cache, head_dim, qk_mult, qk_mult_scale):
    """The same on a RoPE layer: ``RotaryEmbedding.forward_native`` (``rotary_embedding/base.py:203-219``) rotates
    half of each head, NeoX style, the rotation the gradient kernels implement (``reference_kernels``)."""
    from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding  # noqa: PLC0415 - GPU-only extra

    q, k = _normalized_query_key(q, k, head_dim)
    q, k = RotaryEmbedding.forward_static(positions, q, k, head_dim, head_dim // 2, cos_sin_cache, True)
    return q * qk_mult * qk_mult_scale, k


rms_norm = _Region(_rms_norm, row_inputs=1)
rms_norm_from_square_sum = _Region(_rms_norm_from_square_sum, row_inputs=2)
residual_square_sum = _Region(_residual_square_sum, row_inputs=2)
final_norm = _Region(_final_norm, row_inputs=2)
gate_silu = _Region(_gate_silu, row_inputs=1)
gated_product = _Region(_gated_product, row_inputs=2)
gated_product_square_sum = _Region(_gated_product_square_sum, row_inputs=2)
shared_activation = _Region(_shared_activation, row_inputs=2)
xsa_head_gate = _Region(_xsa_head_gate, row_inputs=3)
query_key = _Region(_query_key, row_inputs=2)
_QUERY_KEY_ROPE = _Region(_query_key_rope, row_inputs=3)


@functools.cache
def _rotary_table(device: int, base: float, positions: int, rotary_dim: int) -> torch.Tensor:
    """The engine's bf16 cos/sin table, built on the GPU as vLLM's ``RotaryEmbedding`` builds it at load
    (``rotary_embedding/base.py:57-61, 80-103``)."""
    with torch.device(f"cuda:{device}"):
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        freqs = torch.einsum("i,j -> ij", torch.arange(positions, dtype=torch.float), inv_freq)
        return torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(torch.bfloat16)


@functools.lru_cache(maxsize=16)
def _token_positions(sequence: int, batch: int, device: torch.device) -> torch.Tensor:
    """The position in its sequence of each token of ``[sequence, batch]`` tokens, in that order."""
    return torch.arange(sequence, device=device).repeat_interleave(batch)


def query_key_rope(q, k, head_dim: int, qk_mult: float, qk_mult_scale: float, base: float, positions: int):
    """``query_key`` on a RoPE layer of ``base`` and a ``positions``-row table for ``[S, B, width]`` q and k, each
    token at its position in its sequence."""
    sequence, batch = q.shape[:2]
    table = _rotary_table(q.device.index, base, positions, head_dim // 2)
    return _QUERY_KEY_ROPE(q, k, _token_positions(sequence, batch, q.device), table, head_dim, qk_mult, qk_mult_scale)
