"""The regions of the decode-invariant engine's compiled vLLM model, run with the captured Inductor kernels
(``inductor_kernels``) on the trainer's tensors.

Every function takes bf16 tensors whose last dimension is the width the kernels hard-code, runs the kernel outside
autograd, and returns the kernel's output; the callers attach the gradients (``layers``).
"""

from __future__ import annotations

import functools

import torch
from torch._inductor.async_compile import AsyncCompile

from skyrl_train.models.grug_exact import inductor_kernels as kernels
from skyrl_train.models.grug_exact.capture_inductor_kernels import kernel_parameters
from skyrl_train.models.grug_exact.inductor_kernels import COMPILED_MODEL

HIDDEN = COMPILED_MODEL["hidden_size"]
HEADS = COMPILED_MODEL["num_attention_heads"]
KV_HEADS = COMPILED_MODEL["num_key_value_heads"]
HEAD_DIM = COMPILED_MODEL["head_dim"]
ROTARY_DIM = HEAD_DIM // 2
ROTARY_POSITIONS = COMPILED_MODEL["max_position_embeddings"]
KV_WIDTH = KV_HEADS * HEAD_DIM
SQUARE_SUM_PARAMETER = "out_ptr_square_sum"


def _replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"expected one occurrence of {old!r} in a captured kernel source")
    return text.replace(old, new)


def _with_square_sum_output(source: tuple[str, str], square_sum: str) -> tuple[str, str]:
    """The kernel with one more output after ``r0_numel``: its per-row fp32 sum of squares ``square_sum``, stored
    after the reduction."""
    name, text = source
    derived = f"{name}_square_sum"
    index = kernel_parameters(text).index("r0_numel") + 1
    text = _replace_once(text, f"def {name}(", f"def {derived}(")
    text = _replace_once(text, f"'kernel_name': '{name}'", f"'kernel_name': '{derived}'")
    text = _replace_once(
        text, "'r0_numel': 'i32', 'XBLOCK'", f"'r0_numel': 'i32', '{SQUARE_SUM_PARAMETER}': '*fp32', 'XBLOCK'"
    )
    text = _replace_once(
        text,
        "xnumel, r0_numel, XBLOCK : tl.constexpr",
        f"xnumel, r0_numel, {SQUARE_SUM_PARAMETER}, XBLOCK : tl.constexpr",
    )
    text = _replace_once(
        text, "[['tt.divisibility', 16]]}]}", f"[['tt.divisibility', 16]], ({index},): [['tt.divisibility', 16]]}}]}}"
    )
    reduction = f"    {square_sum} = tl.sum(_{square_sum}, 1)[:, None]\n"
    text = _replace_once(
        text, reduction, f"{reduction}    tl.store({SQUARE_SUM_PARAMETER} + (x0), {square_sum}, xmask)\n"
    )
    return derived, text


@functools.cache
def _norm_from_square_sum() -> tuple[str, str]:
    """The second loop of ``RESIDUAL_NORM`` (the same arithmetic): the stored bf16 row normalized with a sum of squares
    read from memory instead of the one the first loop reduced."""
    name = "triton_red_rms_norm_from_square_sum"
    original, text = kernels.RESIDUAL_NORM
    header, decorated = text.split("@triton_heuristics.reduction(")
    decorator, body = decorated.split("@triton.jit\n")
    for old, new in (
        (
            "{'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr1': '*bf16',",
            "{'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'out_ptr1': '*bf16',",
        ),
        (f"'kernel_name': '{original}'", f"'kernel_name': '{name}'"),
        ("'mutated_arg_names': ['in_out_ptr0']", "'mutated_arg_names': []"),
    ):
        decorator = _replace_once(decorator, old, new)
    preamble, loops = body.split("    _tmp6 = tl.full(", 1)
    preamble = _replace_once(
        preamble,
        f"def {original}(in_out_ptr0, in_ptr0, in_ptr1, out_ptr1,",
        f"def {name}(in_ptr0, in_ptr1, in_ptr2, out_ptr1,",
    )
    second_loop = "    for r0_offset" + loops.split("    for r0_offset")[2]
    second_loop = _replace_once(second_loop, "        tl.debug_barrier()\n", "")
    second_loop = _replace_once(
        second_loop, "tl.load(in_out_ptr0 + (r0_1 + 2560*x0)", "tl.load(in_ptr0 + (r0_1 + 2560*x0)"
    )
    second_loop = _replace_once(second_loop, "tl.load(in_ptr1 + (r0_1)", "tl.load(in_ptr2 + (r0_1)")
    statistic = "    tmp6 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')\n"
    return name, f"{header}@triton_heuristics.reduction({decorator}@triton.jit\n{preamble}{statistic}{second_loop}"


@functools.cache
def _residual_norm_square_sum() -> tuple[str, str]:
    """``RESIDUAL_NORM`` with its sum of squares stored."""
    return _with_square_sum_output(kernels.RESIDUAL_NORM, "tmp6")


@functools.cache
def _embedding_product_norm_square_sum() -> tuple[str, str]:
    """``EMBEDDING_PRODUCT_NORM`` with its sum of squares stored."""
    return _with_square_sum_output(kernels.EMBEDDING_PRODUCT_NORM, "tmp7")


@functools.cache
def _kernel(name: str, text: str, device: int):
    """``name`` compiled from ``text`` by Inductor's runtime for ``device``, which picks the launch configuration the
    engine's compile picked (the source carries ``deterministic``)."""
    text = _replace_once(
        text, "DeviceProperties(type='cuda', index=0,", f"DeviceProperties(type='cuda', index={device},"
    )
    text = _replace_once(text, "'autotune_local_cache': True", "'autotune_local_cache': False")
    compiler = AsyncCompile()
    scope = {"kernel": compiler.triton(name, text, device_str="cuda")}
    compiler.wait(scope)
    return scope["kernel"]


def _run(source: tuple[str, str], *arguments) -> None:
    device = torch.cuda.current_device()
    _kernel(*source, device).run(*arguments, stream=torch.cuda.current_stream(device).cuda_stream)


@functools.cache
def _unit_weight(device: torch.device) -> torch.Tensor:
    return torch.ones(HIDDEN, dtype=torch.bfloat16, device=device)


def _rows(tensor: torch.Tensor, width: int) -> torch.Tensor:
    """``tensor``'s values as contiguous bf16 ``[rows, width]``, outside autograd (the kernels compute values only)."""
    if tensor.dtype != torch.bfloat16 or tensor.numel() % width:
        raise ValueError(f"the kernels take bf16 rows of width {width}, got {tensor.dtype}{tuple(tensor.shape)}")
    return tensor.detach().reshape(-1, width).contiguous()


def gate_silu(gate_hidden: torch.Tensor) -> torch.Tensor:
    """The engine's SiLU of a gated norm's down projection, one rounding."""
    output = gate_hidden.detach().contiguous().clone()
    _run(kernels.GATE_SILU, output, output.numel())
    return output


def gated_product(normalized: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """The engine's gated-norm output ``normalized * sigmoid(gate)``, one rounding."""
    output = _rows(normalized, HIDDEN).clone()
    _run(kernels.GATED_PRODUCT, output, _rows(gate, HIDDEN), output.numel())
    return output


def rms_norm(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """The engine's post-attention RMSNorm of the stored bf16 hidden states ``[rows, 2560]``."""
    hidden = _rows(hidden, HIDDEN)
    output = torch.empty_like(hidden)
    _run(kernels.RMS_NORM, hidden, weight, output, hidden.shape[0], HIDDEN)
    return output


def rms_norm_from_square_sum(hidden: torch.Tensor, square_sum: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """The engine's input norm of the stored bf16 sum ``[rows, 2560]`` given its kernel's sum of squares."""
    hidden = _rows(hidden, HIDDEN)
    if square_sum.dtype != torch.float32 or square_sum.shape != (hidden.shape[0],):
        raise ValueError(f"one fp32 sum of squares per row expected, got {square_sum.dtype}{tuple(square_sum.shape)}")
    output = torch.empty_like(hidden)
    _run(_norm_from_square_sum(), hidden, square_sum.contiguous(), weight, output, hidden.shape[0], HIDDEN)
    return output


def _fused_square_sum(
    source: tuple[str, str], first: torch.Tensor, second: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The bf16 result of a fused norm kernel's elementwise step on ``first`` and ``second`` and the fp32 per-row sum
    of squares of its unrounded result; the kernel normalizes with a unit weight into an output that is discarded."""
    output = _rows(first, HIDDEN).clone()
    rows = output.shape[0]
    square_sum = torch.empty(rows, dtype=torch.float32, device=output.device)
    scratch = torch.empty_like(output)
    _run(source, output, _rows(second, HIDDEN), _unit_weight(output.device), scratch, rows, HIDDEN, square_sum)
    return output, square_sum


def residual_square_sum(residual: torch.Tensor, mlp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The bf16 layer output ``residual + mlp`` and the fp32 per-row sum of squares of the unrounded sum, from the
    engine's fused residual-add and input-norm kernel."""
    return _fused_square_sum(_residual_norm_square_sum(), residual, mlp)


def gated_product_square_sum(normalized: torch.Tensor, gate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The embedding gated norm's bf16 product and the fp32 per-row sum of squares of the unrounded product, from
    the engine's fused product and layer-0 input-norm kernel."""
    return _fused_square_sum(_embedding_product_norm_square_sum(), normalized, gate)


def final_norm(residual: torch.Tensor, mlp: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """The engine's final RMSNorm of the unrounded last-layer sum ``residual + mlp``."""
    output = _rows(residual, HIDDEN).clone()
    _run(kernels.FINAL_NORM, output, _rows(mlp, HIDDEN), weight, output.shape[0], HIDDEN)
    return output


def embedding_norm(embeddings: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """The engine's embedding RMSNorm of the looked-up bf16 embedding rows ``[rows, 2560]``.

    The kernel gathers rows of a table by token id; given the looked-up rows as the table and each row's own index
    as its id, it reads the same values.
    """
    table = _rows(embeddings, HIDDEN)
    rows = table.shape[0]
    output = torch.empty_like(table)
    identity = torch.arange(rows, dtype=torch.int32, device=table.device)
    _run(kernels.EMBEDDING_NORM, identity, table, weight, output, rows, HIDDEN)
    return output


def shared_activation(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """The engine's shared-expert activation ``silu(gate) * up`` of the two projections, one rounding."""
    if gate.shape != up.shape:
        raise ValueError(f"gate and up of one shape expected, got {tuple(gate.shape)} and {tuple(up.shape)}")
    output = gate.detach().contiguous().clone()
    _run(kernels.SHARED_ACTIVATION, output, up.detach().contiguous(), output.numel())
    return output


def xsa_head_gate(attention: torch.Tensor, value: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """The engine's XSA and ``2 * sigmoid`` head gate: ``[rows, 2560]`` from attention, ``[rows, 640]`` value and the
    fp32 ``[rows, 20]`` gate logits."""
    attention, value = _rows(attention, HIDDEN), _rows(value, KV_WIDTH)
    rows = attention.shape[0]
    if gate.dtype != torch.float32 or gate.shape[-1] != HEADS:
        raise ValueError(f"the XSA kernel reads fp32 gate logits of width {HEADS}, got {gate.dtype}{tuple(gate.shape)}")
    output = torch.empty_like(attention)
    _run(
        kernels.XSA_HEAD_GATE,
        attention,
        value,
        gate.detach().reshape(-1, HEADS).contiguous(),
        output,
        HEADS * rows,
        HEAD_DIM,
    )
    return output


@functools.cache
def rotary_table(device: int, base: float) -> torch.Tensor:
    """The engine's bf16 rotary cos/sin table, built on the GPU as vLLM's ``RotaryEmbedding`` builds it at load."""
    with torch.device(f"cuda:{device}"):
        inv_freq = 1.0 / (base ** (torch.arange(0, ROTARY_DIM, 2, dtype=torch.float) / ROTARY_DIM))
        positions = torch.arange(ROTARY_POSITIONS, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", positions, inv_freq)
        return torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(torch.bfloat16)


def _head_square_sums(rows: torch.Tensor, heads: int) -> torch.Tensor:
    sums = torch.empty(rows.shape[0] * heads, dtype=torch.float32, device=rows.device)
    _run(kernels.HEAD_SQUARE_SUMS, rows, sums, sums.numel(), HEAD_DIM)
    return sums


@functools.lru_cache(maxsize=16)
def _row_positions(sequence: int, batch: int, device: torch.device) -> torch.Tensor:
    """The position of each row of ``[sequence, batch]`` tokens flattened in that order."""
    return torch.arange(sequence, dtype=torch.int64, device=device).repeat_interleave(batch)


def query_key_rope(
    query: torch.Tensor, key: torch.Tensor, sequence: int, rotary_base: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """The engine's q/k RMS norm, half RoPE and query scale on a sliding-window layer's raw q (``[rows, 2560]``) and
    k (``[rows, 640]``) of ``[sequence, batch]`` tokens, each row at its position in the sequence, giving the q and k
    the engine hands to attention."""
    query, key = _rows(query, HIDDEN), _rows(key, KV_WIDTH)
    rows = query.shape[0]
    if rows % sequence:
        raise ValueError(f"{rows} rows are not whole sequences of {sequence} tokens")
    if sequence > ROTARY_POSITIONS:
        raise ValueError(f"the rotary kernels read positions below {ROTARY_POSITIONS}, got {sequence} tokens")
    table = rotary_table(query.device.index, rotary_base)
    positions = _row_positions(sequence, rows // sequence, query.device)
    query_sums, key_sums = _head_square_sums(query, HEADS), _head_square_sums(key, KV_HEADS)
    key_out = torch.empty(rows, KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=query.device)
    _run(
        kernels.KEY_ROPE,
        key,
        key_sums,
        positions,
        table,
        key_out,
        key_out[:, :, ROTARY_DIM:],
        KV_HEADS * ROTARY_DIM * rows,
    )
    query_rotary = torch.empty(rows, HEADS, ROTARY_DIM, dtype=torch.bfloat16, device=query.device)
    _run(kernels.QUERY_ROPE, query, query_sums, positions, table, query_rotary, HEADS * ROTARY_DIM * rows)
    query_out = torch.empty_like(query)
    _run(kernels.QUERY_SCALE, query_rotary, query, query_sums, query_out, HIDDEN * rows)
    return query_out, key_out.view(rows, KV_WIDTH)


def query_key_full(query: torch.Tensor, key: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The engine's q/k RMS norm and query scale on a full-attention layer's raw q ``[rows, 2560]`` and k
    ``[rows, 640]``."""
    query = _rows(query, HIDDEN)
    key_out = _rows(key, KV_WIDTH).clone()
    rows = query.shape[0]
    _run(kernels.KEY_NORM, key_out, KV_HEADS * rows, HEAD_DIM)
    query_out = torch.empty_like(query)
    _run(kernels.QUERY_NORM_SCALE, query, query_out, rows, HEADS * rows, HEAD_DIM)
    return query_out, key_out
