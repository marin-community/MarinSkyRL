from types import SimpleNamespace

import torch

from skyrl_train.mismatch_harness.output_code import LaunchKind, parse_output_code
from skyrl_train.mismatch_harness.replay import run_call

# An output-code module in Inductor's format: a GEMM, a fused kernel that updates the GEMM's
# buffer in place after the storage is reused under a new name, and an opaque custom op.
MODULE = """
# AOT ID: ['0_inference']
triton_poi_fused_add_0 = async_compile.triton('triton_poi_fused_add_0', '''
@triton_heuristics.pointwise(
    size_hints={'x': 16},
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}},
)
@triton.jit
def triton_poi_fused_add_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


class Runner:
    def call(self, args):
        arg0_1, arg1_1, arg2_1 = args
        args.clear()
        with torch.cuda._DeviceGuard(0):
            buf0 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(arg0_1, arg1_1, out=buf0)
            buf1 = buf0; del buf0  # reuse
            # Topologically Sorted Source Nodes: [add], Original ATen: [aten.add]
            triton_poi_fused_add_0.run(buf1, arg2_1, 16, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [moe_forward], Original ATen: [vllm.moe_forward]
            buf2 = torch.ops.vllm.moe_forward.default(buf1, arg2_1, None, None, None, 0)
        return (buf2, buf1, )
"""


class _AddKernel:
    """Stands in for the compiled Triton kernel: adds ``in_ptr0`` into ``in_out_ptr0``."""

    def run(self, in_out, addend, xnumel, stream):
        in_out.add_(addend)


def _namespace():
    return {
        "torch": torch,
        "empty_strided_cuda": lambda size, stride, dtype: torch.empty_strided(size, stride, dtype=dtype),
        "extern_kernels": SimpleNamespace(mm=lambda left, right, out: torch.mm(left, right, out=out)),
        "triton_poi_fused_add_0": _AddKernel(),
        "raw_stream0": 0,
    }


def test_replay_keeps_each_launch_output_and_feeds_substitutions_to_the_next_launch():
    code = parse_output_code(MODULE)
    left, right, addend = torch.randn(4, 4), torch.randn(4, 4), torch.randn(4, 4)
    received = []

    def moe(hidden, logits, *rest):
        received.append(hidden.clone())
        return hidden * 2

    result = run_call(code, _namespace(), [left, right, addend], ops={"vllm.moe_forward.default": moe})

    assert [(launch.kind, launch.target) for launch in code.launches] == [
        (LaunchKind.EXTERN, "mm"),
        (LaunchKind.TRITON, "triton_poi_fused_add_0"),
        (LaunchKind.CUSTOM_OP, "vllm.moe_forward.default"),
    ]
    assert code.root("buf1") == "buf0"
    gemm, fused, op = result.values
    # The GEMM's value is kept as it was stored, before the fused kernel rewrote the same storage.
    torch.testing.assert_close(gemm.tensor, left @ right, rtol=0, atol=0)
    torch.testing.assert_close(fused.tensor, left @ right + addend, rtol=0, atol=0)
    torch.testing.assert_close(op.tensor, (left @ right + addend) * 2, rtol=0, atol=0)
    torch.testing.assert_close(result.outputs[1], left @ right + addend, rtol=0, atol=0)

    # A buffer written before the fused kernel's statement is what that kernel reads.
    substitute = torch.full((4, 4), 3.0)
    fused_statement = code.launches[1].statement

    def before(index, environment):
        if index == fused_statement:
            environment["buf1"].copy_(substitute)

    substituted = run_call(
        code, _namespace(), [left, right, addend], ops={"vllm.moe_forward.default": moe}, before=before
    )
    torch.testing.assert_close(substituted.values[0].tensor, left @ right, rtol=0, atol=0)
    torch.testing.assert_close(substituted.values[1].tensor, substitute + addend, rtol=0, atol=0)
    torch.testing.assert_close(received[-1], substitute + addend, rtol=0, atol=0)
