"""Native-Grug grouped_mm parity gates (1 GPU, EP=1).

The workstream turns `trainer.policy.fsdp_config.use_grouped_mm=true` on to replace the eager
256-expert Python loop, and then reads the step time as a result. Nothing gated that path before
this file: `test_grouped_gemm_parity.py` covers the *generic HF* `moe_grouped_gemm` swap and never
passes `use_grouped_mm=True`, so the native Grug path -- `GrugMoeSparseMoeBlock.enable_grouped_mm`
-> `_GrugGroupedExpertExecution` -> `_run_experts_grouped_mm` -- had no eval/train parity gate at
EP=1. A speedup measured on a numerically wrong path is worse than no measurement.

Gates:
  G4a-1  padded tail rows are ZEROED. pytorch#186365: ``torch._grouped_mm`` writes only rows
         covered by ``offs`` and leaves the ALIGN_SIZE_M-padded tail uninitialized. moe.py mitigates
         it; this asserts the mitigation, and is the only test here that is not tolerance-sensitive.
  G4a-2  enable_grouped_mm actually engages on every block (a silent no-op would return a null
         result that reads as a real one).
  G4a-3  eager vs grouped forward parity on identical weights and inputs.
  G4a-4  eager vs grouped backward parity.
  G4a-5  grouped forward is deterministic across repeats -- reading uninitialized memory would not
         be, and this catches it without depending on a tolerance.

Run::

    uv run --isolated --group dev pytest tests/gpu/gpu_ci/test_grug_grouped_mm_parity.py -x -q
"""

import pytest
import torch

from skyrl_train.models.grug_moe import (
    GrugMoeConfig,
    GrugMoeForCausalLM,
    enable_grug_grouped_mm,
)
from skyrl_train.models.layers.moe_routing import TokenReorderer, combine_routed_rows

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="grouped_mm requires CUDA")


def tiny_config(**overrides) -> GrugMoeConfig:
    values = {
        "vocab_size": 48,
        "hidden_size": 32,
        "intermediate_size": 64,
        "shared_expert_intermediate_size": 48,
        "num_local_experts": 8,
        "num_experts_per_tok": 2,
        "num_hidden_layers": 3,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 12,
        "max_position_embeddings": 32,
        "sliding_window": 4,
        "qk_mult": 1.37,
        "initializer_range": 0.02,
    }
    values.update(overrides)
    return GrugMoeConfig(**values)


def _model(dtype=torch.bfloat16) -> GrugMoeForCausalLM:
    torch.manual_seed(7)
    return GrugMoeForCausalLM(tiny_config()).to(device="cuda", dtype=dtype).eval()


def _ids() -> torch.Tensor:
    torch.manual_seed(11)
    return torch.randint(0, 48, (2, 16), device="cuda")


def test_g4a_1_grouped_mm_zeroes_the_padded_tail_rows():
    """The pytorch#186365 mitigation, asserted directly rather than assumed."""
    # The BARE impl, not the @expert_parallel-decorated wrapper. The decorator runs
    # generate_permute_indices first and expects num_tokens_per_expert in the EP-sharded
    # 128-wide cross-rank layout, so calling the wrapper with a plain per-expert count vector
    # dies in torchtitan with "shape '[0, -1]' is invalid". The impl is also where the
    # pytorch#186365 mitigation lives, which is what this gate is about.
    # NOTE the argument order: (w1, w2, w3, x, counts), not (x, w1, w2, w3, counts).
    from skyrl_train.models.layers.moe import _run_experts_grouped_mm_impl

    torch.manual_seed(3)
    n_experts, hidden, inter = 4, 32, 64
    # 40 routed rows, but a 64-row buffer: rows 40.. are the padded tail grouped_mm never writes.
    routed, padded = 40, 64
    x = torch.randn(padded, hidden, device="cuda", dtype=torch.bfloat16)
    w1 = torch.randn(n_experts, inter, hidden, device="cuda", dtype=torch.bfloat16)
    w3 = torch.randn(n_experts, inter, hidden, device="cuda", dtype=torch.bfloat16)
    w2 = torch.randn(n_experts, hidden, inter, device="cuda", dtype=torch.bfloat16)
    counts = torch.tensor([10, 10, 10, 10], device="cuda", dtype=torch.int32)
    assert int(counts.sum()) == routed

    out = _run_experts_grouped_mm_impl(w1, w2, w3, x, counts)
    tail = out[routed:]
    assert tail.shape[0] == padded - routed
    assert torch.equal(tail, torch.zeros_like(tail)), (
        f"padded tail rows are not zeroed -- {int((tail != 0).sum())} nonzero elements. "
        "pytorch#186365 leaks uninitialized memory into the expert output."
    )


def test_g4a_2_enable_grouped_mm_engages_on_every_block():
    """A silent no-op arm returns a null result that looks like a real one."""
    model = _model()
    blocks = [m for m in model.modules() if hasattr(m, "enable_grouped_mm")]
    assert blocks, "no Grug MoE blocks found -- the fixture is wrong, not the code"
    assert all(not b.experts.use_grouped_mm for b in blocks)

    n = enable_grug_grouped_mm(model)

    assert n == len(blocks) > 0
    assert all(b.experts.use_grouped_mm for b in blocks)


def _forward(model, ids):
    return model(input_ids=ids).logits.float()


def test_g4a_3_eager_and_grouped_forward_agree():
    ids = _ids()
    eager = _model()
    with torch.no_grad():
        want = _forward(eager, ids)

    grouped = _model()
    enable_grug_grouped_mm(grouped)
    with torch.no_grad():
        got = _forward(grouped, ids)

    torch.testing.assert_close(got, want, rtol=2e-2, atol=2e-2)


def test_g4a_4_eager_and_grouped_gradients_agree():
    ids = _ids()

    def grads(use_grouped: bool):
        m = _model().train()
        if use_grouped:
            enable_grug_grouped_mm(m)
        _forward(m, ids).square().mean().backward()
        return {k: p.grad.detach().float() for k, p in m.named_parameters() if p.grad is not None}

    want, got = grads(False), grads(True)
    assert want.keys() == got.keys()
    assert want, "no parameter received a gradient -- the backward did not run"
    for k in want:
        torch.testing.assert_close(got[k], want[k], rtol=5e-2, atol=5e-2, msg=lambda s, k=k: f"{k}: {s}")


def test_g4a_5_grouped_forward_is_deterministic():
    """Uninitialized memory would not repeat. No tolerance involved."""
    ids = _ids()
    m = _model()
    enable_grug_grouped_mm(m)
    with torch.no_grad():
        a = _forward(m, ids)
        b = _forward(m, ids)
    assert torch.equal(a, b), "grouped forward is not repeatable on identical inputs"


def test_g4a_6_combine_is_repeatable_under_contention():
    """The float32 combine must be bit-identical across launches while other kernels contend for SMs.

    Production shape (top-4 of 256 experts, hidden 2560) with heavy-tailed rows, so thousands of the
    per-token sums are order-sensitive. The former ``scatter_add`` combine reduced with atomics and
    let block scheduling pick the order; this is the test that did not exist when that shipped.
    """
    torch.manual_seed(5)
    num_tokens, top_k, num_experts, hidden = 8192, 4, 256, 2560
    selected = torch.rand(num_tokens, num_experts, device="cuda").topk(top_k).indices
    routing = TokenReorderer(num_experts, top_k)(torch.ones(num_tokens, top_k, device="cuda"), selected)
    scale = torch.exp(2.0 * torch.randn(num_tokens * top_k, hidden, device="cuda"))
    rows = (torch.randn(num_tokens * top_k, hidden, device="cuda") * scale).to(torch.bfloat16)

    first = combine_routed_rows(rows, routing.token_indices, num_tokens, top_k)
    # The contender lives entirely on the side stream: allocated there, consumed there, joined at
    # the end, so it never races the default stream's allocator.
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        contender = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    for _ in range(20):
        with torch.cuda.stream(side):
            contender = (contender @ contender).clamp_(-1, 1)
        again = combine_routed_rows(rows, routing.token_indices, num_tokens, top_k)
        assert torch.equal(again, first), "the combine changed between launches"
    torch.cuda.current_stream().wait_stream(side)
